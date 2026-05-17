"""AI6700 audio / TTS / music generation transformation.

AI6700 collapses TTS and music into ``type=audio`` server-side (and the spec
also reserves ``tts`` / ``music`` types for the future). All three modalities
share the same async-poll API and the same model param surface, so this
single config handles them — branching only on ``result_type`` from the
terminal task-status response (``audio`` / ``tts`` / ``music`` / ...).

litellm has no first-class "audio with URL" response type — the standard
``HttpxBinaryResponseContent`` is bytes-shaped, which doesn't fit a
poll-then-URL provider. So we expose a small Pydantic model
``AI6700AudioResponse`` here; the upcoming ``MediaResponse`` (step 9) will
generalize this shape across all media types.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from litellm._logging import verbose_logger
from litellm.llms.ai6700.common_utils import AI6700Error, AI6700Helper, AI6700TaskStatus

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

# OpenAI ``speech`` standard surface — values we recognise and may map.
# Most are passed through unchanged because AI6700 voices/formats are
# model-specific (e.g. doubao has 100+ voices with proprietary IDs that
# don't match OpenAI's alloy/echo/...).
_OPENAI_STANDARD_KEYS = {
    "voice",
    "response_format",
    "speed",
    "input",
    "user",
    "extra_headers",
    "extra_body",
    "parameters",
    "model",
}

# AI6700 ``speech_rate`` is a select param with discrete values
# (-50, -25, 0, 25, 50, 100) corresponding to 0.5x..2.0x playback.
# OpenAI ``speed`` is a continuous float 0.25..4.0 with 1.0 = normal.
_SPEED_TO_SPEECH_RATE = {
    0.5: "-50",
    0.75: "-25",
    1.0: "0",
    1.25: "25",
    1.5: "50",
    2.0: "100",
}


class AI6700AudioResponse(BaseModel):
    """litellm-compatible response object for AI6700 audio/TTS/music tasks.

    Mirrors the shape we'll generalize as ``MediaResponse`` in step 9.
    Always carries:
      - ``url``: public CDN URL of the generated audio file
      - ``task_id`` / ``model`` / ``cost`` / ``raw_cost`` for ops & billing
      - ``_hidden_params['response_cost']`` so the standard litellm cost
        logger picks it up.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    object: Literal["audio.task"] = "audio.task"
    task_id: int
    status: Literal["completed"] = "completed"
    model: str
    url: str
    result_type: Optional[str] = None
    cost: float = 0.0
    raw_cost: float = 0.0
    channel_group: Optional[str] = None
    duration_seconds: Optional[float] = None
    created: Optional[int] = None
    completed_at: Optional[int] = None
    _hidden_params: Dict[str, Any] = {}

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class AI6700AudioConfig:
    """Shape converter between litellm's speech surface and AI6700 /v1/media/generate.

    Used for ``type=audio`` (TTS), and also for future ``type=tts`` / ``type=music``
    — the wire protocol is identical, only ``result_type`` differs on the
    terminal status payload.
    """

    # ------------------------------------------------------------------
    # Surface
    # ------------------------------------------------------------------

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "input",
            "voice",
            "response_format",
            "speed",
            "user",
            "parameters",
            "extra_headers",
            "extra_body",
        ]

    def validate_environment(
        self,
        headers: Dict[str, str],
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        if not api_key:
            raise AI6700Error(
                status_code=401,
                message="ai6700 audio: api_key is required (set LINGKE_API_KEY)",
            )
        merged = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if headers:
            user_has_auth = "Authorization" in headers
            merged.update(headers)
            if not user_has_auth:
                merged["Authorization"] = f"Bearer {api_key}"
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def strip_provider_prefix(model: str) -> str:
        return model.split("/", 1)[1] if "/" in model else model

    @staticmethod
    def _coerce_speed_to_speech_rate(speed: Any) -> Optional[str]:
        """OpenAI ``speed`` (continuous float) → AI6700 ``speech_rate`` (discrete select).

        Snaps to the nearest AI6700-supported value. Returns None for
        unparseable input so callers can choose to drop or pass-through.
        """
        if speed is None:
            return None
        try:
            x = float(speed)
        except (TypeError, ValueError):
            return None
        # Exact match
        if x in _SPEED_TO_SPEECH_RATE:
            return _SPEED_TO_SPEECH_RATE[x]
        # Snap to nearest supported
        choices = sorted(_SPEED_TO_SPEECH_RATE.keys())
        nearest = min(choices, key=lambda c: abs(c - x))
        return _SPEED_TO_SPEECH_RATE[nearest]

    # ------------------------------------------------------------------
    # Request shaping
    # ------------------------------------------------------------------

    def map_openai_params(
        self,
        optional_params: Dict[str, Any],
        model: str,
        drop_params: bool = False,
    ) -> Dict[str, Any]:
        """Map OpenAI-style speech optional params into AI6700 ``params``.

        Pass-through is the dominant strategy — AI6700 audio params
        (voice IDs, emotion, model_version, ...) are model-specific and
        have no clean OpenAI equivalent. Only ``speed`` gets snapped to
        AI6700's discrete ``speech_rate`` ladder; everything else is
        either taken verbatim from ``parameters=`` or passed through
        from unknown top-level keys.

        ``voice`` is treated as a hint: pass through under the same name
        if not present in ``parameters``. AI6700 model may accept ``voice``
        directly (doubao does; gemini-tts doesn't).
        """
        params: Dict[str, Any] = {}

        parameters = optional_params.get("parameters")
        if isinstance(parameters, dict):
            params.update(parameters)

        # OpenAI speed → AI6700 speech_rate (snap to ladder).
        speed = optional_params.get("speed")
        if speed is not None and "speech_rate" not in params:
            speech_rate = self._coerce_speed_to_speech_rate(speed)
            if speech_rate is not None:
                params["speech_rate"] = speech_rate

        # voice: best-effort pass-through.
        voice = optional_params.get("voice")
        if voice is not None and "voice" not in params:
            params["voice"] = voice

        # Pass through unknown keys (e.g. emotion, emotion_scale, model_version).
        for key, value in optional_params.items():
            if key in _OPENAI_STANDARD_KEYS or value is None:
                continue
            if key not in params:
                params[key] = value

        return params

    def build_request_body(
        self,
        *,
        model: str,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        count: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not prompt:
            raise AI6700Error(
                status_code=400,
                message=(
                    "ai6700 audio: prompt (text to synthesize) is required. "
                    "If you're calling via litellm.aspeech(), pass the text "
                    "via the standard 'input' field — it gets routed here as prompt."
                ),
            )
        body: Dict[str, Any] = {
            "model": self.strip_provider_prefix(model),
            "prompt": prompt,
        }
        if params:
            body["params"] = params
        if count is not None and count != 1:
            body["count"] = int(count)
        verbose_logger.debug("ai6700 audio submit body: %s", body)
        return body

    # ------------------------------------------------------------------
    # Response shaping
    # ------------------------------------------------------------------

    def transform_task_to_audio_response(
        self,
        *,
        status: AI6700TaskStatus,
        model: str,
        price_markup: float = 1.0,
        extra_hidden: Optional[Dict[str, Any]] = None,
    ) -> AI6700AudioResponse:
        """AI6700 terminal task-status → AI6700AudioResponse.

        Works for ``type=audio`` (TTS), ``type=tts``, and ``type=music`` —
        ``result_type`` from the task is surfaced on the response so the
        caller can branch if needed.
        """
        result_url = status.get("result_url")
        if not result_url:
            raise AI6700Error(
                status_code=502,
                message=f"ai6700 audio: task {status.get('task_id')} missing result_url",
            )

        raw_cost = float(status.get("cost") or 0)
        billed_cost = raw_cost * (price_markup or 1.0)
        created = AI6700Helper.iso_to_unix(status.get("created_at")) or int(time.time())

        duration = status.get("duration_seconds")
        try:
            duration_val: Optional[float] = (
                float(duration) if duration is not None else None
            )
        except (TypeError, ValueError):
            duration_val = None

        bare_model = self.strip_provider_prefix(model)
        resp = AI6700AudioResponse(
            task_id=int(status.get("task_id") or 0),
            model=bare_model,
            url=result_url,
            result_type=status.get("result_type"),
            cost=billed_cost,
            raw_cost=raw_cost,
            channel_group=status.get("channel_group"),
            duration_seconds=duration_val,
            created=created,
            completed_at=AI6700Helper.iso_to_unix(status.get("completed_at")),
        )

        # Pydantic private attr — assign via __dict__ to bypass model validation.
        resp.__dict__["_hidden_params"] = AI6700Helper.build_hidden_params(
            status=status,
            bare_model=bare_model,
            price_markup=price_markup,
            extra=extra_hidden,
        )
        return resp
