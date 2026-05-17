"""AI6700 video generation transformation.

Owns *shape* conversions only:
  - OpenAI-style video params (size/seconds/input_reference/parameters)
    → AI6700 ``params`` dict for ``POST /v1/media/generate``
  - AI6700 ``task-status`` JSON → litellm ``VideoObject`` + real ``cost``

The submit+poll loop lives in ``AI6700Helper`` (common_utils.py); the handler
in ``litellm.media.main`` / ``litellm.videos.main`` glues them together.

We deliberately do NOT inherit from ``BaseVideoConfig`` here: that base class
forces 15+ abstract methods (remix/list/delete/character/edit/extension)
which AI6700 does not support, and its single-shot create+response shape
doesn't fit a poll-until-done provider cleanly. AI6700VideoConfig is consumed
by the ai6700-aware handler directly.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.llms.ai6700.common_utils import AI6700Error, AI6700Helper, AI6700TaskStatus
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject

# OpenAI-style "WIDTHxHEIGHT" → AI6700 resolution shorthand.
# Anything we cannot recognise is passed through unchanged for the platform
# to validate (or reject).
_SIZE_TO_RESOLUTION: Dict[str, str] = {
    "640x480": "480p",
    "854x480": "480p",
    "1280x720": "720p",
    "1920x1080": "1080p",
    "3840x2160": "4K",
}

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

_DEFAULT_SUPPORTED_OPENAI_PARAMS: List[str] = [
    "input_reference",
    "image",
    "parameters",
    "seconds",
    "size",
    "user",
    "extra_headers",
    "extra_body",
]


class AI6700VideoConfig:
    """Shape converter between litellm's video API and AI6700 /v1/media/generate."""

    # ------------------------------------------------------------------
    # Header / param surface
    # ------------------------------------------------------------------

    def get_supported_openai_params(self, model: str) -> List[str]:
        return list(_DEFAULT_SUPPORTED_OPENAI_PARAMS)

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
                message="ai6700 video: api_key is required (set LINGKE_API_KEY)",
            )
        merged = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if headers:
            merged.update(headers)
            # User-provided Authorization wins only if explicitly set.
            if "Authorization" not in headers:
                merged["Authorization"] = f"Bearer {api_key}"
        return merged

    # ------------------------------------------------------------------
    # Request shaping
    # ------------------------------------------------------------------

    @staticmethod
    def strip_provider_prefix(model: str) -> str:
        """``ai6700/doubao-seedance-...`` → ``doubao-seedance-...``."""
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    @staticmethod
    def _coerce_size_to_resolution(size: Any) -> Optional[str]:
        if not isinstance(size, str):
            return None
        size = size.strip()
        if not size:
            return None
        # Already AI6700-shaped (e.g. "720p" / "1080p" / "4K") → pass through.
        if re.match(r"^\d+p$", size, flags=re.IGNORECASE) or size.lower() in {
            "4k",
            "8k",
        }:
            return size
        normalized = size.lower().replace(" ", "")
        return _SIZE_TO_RESOLUTION.get(normalized, size)

    @staticmethod
    def _coerce_duration(seconds: Any) -> Optional[str]:
        if seconds is None:
            return None
        try:
            # Accept "5", "5s", 5, 5.0 — AI6700 wants the bare numeric string
            # because select-type params compare against ``value`` exactly.
            if isinstance(seconds, str):
                s = seconds.strip().rstrip("sS")
                num = float(s)
            else:
                num = float(seconds)
        except (TypeError, ValueError):
            return None
        # AI6700 select values are typically integer-shaped strings ("4","5","8","12").
        if num.is_integer():
            return str(int(num))
        return str(num)

    @staticmethod
    def _validate_upload_value(param_name: str, value: Any) -> None:
        """``type=upload`` must be public URL(s). Reject base64/local paths early."""
        if isinstance(value, str):
            if not _URL_RE.match(value):
                raise AI6700Error(
                    status_code=400,
                    message=(
                        f"ai6700 video: param {param_name!r} must be an http(s):// URL "
                        f"(got {value[:80]!r}). AI6700 does not host files — upload "
                        f"to your own object storage and pass the public URL."
                    ),
                )
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                AI6700VideoConfig._validate_upload_value(f"{param_name}[{idx}]", item)
        else:
            raise AI6700Error(
                status_code=400,
                message=(
                    f"ai6700 video: param {param_name!r} must be a URL string or "
                    f"list of URL strings (got {type(value).__name__})."
                ),
            )

    # OpenAI "input_reference"/"image" → AI6700 "images" (URL or list of URLs).
    _UPLOAD_TARGET_PARAM = "images"

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool = False,
    ) -> Dict[str, Any]:
        """Map OpenAI-style optional params into AI6700's flat ``params`` dict.

        Standard mappings:
          - ``parameters`` (dict)     → merged verbatim (caller-provided)
          - ``size``                  → ``params.resolution``
          - ``seconds``               → ``params.audio_duration``
          - ``input_reference``/``image`` → ``params.images``

        Anything else under ``video_create_optional_params`` that is not an
        OpenAI standard surface is passed through as a sibling under
        ``params`` so callers can target model-specific knobs like
        ``ratio`` / ``generate_audio`` directly without going through
        ``parameters=``.
        """
        params: Dict[str, Any] = {}

        # 1. caller-supplied dict wins as the base (so explicit AI6700 fields override mapping)
        parameters = (
            video_create_optional_params.get("parameters")
            if isinstance(video_create_optional_params, dict)
            else None
        )
        if isinstance(parameters, dict):
            params.update(parameters)

        # 2. size → resolution
        size = video_create_optional_params.get("size")
        if size:
            resolution = self._coerce_size_to_resolution(size)
            if resolution and "resolution" not in params:
                params["resolution"] = resolution

        # 3. seconds → audio_duration (the canonical name on most AI6700 video models)
        seconds = video_create_optional_params.get("seconds")
        if seconds is not None:
            duration = self._coerce_duration(seconds)
            if duration and "audio_duration" not in params and "duration" not in params:
                params["audio_duration"] = duration

        # 4. input_reference / image → images
        for src_key in ("input_reference", "image"):
            ref = video_create_optional_params.get(src_key)
            if ref is None:
                continue
            if self._UPLOAD_TARGET_PARAM in params:
                continue
            params[self._UPLOAD_TARGET_PARAM] = ref

        # 5. validate any upload-shaped params we know about
        for upload_key in (
            "images",
            "image_url",
            "first_frame",
            "last_frame",
            "reference",
        ):
            if upload_key in params and params[upload_key] is not None:
                self._validate_upload_value(upload_key, params[upload_key])

        # 6. pass through any other unknown keys directly under top-level
        #    (so users can do video_create_optional_params={"ratio": "16:9"})
        standard_openai = {
            "input_reference",
            "image",
            "parameters",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "extra_body",
            "model",
        }
        for key, value in video_create_optional_params.items():
            if key in standard_openai or value is None:
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
        """Produce the JSON body for ``POST /v1/media/generate``."""
        if not prompt:
            raise AI6700Error(
                status_code=400,
                message="ai6700 video: prompt is required",
            )
        body: Dict[str, Any] = {
            "model": self.strip_provider_prefix(model),
            "prompt": prompt,
        }
        if params:
            body["params"] = params
        if count is not None and count != 1:
            body["count"] = int(count)
        verbose_logger.debug("ai6700 video submit body: %s", body)
        return body

    # ------------------------------------------------------------------
    # Response shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _progress_to_percent(progress: Any) -> Optional[int]:
        if progress is None:
            return None
        if isinstance(progress, (int, float)):
            return int(progress)
        if isinstance(progress, str):
            s = progress.strip().rstrip("%")
            try:
                return int(float(s))
            except (TypeError, ValueError):
                return None
        return None

    def transform_task_to_video_object(
        self,
        *,
        status: AI6700TaskStatus,
        model: str,
        price_markup: float = 1.0,
        extra_hidden: Optional[Dict[str, Any]] = None,
    ) -> VideoObject:
        """AI6700 terminal task-status → litellm VideoObject.

        Caller is expected to have ensured ``status`` is the *finished*
        (is_final=true) payload — ``AI6700Helper._check_task_outcome`` does
        this. We still defensively validate ``result_url``.
        """
        if not status.get("result_url"):
            raise AI6700Error(
                status_code=502,
                message=f"ai6700 video: task {status.get('task_id')} missing result_url",
            )

        bare_model = self.strip_provider_prefix(model)
        video = VideoObject(
            id=str(status.get("task_id")),
            object="video",
            status="completed",
            created_at=AI6700Helper.iso_to_unix(status.get("created_at")),
            completed_at=AI6700Helper.iso_to_unix(status.get("completed_at")),
            progress=self._progress_to_percent(status.get("progress")) or 100,
            model=bare_model,
        )

        # Pydantic v2: private attrs can be set via __dict__ assignment.
        video.__dict__["_hidden_params"] = AI6700Helper.build_hidden_params(
            status=status,
            bare_model=bare_model,
            price_markup=price_markup,
            extra=extra_hidden,
        )
        return video
