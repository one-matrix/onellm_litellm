"""Top-level entry points for poll-then-URL media generation.

Currently backed by the AI6700 provider only; the function shape is
provider-neutral so additional async-poll gateways can be added later
without changing the public surface.

Usage::

    resp = await litellm.amedia_generation(
        model="ai6700/grok-video-3",
        prompt="一只猫在草地上奔跑",
        params={"resolution": "720p", "audio_duration": "5"},
    )
    print(resp.url, resp.cost)

If the model is registered via ``model_list`` (proxy config), the
``ai6700_media_type`` is auto-detected from ``model_info``. Otherwise
pass ``type="video"`` (or image/audio/tts/music) explicitly.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.ai6700.audio import AI6700AudioConfig
from litellm.llms.ai6700.common_utils import (
    AI6700_DEFAULT_API_BASE,
    AI6700_DEFAULT_POLL_INTERVAL,
    AI6700_DEFAULT_TIMEOUT,
    AI6700Error,
    AI6700Helper,
    AI6700TaskStatus,
)
from litellm.llms.ai6700.image_generation import AI6700ImageConfig
from litellm.llms.ai6700.videos import AI6700VideoConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.media.main import MediaAsset, MediaResponse

__all__ = [
    "amedia_generation",
    "asubmit_media_task",
    "build_media_response_from_status",
    "media_generation",
    "MediaSubmitResult",
]

MediaType = Literal["image", "video", "audio", "tts", "music"]

_AUDIO_LIKE_TYPES = {"audio", "tts", "music"}

# Mode coming from model_info (litellm-standard) → AI6700 media_type.
_MODE_TO_MEDIA_TYPE: Dict[str, str] = {
    "image_generation": "image",
    "video_generation": "video",
    "audio_speech": "audio",
}


# ----------------------------------------------------------------------
# Config dispatch
# ----------------------------------------------------------------------


def _pick_config(media_type: str):
    """Return the (config, label) pair for a given AI6700 media type."""
    if media_type == "video":
        return AI6700VideoConfig(), "video"
    if media_type == "image":
        return AI6700ImageConfig(), "image"
    if media_type in _AUDIO_LIKE_TYPES:
        return AI6700AudioConfig(), media_type
    raise AI6700Error(
        status_code=400,
        message=(
            f"ai6700: unknown media_type {media_type!r} (expected one of "
            f"image / video / audio / tts / music)"
        ),
    )


# ----------------------------------------------------------------------
# Resolution helpers
# ----------------------------------------------------------------------


def _resolve_media_type(
    *,
    model: str,
    explicit_type: Optional[str],
    model_info: Optional[Dict[str, Any]],
) -> str:
    """Determine the AI6700 media type.

    Priority:
      1. explicit ``type=`` kwarg (caller knows best)
      2. ``model_info['ai6700_media_type']`` (set by our gen script)
      3. ``model_info['mode']`` reverse-mapped (covers manually-written configs)
      4. fail loudly with an actionable error
    """
    if explicit_type:
        return explicit_type

    if model_info:
        from_field = model_info.get("ai6700_media_type")
        if from_field:
            return from_field
        mode = model_info.get("mode")
        if mode in _MODE_TO_MEDIA_TYPE:
            return _MODE_TO_MEDIA_TYPE[mode]

    raise AI6700Error(
        status_code=400,
        message=(
            f"ai6700: cannot infer media_type for {model!r}. Either "
            f"register the model in model_list (with model_info.ai6700_media_type) "
            f'or pass type="image|video|audio|tts|music" explicitly.'
        ),
    )


def _resolve_api_key(passed: Optional[str]) -> str:
    """Resolve api_key from arg / env (LINGKE_API_KEY preferred, AI6700_API_KEY fallback)."""
    key = passed or get_secret_str("LINGKE_API_KEY") or get_secret_str("AI6700_API_KEY")
    if not key:
        raise AI6700Error(
            status_code=401,
            message=(
                "ai6700: no api_key found. Set LINGKE_API_KEY (or AI6700_API_KEY) "
                "environment variable, or pass api_key=... explicitly."
            ),
        )
    return key


def _resolve_api_base(passed: Optional[str]) -> str:
    return passed or get_secret_str("AI6700_API_BASE") or AI6700_DEFAULT_API_BASE


def _safe_get_model_info(model: str) -> Optional[Dict[str, Any]]:
    """Best-effort model_info lookup. Returns None if not registered.

    ``get_model_info`` raises for unknown models; that's expected for ad-hoc
    calls so we swallow and let the resolver fall back to ``type=`` arg.
    """
    try:
        return dict(litellm.get_model_info(model=model, custom_llm_provider="ai6700"))
    except Exception:
        return None


# ----------------------------------------------------------------------
# Body building (per-modality)
# ----------------------------------------------------------------------


def _build_body_and_count(
    *,
    config: Any,
    model: str,
    prompt: str,
    params: Optional[Dict[str, Any]],
    count: Optional[int],
    optional_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Map openai-style optional params + caller-supplied params → AI6700 body.

    Precedence (highest to lowest):
      1. caller's explicit ``params=`` dict
      2. mapped fields from openai-style ``optional_params``
    """
    mapped_params = config.map_openai_params(optional_params, model=model)
    if params:
        # caller wins
        merged = {**mapped_params, **params}
    else:
        merged = mapped_params

    # count: explicit > openai n/num_images
    effective_count = count
    if effective_count is None and isinstance(config, AI6700ImageConfig):
        effective_count = config.extract_count(optional_params)

    return config.build_request_body(
        model=model,
        prompt=prompt,
        params=merged or None,
        count=effective_count,
    )


# ----------------------------------------------------------------------
# Response building
# ----------------------------------------------------------------------


def _build_media_response(
    *,
    status: AI6700TaskStatus,
    bare_model: str,
    media_type: str,
    price_markup: float,
) -> MediaResponse:
    urls = AI6700Helper.collect_result_urls(status)
    if not urls:
        raise AI6700Error(
            status_code=502,
            message=f"ai6700: task {status.get('task_id')} missing result_url",
        )

    result_type = status.get("result_type") or media_type
    assets = [MediaAsset(url=u, result_type=result_type) for u in urls]

    raw_cost = float(status.get("cost") or 0)
    billed_cost = raw_cost * (price_markup or 1.0)

    duration = status.get("duration_seconds")
    try:
        duration_val: Optional[float] = (
            float(duration) if duration is not None else None
        )
    except (TypeError, ValueError):
        duration_val = None

    created = AI6700Helper.iso_to_unix(status.get("created_at")) or int(time.time())

    response = MediaResponse(
        task_id=str(status.get("task_id") or ""),
        model=bare_model,
        media_type=media_type,
        data=assets,
        cost=billed_cost,
        raw_cost=raw_cost,
        price_markup=price_markup,
        channel_group=status.get("channel_group"),
        duration_seconds=duration_val,
        created=created,
        completed_at=AI6700Helper.iso_to_unix(status.get("completed_at")),
        provider="ai6700",
        raw_response=dict(status),
    )

    response.__dict__["_hidden_params"] = AI6700Helper.build_hidden_params(
        status=status,
        bare_model=bare_model,
        price_markup=price_markup,
        extra={"media_type": media_type},
    )
    return response


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


async def amedia_generation(
    model: str,
    prompt: str,
    *,
    type: Optional[MediaType] = None,
    params: Optional[Dict[str, Any]] = None,
    count: Optional[int] = None,
    timeout: float = AI6700_DEFAULT_TIMEOUT,
    poll_interval: float = AI6700_DEFAULT_POLL_INTERVAL,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    price_markup: Optional[float] = None,
    client: Optional[AsyncHTTPHandler] = None,
    **kwargs: Any,
) -> MediaResponse:
    """Submit a media-generation task and poll until completion.

    Args:
        model: e.g. ``"ai6700/grok-video-3"`` (with or without provider prefix).
        prompt: text description (required).
        type: explicit media type override; if omitted we look up
            ``model_info.ai6700_media_type`` first, then ``model_info.mode``.
        params: AI6700-native param dict (e.g. ``{"resolution":"720p"}``);
            wins over openai-style auto-mapping.
        count: AI6700 batch count (top-level). For image, falls back to
            ``n`` / ``num_images`` from kwargs.
        timeout / poll_interval: poll loop control.
        api_key / api_base: per-call override; otherwise read from env.
        extra_headers: extra HTTP headers (Authorization is always injected).
        price_markup: per-call override; otherwise read from model_info.
        client: optional pre-built AsyncHTTPHandler (lets tests inject mocks).
        **kwargs: any leftover openai-style optional params (``size``,
            ``seconds``, ``image_url``, ``voice``, ``speed``, ``parameters``,
            etc.) — passed through ``config.map_openai_params``.

    Returns:
        ``MediaResponse`` with the asset URL(s), real cost (× markup),
        provider channel, and the raw task-status under ``raw_response``.
    """
    # 1. provider sanity
    _, custom_llm_provider, _, _ = get_llm_provider(model=model)
    if custom_llm_provider != "ai6700":
        raise AI6700Error(
            status_code=400,
            message=(
                f"amedia_generation: model {model!r} resolves to provider "
                f"{custom_llm_provider!r}, expected 'ai6700'. Use the "
                f"'ai6700/<model_name>' prefix or register via model_list."
            ),
        )

    # 2. model info + media type
    model_info = _safe_get_model_info(model)
    media_type = _resolve_media_type(
        model=model, explicit_type=type, model_info=model_info
    )
    config, _modality_label = _pick_config(media_type)

    # 3. credentials + base + markup
    resolved_api_key = _resolve_api_key(api_key)
    resolved_api_base = _resolve_api_base(api_base)
    effective_markup = (
        price_markup
        if price_markup is not None
        else (model_info or {}).get("price_markup", 1.0)
    )

    # 4. headers
    headers = config.validate_environment(
        headers=dict(extra_headers or {}),
        model=model,
        api_key=resolved_api_key,
        litellm_params=None,
    )

    # 5. body
    body = _build_body_and_count(
        config=config,
        model=model,
        prompt=prompt,
        params=params,
        count=count,
        optional_params=kwargs,
    )

    # 6. submit + poll
    owns_client = False
    if client is None:
        client = AsyncHTTPHandler(timeout=timeout)
        owns_client = True
    try:
        verbose_logger.debug(
            "amedia_generation submitting: model=%s media_type=%s base=%s",
            model,
            media_type,
            resolved_api_base,
        )
        status = await AI6700Helper.asubmit_and_poll(
            client,
            api_base=resolved_api_base,
            api_key=resolved_api_key,
            body=body,
            timeout=timeout,
            poll_interval=poll_interval,
            extra_headers={"Authorization": headers["Authorization"]},
        )
    finally:
        if owns_client:
            try:
                await client.close()
            except Exception:
                pass

    # 7. wrap
    bare_model = config.strip_provider_prefix(model)
    return _build_media_response(
        status=status,
        bare_model=bare_model,
        media_type=media_type,
        price_markup=float(effective_markup or 1.0),
    )


@dataclass
class MediaSubmitResult:
    """Result of a submit-only call (no polling).

    Carries everything a caller needs to either (a) hand the task_id back
    to a client for later polling, or (b) call ``apoll`` + the response
    builder themselves to assemble the final ``MediaResponse``.
    """

    task_id: int
    model: str  # provider-prefixed (e.g. "ai6700/grok-video-3")
    bare_model: str  # provider stripped
    media_type: str
    body: Dict[str, Any] = field(default_factory=dict)  # the JSON sent to AI6700
    price_markup: float = 1.0
    api_base: str = ""
    api_key: str = ""
    model_info: Optional[Dict[str, Any]] = None


async def asubmit_media_task(
    model: str,
    prompt: str,
    *,
    type: Optional[MediaType] = None,
    params: Optional[Dict[str, Any]] = None,
    count: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    price_markup: Optional[float] = None,
    client: Optional[AsyncHTTPHandler] = None,
    **kwargs: Any,
) -> MediaSubmitResult:
    """Submit-only counterpart of :func:`amedia_generation`.

    Goes through the same resolution chain (media_type, config, body
    mapping, credentials) but stops after the ``/v1/media/generate`` POST —
    no polling, no response builder. Use this when the caller wants to
    own the polling cadence (e.g. proxy returns 202 + task_id, clients
    poll via ``GET /v1/media/tasks/{task_id}``).

    All resolution metadata is returned in :class:`MediaSubmitResult` so
    follow-up calls (status check, settlement) don't have to re-resolve.
    """
    _, custom_llm_provider, _, _ = get_llm_provider(model=model)
    if custom_llm_provider != "ai6700":
        raise AI6700Error(
            status_code=400,
            message=(
                f"asubmit_media_task: model {model!r} resolves to provider "
                f"{custom_llm_provider!r}, expected 'ai6700'."
            ),
        )

    model_info = _safe_get_model_info(model)
    media_type = _resolve_media_type(
        model=model, explicit_type=type, model_info=model_info
    )
    config, _modality_label = _pick_config(media_type)

    resolved_api_key = _resolve_api_key(api_key)
    resolved_api_base = _resolve_api_base(api_base)
    effective_markup = (
        price_markup
        if price_markup is not None
        else (model_info or {}).get("price_markup", 1.0)
    )

    headers = config.validate_environment(
        headers=dict(extra_headers or {}),
        model=model,
        api_key=resolved_api_key,
        litellm_params=None,
    )

    body = _build_body_and_count(
        config=config,
        model=model,
        prompt=prompt,
        params=params,
        count=count,
        optional_params=kwargs,
    )

    owns_client = False
    if client is None:
        client = AsyncHTTPHandler(timeout=AI6700_DEFAULT_TIMEOUT)
        owns_client = True
    try:
        task_id = await AI6700Helper.asubmit(
            client,
            api_base=resolved_api_base,
            api_key=resolved_api_key,
            body=body,
            extra_headers={"Authorization": headers["Authorization"]},
        )
    finally:
        if owns_client:
            try:
                await client.close()
            except Exception:
                pass

    return MediaSubmitResult(
        task_id=task_id,
        model=model,
        bare_model=config.strip_provider_prefix(model),
        media_type=media_type,
        body=body,
        price_markup=float(effective_markup or 1.0),
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        model_info=model_info,
    )


def build_media_response_from_status(
    *,
    status: AI6700TaskStatus,
    bare_model: str,
    media_type: str,
    price_markup: float,
) -> MediaResponse:
    """Public re-export of the response builder.

    Lets the proxy's status endpoint reuse the same ``MediaResponse``
    shape that ``amedia_generation`` returns, instead of hand-rolling
    a duplicate construction.
    """
    return _build_media_response(
        status=status,
        bare_model=bare_model,
        media_type=media_type,
        price_markup=price_markup,
    )


def media_generation(
    model: str,
    prompt: str,
    *,
    type: Optional[MediaType] = None,
    params: Optional[Dict[str, Any]] = None,
    count: Optional[int] = None,
    timeout: float = AI6700_DEFAULT_TIMEOUT,
    poll_interval: float = AI6700_DEFAULT_POLL_INTERVAL,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    price_markup: Optional[float] = None,
    **kwargs: Any,
) -> MediaResponse:
    """Sync wrapper around :func:`amedia_generation`.

    Provided for parity with ``image_generation`` / ``video_generation``.
    Uses ``asyncio.run`` and therefore must not be called from inside an
    already-running event loop (use ``amedia_generation`` there).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        raise RuntimeError(
            "media_generation() is a sync entry; call amedia_generation() from "
            "an async context to avoid blocking the event loop."
        )
    return asyncio.run(
        amedia_generation(
            model=model,
            prompt=prompt,
            type=type,
            params=params,
            count=count,
            timeout=timeout,
            poll_interval=poll_interval,
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers,
            price_markup=price_markup,
            **kwargs,
        )
    )
