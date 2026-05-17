"""Unified async-poll media generation endpoint.

Single endpoint that fans out to the right modality (image / video / audio /
TTS / music) based on the model's ``ai6700_media_type`` in model_info. Returns
the litellm-native ``MediaResponse`` shape — not a passthrough of any
upstream provider's wire format.

The endpoint goes through litellm's standard proxy lifecycle
(``user_api_key_auth`` → ``pre_call_hook`` → ``amedia_generation`` →
``post_call_success_hook``), so guardrails, spend logging, alerting, and
response headers all behave the same as ``/v1/images/generations``.
"""

import traceback
from typing import Any, Dict, Optional

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

router = APIRouter()


_AMEDIA_FORWARD_KEYS = {
    "type",
    "params",
    "count",
    "timeout",
    "poll_interval",
    "api_key",
    "api_base",
    "extra_headers",
    "price_markup",
    # openai-style optional params auto-mapped per modality
    "size",
    "seconds",
    "input_reference",
    "image",
    "image_url",
    "voice",
    "speed",
    "response_format",
    "n",
    "num_images",
    "parameters",
    "user",
}


def _extract_kwargs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the subset of request body fields that ``amedia_generation`` understands.

    Unknown keys (e.g. ``aspect_ratio``, ``ratio``, ``model_version``,
    ``emotion``, ``audio_duration``, ...) are deliberately also forwarded —
    they pass through ``config.map_openai_params`` and end up in the AI6700
    ``params`` dict. We only strip framework-internal keys like ``model`` /
    ``prompt`` (already top-level) and ``litellm_*`` plumbing.
    """
    drop = {"model", "prompt", "messages", "metadata"}
    drop |= {k for k in data if k.startswith("litellm_") or k.startswith("proxy_")}
    return {k: v for k, v in data.items() if k not in drop}


@router.post(
    "/v1/media/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
)
@router.post(
    "/media/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
)
async def media_generations(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Submit a media-generation task and synchronously return the final result.

    Request body (JSON)::

        {
          "model": "ai6700/grok-video-3" | "grok-video-3-tier1",
          "prompt": "a cat running in a meadow",
          "type": "video",                   // optional, inferred from model_info
          "params": {"resolution": "720p"},  // optional, AI6700-native
          "count": 1,                        // optional
          "size": "1280x720",                // optional, auto-mapped per modality
          "seconds": 5                       // optional, auto-mapped (video)
        }

    Response: ``MediaResponse`` (see ``litellm.types.media.MediaResponse``).
    """
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        proxy_config,
        proxy_logging_obj,
        user_model,
        version,
    )

    data: Dict[str, Any] = {}
    try:
        body = await request.body()
        data = orjson.loads(body) if body else {}

        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        resolved_model = (
            data.get("model")
            or general_settings.get("media_generation_model")
            or user_model
        )
        if not resolved_model:
            raise HTTPException(
                status_code=400,
                detail="media/generations: 'model' is required",
            )
        if resolved_model in litellm.model_alias_map:
            resolved_model = litellm.model_alias_map[resolved_model]
        data["model"] = resolved_model

        prompt = data.get("prompt")
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="media/generations: 'prompt' is required",
            )

        # Run guardrails via the existing image-generation hook category
        # (guardrails are content-shaped, not API-shaped).
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            data=data,
            call_type="image_generation",
        )

        kwargs = _extract_kwargs(data)
        response = await litellm.amedia_generation(
            model=resolved_model,
            prompt=data["prompt"],
            **kwargs,
        )

        # Standard proxy post-call hook
        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
        )

        # Response headers — same set as image_generation for parity
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id") or ""
        cache_key = hidden_params.get("cache_key") or ""
        api_base = hidden_params.get("api_base") or ""
        response_cost = hidden_params.get("response_cost") or ""
        litellm_call_id = hidden_params.get("litellm_call_id") or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                call_id=litellm_call_id,
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=data,
        )
        verbose_proxy_logger.error(
            "litellm.proxy.media_endpoints.media_generations(): %s", e
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        raise ProxyException(
            message=getattr(e, "message", str(e)),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            openai_code=getattr(e, "code", None),
        )


@router.get(
    "/v1/media/tasks/{task_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
)
async def media_task_status(
    task_id: int,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Poll an existing media task once and return the latest status.

    Useful when a caller submitted a task elsewhere (or wants to manually
    poll instead of using ``/v1/media/generations`` which blocks until done).
    Returns the raw task-status payload as reported by the upstream provider.
    """
    from litellm.llms.ai6700.common_utils import AI6700Helper
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.media.main import _resolve_api_base, _resolve_api_key

    api_key = _resolve_api_key(None)
    api_base = _resolve_api_base(None)

    client = AsyncHTTPHandler(timeout=30)
    try:
        return await AI6700Helper.aget_status(
            client,
            api_base=api_base,
            api_key=api_key,
            task_id=task_id,
        )
    finally:
        try:
            await client.close()
        except Exception:
            pass
