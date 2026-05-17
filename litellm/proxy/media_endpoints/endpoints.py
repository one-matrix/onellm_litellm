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

import re
import traceback
from typing import Any, Dict, List

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.route_llm_request import route_request

router = APIRouter()

# Trailing "-tierN" (N integer) marks a per-channel-group variant of the
# same logical model. The base name is what callers send in /v1/media/generations.
_TIER_SUFFIX_RE = re.compile(r"-tier\d+$")


def _strip_tier(model_name: str) -> str:
    """Drop the ``-tierN`` suffix so deployments can be grouped by base name."""
    return _TIER_SUFFIX_RE.sub("", model_name)


def _is_media_deployment(deployment: Dict[str, Any]) -> bool:
    """A deployment counts as 'media' iff ``model_info.ai6700_media_type`` is set.

    We deliberately do not fall back to ``mode`` here — the media endpoints
    are AI6700-shaped (poll-then-URL), and ``mode`` alone (e.g. ``image_generation``)
    doesn't guarantee the deployment is on that async-poll surface.
    """
    info = deployment.get("model_info") or {}
    return bool(info.get("ai6700_media_type"))


def _list_media_deployments() -> List[Dict[str, Any]]:
    """All media-capable deployments from the router. Empty list if no router."""
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return []
    deployments = llm_router.get_model_list() or []
    return [d for d in deployments if _is_media_deployment(d)]


def _channel_view(deployment: Dict[str, Any]) -> Dict[str, Any]:
    """Per-channel pricing view (one row per registered tier deployment)."""
    info = deployment.get("model_info") or {}
    return {
        "name": deployment.get("model_name"),
        "channel_group": info.get("channel_group"),
        "billing_method": info.get("billing_method"),
        "is_active": info.get("is_active"),
        "base_price": info.get("base_price"),
        "cost_per_second": info.get("cost_per_second"),
        "input_token_price": info.get("input_token_price"),
        "output_token_price": info.get("output_token_price"),
        "option_prices": info.get("option_prices"),
        "price_markup": info.get("price_markup"),
        "success_rate_24h": info.get("success_rate_24h"),
        "avg_response_seconds": info.get("avg_response_seconds"),
    }


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
        llm_router,
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
        # Two-stage alias resolution — mirrors chat/completions
        # (common_request_processing.common_processing_pre_call_logic):
        #   1. global litellm.model_alias_map
        #   2. per-key aliases from the caller's API key
        # The router's model_name → litellm_params.model translation runs
        # afterwards inside route_request.
        if resolved_model in litellm.model_alias_map:
            resolved_model = litellm.model_alias_map[resolved_model]
        key_aliases = getattr(user_api_key_dict, "aliases", None)
        if isinstance(key_aliases, dict) and resolved_model in key_aliases:
            resolved_model = key_aliases[resolved_model]
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

        # Route through llm_router so ``model_name`` aliases in proxy config
        # (e.g. ``kwvideo-v2`` → ``ai6700/kwvideo-v2``) resolve to the
        # provider-prefixed model string that ``amedia_generation`` requires.
        llm_call = await route_request(
            data=data,
            route_type="amedia_generation",
            llm_router=llm_router,
            user_model=user_model,
            user_api_key_dict=user_api_key_dict,
        )
        response = await llm_call

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


@router.get(
    "/v1/media/models",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
)
async def media_models_list(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List media models registered on this proxy, deduplicated by base name.

    Source of truth is the local ``model_list`` (config.yaml or DB). Tier
    deployments (``foo``, ``foo-tier1``, ``foo-tier2``...) collapse into one
    entry per base name; ``channel_count`` exposes how many are registered.
    """
    deployments = _list_media_deployments()

    by_base: Dict[str, Dict[str, Any]] = {}
    for dep in deployments:
        name = dep.get("model_name") or ""
        base = _strip_tier(name)
        info = dep.get("model_info") or {}
        entry = by_base.setdefault(
            base,
            {
                "name": base,
                "display_name": info.get("display_name"),
                "type": info.get("ai6700_media_type"),
                "mode": info.get("mode"),
                "channel_count": 0,
            },
        )
        # Prefer the bare-name deployment's display_name/mode if it exists.
        if name == base:
            entry["display_name"] = info.get("display_name") or entry["display_name"]
            entry["mode"] = info.get("mode") or entry["mode"]
        entry["channel_count"] += 1

    data = sorted(by_base.values(), key=lambda e: e["name"])
    return {"object": "list", "data": data}


@router.get(
    "/v1/media/models/{model}/pricing",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
)
async def media_model_pricing(
    model: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """All registered tier deployments for ``model`` with their pricing fields.

    404s if no deployment with that base name is registered as a media model.
    """
    deployments = _list_media_deployments()
    matching = [
        d for d in deployments if _strip_tier(d.get("model_name") or "") == model
    ]
    if not matching:
        raise HTTPException(
            status_code=404,
            detail=f"media model {model!r} not registered on this proxy",
        )

    # Sort: bare name first, then tier1, tier2... lexicographically.
    matching.sort(
        key=lambda d: (d.get("model_name") != model, d.get("model_name") or "")
    )
    info = matching[0].get("model_info") or {}
    return {
        "model": model,
        "type": info.get("ai6700_media_type"),
        "channels": [_channel_view(d) for d in matching],
    }


@router.get(
    "/v1/media/models/{model}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
)
async def media_model_detail(
    model: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Single media model's metadata from the local registry.

    Returns the bare-name deployment if present, otherwise the first tier
    found. 404 if no registered media model matches.
    """
    deployments = _list_media_deployments()
    bare = next((d for d in deployments if (d.get("model_name") or "") == model), None)
    tiered = next(
        (d for d in deployments if _strip_tier(d.get("model_name") or "") == model),
        None,
    )
    deployment = bare or tiered
    if deployment is None:
        raise HTTPException(
            status_code=404,
            detail=f"media model {model!r} not registered on this proxy",
        )

    info = dict(deployment.get("model_info") or {})
    params = deployment.get("litellm_params") or {}
    info["name"] = model
    info["type"] = info.get("ai6700_media_type")
    info["litellm_model"] = params.get("model")
    return info
