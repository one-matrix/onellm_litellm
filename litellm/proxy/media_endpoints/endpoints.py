"""Media generation endpoints — async submit (default) + sync wrapper.

Two POST surfaces share one resolution / billing pipeline:

  - ``POST /v1/media/generations``       async submit. Returns 202 with the
                                         upstream task_id; client polls
                                         ``/v1/media/tasks/{task_id}`` for
                                         result. Credits are frozen on the
                                         caller's wallet at submit time.
  - ``POST /v1/media/generations/sync``  blocking submit + poll + settle.
                                         Returns the final ``MediaResponse``.
                                         Same pre-deduct safety as async.

  - ``GET  /v1/media/tasks/{task_id}``   single-shot status query. When the
                                         upstream task is final, this also
                                         drives the wallet ``settle`` (on
                                         success) or ``refund`` (on failure).

Both POST paths fan out to the right modality (image / video / audio / TTS /
music) based on ``model_info.ai6700_media_type`` and return / wrap into the
litellm-native ``MediaResponse`` shape.
"""

import re
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import BudgetExceededError
from litellm.llms.ai6700.common_utils import (
    AI6700_DEFAULT_POLL_INTERVAL,
    AI6700_DEFAULT_TIMEOUT,
    AI6700Helper,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.media.main import (
    MediaSubmitResult,
    _resolve_api_base,
    _resolve_api_key,
    asubmit_media_task,
    build_media_response_from_status,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.credit_service import (
    EstimateResult,
    WalletError,
    estimate_media_credits,
    pre_deduct,
    rebind_agent_record_id,
    refund,
    settle,
)

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


# ----------------------------------------------------------------------
# Shared submit pipeline (used by both async and sync POST routes)
# ----------------------------------------------------------------------


def _resolve_tenant_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    """Wallet is keyed by team_id, fall back to user_id if no team.

    One of the two must be present — anonymous keys can't be billed.
    """
    tenant_id = user_api_key_dict.team_id or user_api_key_dict.user_id
    if not tenant_id:
        raise HTTPException(
            status_code=401,
            detail=(
                "media/generations: API key has no team_id or user_id — "
                "cannot bill credits."
            ),
        )
    return tenant_id


async def _prepare_request_body(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
) -> Tuple[Dict[str, Any], str, str]:
    """Parse body, run aliases + guardrails, return ``(data, model, prompt)``.

    Mirrors the resolution sequence chat/completions uses — model_alias_map
    then per-key aliases — so downstream router lookup sees the canonical
    model_group_name.
    """
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        proxy_config,
        proxy_logging_obj,
        user_model,
        version,
    )

    body = await request.body()
    data: Dict[str, Any] = orjson.loads(body) if body else {}

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

    data = await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        data=data,
        call_type="image_generation",
    )

    return data, resolved_model, prompt


_PLACEHOLDER_PREFIX = "pending:"


def _make_placeholder() -> str:
    return f"{_PLACEHOLDER_PREFIX}{uuid.uuid4().hex}"


async def _route_for_submission(
    data: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Run router tier selection → ``(resolved_model_str, forward_kwargs)``.

    Mirrors :func:`Router._amedia_generation` up to (but not including) the
    actual ``litellm.amedia_generation`` call. The returned ``forward_kwargs``
    is the same dict shape that ``litellm.amedia_generation`` and
    :func:`asubmit_media_task` accept.

    When no router is configured (``user_model`` setup), falls back to the
    request's ``model`` field with a minimal kwargs passthrough.
    """
    from litellm.proxy.proxy_server import llm_router
    from litellm.router import _build_amedia_forward_kwargs

    request_model = data["model"]
    request_kwargs = {k: v for k, v in data.items() if k not in ("model", "prompt")}

    if llm_router is None:
        return request_model, request_kwargs

    deployment = await llm_router.async_get_available_deployment(
        model=request_model,
        messages=[{"role": "user", "content": data.get("prompt", "")}],
        specific_deployment=request_kwargs.pop("specific_deployment", None),
        request_kwargs=request_kwargs,
    )
    deployment_params = deployment["litellm_params"]
    resolved_model = deployment_params["model"]
    forward_kwargs = _build_amedia_forward_kwargs(
        deployment_params=deployment_params,
        deployment_model_info=deployment.get("model_info") or {},
        request_kwargs=request_kwargs,
    )
    return resolved_model, forward_kwargs


async def _freeze_and_submit(
    *,
    data: Dict[str, Any],
    prompt: str,
    tenant_id: str,
) -> Tuple[MediaSubmitResult, EstimateResult, str]:
    """Route → submit → estimate → freeze → rebind to upstream task_id.

    Order rationale: routing + submit happen first so we resolve the actual
    deployment (tier) and obtain the task_id. We freeze AFTER submit because
    the agent_record_id must equal the upstream task_id. If freeze fails
    (insufficient balance), we log the orphan upstream task — caller still
    receives the budget error, the upstream task continues unattended (a
    future reaper job is the right cleanup).
    """
    resolved_model, forward_kwargs = await _route_for_submission(data)

    # asubmit_media_task signature accepts `type`, `params`, `count` as
    # explicit kwargs and the rest as openai-style optional params.
    explicit_type = forward_kwargs.pop("type", data.get("type"))
    explicit_params = forward_kwargs.pop("params", data.get("params"))
    explicit_count = forward_kwargs.pop("count", data.get("count"))

    submit_result = await asubmit_media_task(
        model=resolved_model,
        prompt=prompt,
        type=explicit_type,
        params=explicit_params,
        count=explicit_count,
        **forward_kwargs,
    )

    estimate = estimate_media_credits(
        model_info=submit_result.model_info or {},
        media_type=submit_result.media_type,
        ai6700_params=submit_result.body.get("params") or {},
        count=submit_result.body.get("count"),
    )

    placeholder = _make_placeholder()
    try:
        await pre_deduct(
            tenant_id=tenant_id,
            estimated_credits=estimate.estimated_credits,
            agent_record_id=placeholder,
            model_name=submit_result.bare_model,
            description=(
                f"media task submit: model={submit_result.bare_model} "
                f"type={submit_result.media_type} "
                f"qty={estimate.quantity:.4f} base={estimate.base_price:.4f}"
            ),
        )
    except BudgetExceededError:
        verbose_proxy_logger.warning(
            "media/generations: budget exceeded AFTER upstream submit; "
            "orphan ai6700 task_id=%s tenant=%s estimated=%.4f",
            submit_result.task_id,
            tenant_id,
            estimate.estimated_credits,
        )
        raise

    await rebind_agent_record_id(
        placeholder=placeholder, real_id=str(submit_result.task_id)
    )
    return submit_result, estimate, placeholder


def _build_accepted_response(
    *,
    submit_result: MediaSubmitResult,
    estimate: EstimateResult,
    tenant_id: str,
) -> Dict[str, Any]:
    """202 body returned from the async POST."""
    return {
        "object": "media.task",
        "task_id": str(submit_result.task_id),
        "model": submit_result.bare_model,
        "media_type": submit_result.media_type,
        "status": "pending",
        "estimated_cost": estimate.estimated_credits,
        "price_markup": estimate.markup,
        "billing_method": estimate.billing_method,
        "tenant_id": tenant_id,
        "created": int(time.time()),
        "provider": "ai6700",
    }


# ----------------------------------------------------------------------
# POST /v1/media/generations  — async submit (default)
# ----------------------------------------------------------------------


@router.post(
    "/v1/media/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
    status_code=status.HTTP_202_ACCEPTED,
)
@router.post(
    "/media/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
    status_code=status.HTTP_202_ACCEPTED,
)
async def media_generations(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Submit a media-generation task, freeze credits, return 202 + task_id.

    Behaviour summary:

      1. parse body, resolve aliases, run guardrails
      2. submit task to upstream AI6700 → ``task_id``
      3. estimate credit cost from ``model_info`` + post-mapping AI6700 params
      4. freeze that amount on the caller's wallet
         (raises ``BudgetExceededError`` → HTTP 429 if insufficient)
      5. return 202 with ``task_id`` + ``estimated_cost``

    Clients poll ``GET /v1/media/tasks/{task_id}`` to retrieve the result.
    When that endpoint sees ``is_final=true``, it drives the wallet
    ``settle`` (or ``refund`` on failure).

    For callers that want the old "block until done" behaviour, use
    ``POST /v1/media/generations/sync`` instead.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj

    data: Dict[str, Any] = {}
    try:
        data, _model, prompt = await _prepare_request_body(request, user_api_key_dict)
        tenant_id = _resolve_tenant_id(user_api_key_dict)

        submit_result, estimate, _placeholder = await _freeze_and_submit(
            data=data,
            prompt=prompt,
            tenant_id=tenant_id,
        )

        fastapi_response.status_code = status.HTTP_202_ACCEPTED
        return _build_accepted_response(
            submit_result=submit_result,
            estimate=estimate,
            tenant_id=tenant_id,
        )

    except HTTPException:
        raise
    except BudgetExceededError as e:
        # Surface budget failures as 402 (Payment Required) so clients can
        # distinguish "out of credits" from generic 4xx.
        raise ProxyException(
            message=getattr(e, "message", str(e)),
            type="BudgetExceededError",
            param="tenant_balance",
            code=402,
        )
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


# ----------------------------------------------------------------------
# POST /v1/media/generations/sync  — submit + poll + settle in one call
# ----------------------------------------------------------------------


@router.post(
    "/v1/media/generations/sync",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
)
@router.post(
    "/media/generations/sync",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["media"],
)
async def media_generations_sync(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Submit and block until the task completes — returns full ``MediaResponse``.

    Wraps the same freeze pipeline as the async POST, then polls the
    upstream until ``is_final=true``. On success the frozen amount is
    settled to the real cost; on failure / poll timeout it's refunded.

    Use this only for short tasks (image, short audio). Video generation
    can take 10+ minutes — prefer the default async POST for those to
    avoid HTTP timeouts in client / load balancer.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        proxy_logging_obj,
        version,
    )

    data: Dict[str, Any] = {}
    try:
        data, _model, prompt = await _prepare_request_body(request, user_api_key_dict)
        tenant_id = _resolve_tenant_id(user_api_key_dict)

        submit_result, estimate, _placeholder = await _freeze_and_submit(
            data=data,
            prompt=prompt,
            tenant_id=tenant_id,
        )
        task_id_str = str(submit_result.task_id)

        timeout = float(data.get("timeout") or AI6700_DEFAULT_TIMEOUT)
        poll_interval = float(data.get("poll_interval") or AI6700_DEFAULT_POLL_INTERVAL)

        client = AsyncHTTPHandler(timeout=timeout)
        try:
            task_status = await AI6700Helper.apoll(
                client,
                api_base=submit_result.api_base,
                api_key=submit_result.api_key,
                task_id=submit_result.task_id,
                timeout=timeout,
                poll_interval=poll_interval,
            )
        except Exception as poll_exc:
            await refund(
                tenant_id=tenant_id,
                agent_record_id=task_id_str,
                reason=f"poll failed: {poll_exc}",
            )
            raise
        finally:
            try:
                await client.close()
            except Exception:
                pass

        response = build_media_response_from_status(
            status=task_status,
            bare_model=submit_result.bare_model,
            media_type=submit_result.media_type,
            price_markup=submit_result.price_markup,
        )
        await settle(
            tenant_id=tenant_id,
            agent_record_id=task_id_str,
            actual_credits=float(response.cost or 0),
        )

        # Mirror the old endpoint's response headers for parity
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=hidden_params.get("model_id") or "",
                cache_key=hidden_params.get("cache_key") or "",
                api_base=hidden_params.get("api_base") or submit_result.api_base,
                version=version,
                response_cost=hidden_params.get("response_cost") or response.cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                call_id=hidden_params.get("litellm_call_id") or "",
                request_data=data,
                hidden_params=hidden_params,
            )
        )
        return response

    except HTTPException:
        raise
    except BudgetExceededError as e:
        raise ProxyException(
            message=getattr(e, "message", str(e)),
            type="BudgetExceededError",
            param="tenant_balance",
            code=402,
        )
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=data,
        )
        verbose_proxy_logger.error(
            "litellm.proxy.media_endpoints.media_generations_sync(): %s", e
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        raise ProxyException(
            message=getattr(e, "message", str(e)),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            openai_code=getattr(e, "code", None),
        )


# ----------------------------------------------------------------------
# GET /v1/media/tasks/{task_id} — status + on-completion settlement
# ----------------------------------------------------------------------


def _looks_terminal(status_dict: Dict[str, Any]) -> bool:
    if status_dict.get("is_final"):
        return True
    return status_dict.get("status_group") in {"已完成", "失败"}


def _is_failure(status_dict: Dict[str, Any]) -> bool:
    return status_dict.get("status_group") == "失败" or (
        status_dict.get("error") and not status_dict.get("result_url")
    )


async def _lookup_pre_deduct_context(
    task_id_str: str,
) -> Optional[Dict[str, Any]]:
    """Find the pre_deduct row for this task to recover tenant_id + model_name.

    Returns ``{tenant_id, model_name, estimated_amount}`` or ``None`` if the
    task wasn't submitted through this proxy (or its pre_deduct row was
    already cleaned up).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None
    rows = await prisma_client.db.credittransaction.find_many(
        where={"agent_record_id": task_id_str, "tx_type": "pre_deduct"},
        order={"created_at": "desc"},
        take=1,
    )
    if not rows:
        return None
    row = rows[0]
    row_dict = row.model_dump() if hasattr(row, "model_dump") else dict(row)
    return {
        "tenant_id": row_dict.get("tenant_id"),
        "model_name": row_dict.get("model_name"),
        "estimated_amount": abs(float(row_dict.get("amount") or 0)),
    }


def _price_markup_for(model_name: Optional[str]) -> float:
    """Best-effort markup lookup from router model_list for a bare model name."""
    if not model_name:
        return 1.0
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return 1.0
    for dep in llm_router.get_model_list() or []:
        if (dep.get("model_name") or "").rstrip() == model_name:
            info = dep.get("model_info") or {}
            try:
                return float(info.get("price_markup") or 1.0)
            except (TypeError, ValueError):
                return 1.0
    return 1.0


def _media_type_for(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return None
    for dep in llm_router.get_model_list() or []:
        if (dep.get("model_name") or "").rstrip() == model_name:
            info = dep.get("model_info") or {}
            return info.get("ai6700_media_type")
    return None


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
    """Single-shot poll + settle when terminal.

    Behaviour:

      - Always queries the upstream once (no polling loop).
      - If the upstream task is non-terminal, returns the raw status verbatim.
      - If terminal: looks up the originating pre_deduct (by task_id) to
        recover ``tenant_id``, then settles (success) or refunds (failure).
        Settlement is idempotent — re-querying after settle is a no-op
        wallet-wise.

      - On success the response is enriched into a full ``MediaResponse``
        (URL list, cost, raw_cost, channel_group, etc.) so the client
        doesn't need to do its own conversion.
    """
    api_key = _resolve_api_key(None)
    api_base = _resolve_api_base(None)

    client = AsyncHTTPHandler(timeout=30)
    try:
        task_status = await AI6700Helper.aget_status(
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

    task_id_str = str(task_id)
    status_dict = dict(task_status)

    if not _looks_terminal(status_dict):
        # Non-terminal — return as-is so client knows to keep polling.
        return status_dict

    context = await _lookup_pre_deduct_context(task_id_str)
    tenant_id = context.get("tenant_id") if context else None
    model_name = context.get("model_name") if context else None
    media_type = _media_type_for(model_name) or status_dict.get("result_type")
    price_markup = _price_markup_for(model_name)

    if _is_failure(status_dict):
        if tenant_id:
            try:
                await refund(
                    tenant_id=tenant_id,
                    agent_record_id=task_id_str,
                    reason=(
                        f"task {task_id} failed: "
                        f"{status_dict.get('error') or status_dict.get('status')}"
                    ),
                )
            except WalletError as exc:
                verbose_proxy_logger.warning(
                    "media/tasks/%s refund skipped: %s", task_id, exc
                )
        status_dict["settled"] = False
        status_dict["refunded"] = True
        return status_dict

    # Success path — settle then build enriched response
    if tenant_id:
        try:
            raw_cost = float(status_dict.get("cost") or 0)
            billed = raw_cost * price_markup
            await settle(
                tenant_id=tenant_id,
                agent_record_id=task_id_str,
                actual_credits=billed,
            )
        except WalletError as exc:
            verbose_proxy_logger.warning(
                "media/tasks/%s settle skipped: %s", task_id, exc
            )

    try:
        response = build_media_response_from_status(
            status=task_status,
            bare_model=(model_name or status_dict.get("model") or ""),
            media_type=(media_type or "image"),
            price_markup=price_markup,
        )
    except Exception:
        # Builder is strict (e.g. requires result_url) — fall back to raw on
        # any shape mismatch so the client still gets the upstream payload.
        verbose_proxy_logger.debug(
            "media/tasks/%s response builder fell back to raw status", task_id
        )
        status_dict["settled"] = True
        return status_dict
    response_dict = response.model_dump()
    response_dict["settled"] = True
    return response_dict


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
