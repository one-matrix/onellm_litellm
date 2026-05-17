"""LiteLLM ``CustomLogger`` that integrates the credit wallet into every request.

Two hooks on one class:

  - ``async_pre_call_hook``      reject the request with HTTP 402 when the
                                 caller's wallet is empty / overdrawn.
                                 Only enforced for tenants who actually
                                 have a wallet row — wallet-less callers
                                 pass through unchanged (opt-in model).

  - ``async_log_success_event``  debit the wallet by the litellm-computed
                                 cost once the response is ready. Idempotent
                                 by ``litellm_call_id`` so retries don't
                                 double-charge.

Together these turn ``app.credit_wallet`` into a real prepaid account for
chat/completion, image, audio, embedding — anything that flows through
litellm's cost-calculator. Long-running async media (video) uses
``pre_deduct`` / ``settle`` directly in the media endpoint and does NOT
route through this callback (those calls write their own ``settlement``
tx, which our ``charge`` idempotency check won't touch).

Register from ``config.yaml``::

    litellm_settings:
      callbacks:
        - litellm.proxy.credit_service.callbacks.WalletChargeLogger

or programmatically::

    import litellm
    from litellm.proxy.credit_service.callbacks import WalletChargeLogger
    litellm.callbacks.append(WalletChargeLogger())

The callback is a no-op when ``prisma_client`` isn't initialised, so it's
safe to enable in environments without the credit_wallet migration applied.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.credit_service.wallet import (
    WalletError,
    charge,
    get_balance,
)


def _resolve_tenant_id(metadata: Dict[str, Any]) -> Optional[str]:
    """team_id wins, fall back to user_id. Matches the endpoint's resolver."""
    team_id = metadata.get("user_api_key_team_id")
    if isinstance(team_id, str) and team_id:
        return team_id
    user_id = metadata.get("user_api_key_user_id")
    if isinstance(user_id, str) and user_id:
        return user_id
    return None


def _extract_charge_args(
    kwargs: Dict[str, Any],
    response_obj: Any,
) -> Optional[Dict[str, Any]]:
    """Pull tenant / cost / call_id / model from the standard logging payload.

    Returns ``None`` when any required field is missing — the callback then
    skips silently rather than crashing the success-callback bus.
    """
    standard = kwargs.get("standard_logging_object") or {}
    metadata = standard.get("metadata") or {}

    tenant_id = _resolve_tenant_id(metadata)
    if not tenant_id:
        return None

    cost_raw = standard.get("response_cost")
    if cost_raw is None:
        # Fallback path for callbacks that fire before the standard payload
        # is fully populated (rare, mostly tests/streaming corner cases).
        hidden = (
            getattr(response_obj, "_hidden_params", None)
            if response_obj is not None
            else None
        ) or {}
        cost_raw = hidden.get("response_cost") or kwargs.get("response_cost")
    try:
        cost = float(cost_raw or 0)
    except (TypeError, ValueError):
        return None
    if cost <= 0:
        return None

    call_id = standard.get("litellm_call_id") or kwargs.get("litellm_call_id")
    model = standard.get("model") or kwargs.get("model")
    call_type = standard.get("call_type") or kwargs.get("call_type") or "unknown"

    return {
        "tenant_id": tenant_id,
        "actual_credits": cost,
        "agent_record_id": str(call_id) if call_id else None,
        "model_name": model,
        "description": f"{call_type}: {model}",
    }


_BALANCE_GATED_CALL_TYPES = frozenset(
    {
        "completion",
        "acompletion",
        "text_completion",
        "atext_completion",
        "embedding",
        "aembedding",
        "image_generation",
        "aimage_generation",
        "image_edit",
        "aimage_edit",
        "speech",
        "aspeech",
        "transcription",
        "atranscription",
        "moderation",
        "amoderation",
        "rerank",
        "arerank",
        "anthropic_messages",
        "generate_content",
        "agenerate_content",
        "generate_content_stream",
        "agenerate_content_stream",
    }
)


class WalletChargeLogger(CustomLogger):
    """Debit the credit wallet whenever litellm reports a successful cost,
    and pre-gate requests when the wallet is empty.

    Pairs with the proxy's existing ``increment_spend_counters`` (which keeps
    populating ``LiteLLM_UserTable.spend`` for reporting). The wallet is the
    *prepaid balance*; litellm's spend counter is the *historical total*.
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Any]:
        """Reject with HTTP 402 when the caller's wallet is empty.

        Skip rules (return ``None`` = allow):
          - ``call_type`` isn't a billable LLM/media call (e.g. /models route)
          - caller has no team_id and no user_id (master key, internal route)
          - no wallet row exists for this tenant — they're not on the prepaid
            plan, fall back to litellm's other budget gates
          - ``available = gift + paid - frozen > 0``

        Raises ``HTTPException(402)`` only when a wallet exists AND its
        available balance has dropped to ≤ 0 (the overspend case left by
        ``charge``'s allow-negative behaviour).
        """
        if call_type not in _BALANCE_GATED_CALL_TYPES:
            return None

        tenant_id = user_api_key_dict.team_id or user_api_key_dict.user_id
        if not tenant_id:
            return None

        try:
            balance = await get_balance(tenant_id)
        except WalletError as exc:
            verbose_proxy_logger.debug(
                "WalletChargeLogger.pre_call: skipped (tenant=%s): %s",
                tenant_id,
                exc,
            )
            return None
        except Exception as exc:
            # Never fail an inbound request because the wallet read crashed.
            verbose_proxy_logger.error(
                "WalletChargeLogger.pre_call: unexpected error reading balance "
                "(tenant=%s): %s",
                tenant_id,
                exc,
            )
            return None

        if balance is None:
            return None  # wallet not provisioned for this tenant — not gated

        if balance["available"] > 0:
            return None

        raise HTTPException(
            status_code=402,
            detail={
                "error": {
                    "message": (
                        f"积分余额不足，请充值后重试："
                        f"available={balance['available']:.4f} C "
                        f"(gift={balance['gift_balance']:.4f}, "
                        f"paid={balance['paid_balance']:.4f}, "
                        f"frozen={balance['frozen_amount']:.4f})"
                    ),
                    "type": "BudgetExceededError",
                    "param": "wallet_balance",
                    "code": 402,
                }
            },
        )

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        args = _extract_charge_args(kwargs, response_obj)
        if args is None:
            return
        try:
            await charge(**args)
        except WalletError as exc:
            # No DB / wallet creation failure — log and move on. We never
            # want a wallet hiccup to fail a successful LLM response.
            verbose_proxy_logger.warning(
                "WalletChargeLogger: skipped charge (tenant=%s call=%s): %s",
                args["tenant_id"],
                args["agent_record_id"],
                exc,
            )
        except Exception as exc:
            verbose_proxy_logger.error(
                "WalletChargeLogger: unexpected error charging tenant=%s call=%s: %s",
                args["tenant_id"],
                args["agent_record_id"],
                exc,
            )

    # We don't override log_success_event (sync); the proxy is async-first
    # and every cost-emitting code path lands in the async callback. Leaving
    # the sync method as no-op (CustomLogger default) keeps tests + sync SDK
    # use safe without double-charging.
