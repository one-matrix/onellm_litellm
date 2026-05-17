"""Prisma-backed credit wallet — pre_deduct / settle / refund.

State lives in two tables in the **``app`` schema** (separate from litellm
core's ``public`` schema so upstream migrations never collide):

  - ``app.credit_wallet``     one row per tenant_id (= LiteLLM_TeamTable.team_id)
  - ``app.credit_transaction`` append-only log of pre_deduct / consumption /
                                settlement / refund events

Prisma client accessors are ``prisma_client.db.creditwallet`` and
``prisma_client.db.credittransaction`` (no ``LiteLLM_`` prefix — these are
onellm-specific extensions, the naming makes that visible).

Three-step billing:

  1. ``pre_deduct``  freezes ``estimated_credits`` on the wallet and writes a
     ``pre_deduct`` transaction tagged with the upstream task_id (as
     ``agent_record_id``). Raises ``BudgetExceededError`` if the wallet
     can't cover it.

  2. ``settle``      called when the task finishes (or when the status
     endpoint sees ``is_final=true``). Unfreezes the previously frozen
     amount, debits ``actual_credits`` from gift first then paid, and
     writes ``consumption`` + ``settlement`` transactions. Idempotent —
     re-runs are no-ops.

  3. ``refund``      called when the task fails. Unfreezes the previously
     frozen amount and writes a ``refund`` transaction. Idempotent.

All writes happen inside a single ``prisma_client.db.tx()`` so concurrent
requests can't read the same balance and double-spend it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import BudgetExceededError

_PRE_DEDUCT = "pre_deduct"
_CONSUMPTION = "consumption"
_SETTLEMENT = "settlement"
_REFUND = "refund"

_GIFT = "gift"
_PAID = "paid"


class WalletError(Exception):
    """Non-budget wallet failures (missing prisma client, schema drift, etc.)."""


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _get_prisma_client() -> Any:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise WalletError(
            "credit_service: prisma_client is not initialised — "
            "credit wallet requires a database (set DATABASE_URL)."
        )
    return prisma_client


async def _find_wallet(tx: Any, tenant_id: str) -> Optional[Dict[str, Any]]:
    wallet = await tx.creditwallet.find_unique(where={"tenant_id": tenant_id})
    if wallet is None:
        return None
    return wallet.model_dump() if hasattr(wallet, "model_dump") else dict(wallet)


async def _latest_pre_deduct(tx: Any, agent_record_id: str) -> Optional[Dict[str, Any]]:
    rows = await tx.credittransaction.find_many(
        where={"agent_record_id": agent_record_id, "tx_type": _PRE_DEDUCT},
        order={"created_at": "desc"},
        take=1,
    )
    if not rows:
        return None
    row = rows[0]
    return row.model_dump() if hasattr(row, "model_dump") else dict(row)


async def _already_finalised(tx: Any, agent_record_id: str, tx_type: str) -> bool:
    """Idempotency check — has this task already been settled or refunded?"""
    rows = await tx.credittransaction.find_many(
        where={"agent_record_id": agent_record_id, "tx_type": tx_type},
        take=1,
    )
    return bool(rows)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


async def get_balance(tenant_id: str) -> Optional[Dict[str, Any]]:
    """Read-only balance snapshot — used by the pre-call balance check.

    Returns ``None`` when no wallet row exists for the tenant. Callers
    treat ``None`` as "not on the prepaid plan — skip balance gating",
    which keeps the wallet system opt-in: only tenants who actually have
    a wallet row get balance-checked.

    Distinct from :func:`ensure_wallet` (which lazy-creates) so that a
    pre-call hook doesn't auto-create a wallet for every traffic source
    that ever hits the proxy.
    """
    if not tenant_id:
        return None
    prisma_client = _get_prisma_client()
    existing = await prisma_client.db.creditwallet.find_unique(
        where={"tenant_id": tenant_id}
    )
    if existing is None:
        return None
    row = existing.model_dump() if hasattr(existing, "model_dump") else dict(existing)
    gift = float(row.get("gift_balance") or 0)
    paid = float(row.get("paid_balance") or 0)
    frozen = float(row.get("frozen_amount") or 0)
    return {
        "tenant_id": tenant_id,
        "gift_balance": gift,
        "paid_balance": paid,
        "frozen_amount": frozen,
        "available": gift + paid - frozen,
    }


async def ensure_wallet(tenant_id: str) -> Dict[str, Any]:
    """Lazy-create a wallet for ``tenant_id`` if missing. Returns the wallet row.

    No initial gift balance — gifting policy can be layered on later (e.g.
    in a UserTable creation hook). This keeps the credit subsystem
    side-effect-free at boot.
    """
    if not tenant_id:
        raise WalletError("credit_service: tenant_id is required")
    prisma_client = _get_prisma_client()
    existing = await prisma_client.db.creditwallet.find_unique(
        where={"tenant_id": tenant_id}
    )
    if existing is not None:
        return (
            existing.model_dump() if hasattr(existing, "model_dump") else dict(existing)
        )
    created = await prisma_client.db.creditwallet.create(data={"tenant_id": tenant_id})
    return created.model_dump() if hasattr(created, "model_dump") else dict(created)


async def pre_deduct(
    *,
    tenant_id: str,
    estimated_credits: float,
    agent_record_id: str,
    model_name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Freeze ``estimated_credits`` on the wallet.

    Raises ``BudgetExceededError`` (HTTP 429) if available balance
    (``gift + paid - frozen``) is below the estimate.
    """
    if estimated_credits <= 0:
        # Nothing to freeze — caller still wants a tx record for traceability.
        estimated_credits = 0.0
    prisma_client = _get_prisma_client()
    async with prisma_client.db.tx() as tx:
        wallet = await _find_wallet(tx, tenant_id)
        if wallet is None:
            # Create-on-write so a first-task-for-this-tenant flow works.
            await tx.creditwallet.create(data={"tenant_id": tenant_id})
            wallet = await _find_wallet(tx, tenant_id)
            if wallet is None:
                raise WalletError(
                    f"credit_service: wallet for tenant {tenant_id!r} could not be created"
                )

        gift = float(wallet.get("gift_balance") or 0)
        paid = float(wallet.get("paid_balance") or 0)
        frozen = float(wallet.get("frozen_amount") or 0)
        available = gift + paid - frozen

        if estimated_credits > available:
            raise BudgetExceededError(
                current_cost=available,
                max_budget=estimated_credits,
                message=(
                    f"积分不足，当前可用 {available:.4f} C，"
                    f"需要 {estimated_credits:.4f} C"
                ),
            )

        new_frozen = frozen + estimated_credits
        await tx.creditwallet.update(
            where={"tenant_id": tenant_id},
            data={"frozen_amount": new_frozen},
        )

        await tx.credittransaction.create(
            data={
                "tenant_id": tenant_id,
                "tx_type": _PRE_DEDUCT,
                "wallet_type": _GIFT,  # nominal — freeze isn't tied to one purse
                "amount": -estimated_credits,
                "balance_after": gift,  # gift unchanged on freeze
                "agent_record_id": agent_record_id,
                "model_name": model_name,
                "description": description or f"预扣 {estimated_credits:.4f} 积分",
            }
        )

    verbose_proxy_logger.debug(
        "credit_service.pre_deduct: tenant=%s task=%s amount=%.4f available_before=%.4f",
        tenant_id,
        agent_record_id,
        estimated_credits,
        available,
    )
    return {
        "tenant_id": tenant_id,
        "agent_record_id": agent_record_id,
        "frozen_amount": estimated_credits,
        "available_before": available,
    }


async def charge(
    *,
    tenant_id: str,
    actual_credits: float,
    agent_record_id: Optional[str] = None,
    model_name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Sync deduct — no freeze, gift first then paid.

    For chat / sync media where the request lifecycle is short enough that
    we don't need a freeze window. Called from the post-call success hook
    after litellm computes ``response_cost``. The wallet acts as a prepaid
    account; litellm's own ``LiteLLM_UserTable.spend`` accumulator and
    ``max_budget`` enforcement keep running in parallel for reporting and
    hard caps.

    Idempotent when ``agent_record_id`` is provided — second call with the
    same id is a no-op (skips double-charging on retries). Without an id,
    every call deducts.

    Allows the wallet to go negative if ``actual_credits`` exceeds the
    available balance — we log a warning but still record the consumption,
    matching the prepaid account convention that subsequent requests are
    refused by pre-call balance checks rather than failing post-completion.
    """
    actual_credits = max(float(actual_credits or 0), 0.0)
    if actual_credits == 0:
        return {"charged": False, "reason": "zero_amount"}

    prisma_client = _get_prisma_client()
    async with prisma_client.db.tx() as tx:
        if agent_record_id and await _already_finalised(
            tx, agent_record_id, _CONSUMPTION
        ):
            verbose_proxy_logger.debug(
                "credit_service.charge: task=%s already charged — skipping",
                agent_record_id,
            )
            return {"charged": False, "reason": "already_charged"}

        wallet = await _find_wallet(tx, tenant_id)
        if wallet is None:
            await tx.creditwallet.create(data={"tenant_id": tenant_id})
            wallet = await _find_wallet(tx, tenant_id)
            if wallet is None:
                raise WalletError(
                    f"credit_service.charge: wallet for tenant {tenant_id!r} "
                    "could not be created"
                )

        gift = float(wallet.get("gift_balance") or 0)
        paid = float(wallet.get("paid_balance") or 0)
        total_consumed = float(wallet.get("total_consumed") or 0)

        gift_deduct = min(gift, actual_credits)
        paid_deduct = actual_credits - gift_deduct  # may push paid negative

        new_gift = gift - gift_deduct
        new_paid = paid - paid_deduct  # negative if overspent
        if new_paid < 0:
            verbose_proxy_logger.warning(
                "credit_service.charge: wallet went negative — tenant=%s "
                "actual=%.4f gift=%.4f paid=%.4f → new_paid=%.4f",
                tenant_id,
                actual_credits,
                gift,
                paid,
                new_paid,
            )

        await tx.creditwallet.update(
            where={"tenant_id": tenant_id},
            data={
                "gift_balance": new_gift,
                "paid_balance": new_paid,
                "total_consumed": total_consumed + actual_credits,
            },
        )

        if gift_deduct > 0:
            await tx.credittransaction.create(
                data={
                    "tenant_id": tenant_id,
                    "tx_type": _CONSUMPTION,
                    "wallet_type": _GIFT,
                    "amount": -gift_deduct,
                    "balance_after": new_gift,
                    "agent_record_id": agent_record_id,
                    "model_name": model_name,
                    "description": (description or f"消耗赠送积分 {gift_deduct:.4f} C"),
                }
            )
        if paid_deduct > 0:
            await tx.credittransaction.create(
                data={
                    "tenant_id": tenant_id,
                    "tx_type": _CONSUMPTION,
                    "wallet_type": _PAID,
                    "amount": -paid_deduct,
                    "balance_after": new_paid,
                    "agent_record_id": agent_record_id,
                    "model_name": model_name,
                    "description": (description or f"消耗充值积分 {paid_deduct:.4f} C"),
                }
            )

    return {
        "charged": True,
        "actual_credits": actual_credits,
        "gift_deduct": gift_deduct,
        "paid_deduct": paid_deduct,
        "new_balance": new_gift + new_paid,
    }


async def settle(
    *,
    tenant_id: str,
    agent_record_id: str,
    actual_credits: float,
) -> Dict[str, Any]:
    """Settle a previously-frozen task at its actual cost.

    Gift balance is consumed first; paid balance covers the remainder. If
    ``actual_credits > frozen``, the wallet still pays the difference
    (caller has already gated this with the original pre_deduct).
    Idempotent — calling twice with the same ``agent_record_id`` is a no-op.
    """
    if not agent_record_id:
        raise WalletError("credit_service.settle: agent_record_id is required")
    actual_credits = max(float(actual_credits or 0), 0.0)

    prisma_client = _get_prisma_client()
    async with prisma_client.db.tx() as tx:
        if await _already_finalised(tx, agent_record_id, _SETTLEMENT):
            verbose_proxy_logger.debug(
                "credit_service.settle: task=%s already settled — skipping",
                agent_record_id,
            )
            return {"settled": False, "reason": "already_settled"}
        if await _already_finalised(tx, agent_record_id, _REFUND):
            verbose_proxy_logger.debug(
                "credit_service.settle: task=%s already refunded — skipping",
                agent_record_id,
            )
            return {"settled": False, "reason": "already_refunded"}

        wallet = await _find_wallet(tx, tenant_id)
        if wallet is None:
            raise WalletError(
                f"credit_service.settle: wallet missing for tenant {tenant_id!r}"
            )

        pre_row = await _latest_pre_deduct(tx, agent_record_id)
        frozen_for_task = abs(float(pre_row["amount"])) if pre_row is not None else 0.0

        gift = float(wallet.get("gift_balance") or 0)
        paid = float(wallet.get("paid_balance") or 0)
        frozen = float(wallet.get("frozen_amount") or 0)
        total_consumed = float(wallet.get("total_consumed") or 0)

        gift_deduct = min(gift, actual_credits)
        paid_deduct = max(actual_credits - gift_deduct, 0.0)

        new_gift = max(gift - gift_deduct, 0.0)
        new_paid = max(paid - paid_deduct, 0.0)
        new_frozen = max(frozen - frozen_for_task, 0.0)

        await tx.creditwallet.update(
            where={"tenant_id": tenant_id},
            data={
                "gift_balance": new_gift,
                "paid_balance": new_paid,
                "frozen_amount": new_frozen,
                "total_consumed": total_consumed + actual_credits,
            },
        )

        if gift_deduct > 0:
            await tx.credittransaction.create(
                data={
                    "tenant_id": tenant_id,
                    "tx_type": _CONSUMPTION,
                    "wallet_type": _GIFT,
                    "amount": -gift_deduct,
                    "balance_after": new_gift,
                    "agent_record_id": agent_record_id,
                    "description": f"消耗赠送积分 {gift_deduct:.4f} C",
                }
            )
        if paid_deduct > 0:
            await tx.credittransaction.create(
                data={
                    "tenant_id": tenant_id,
                    "tx_type": _CONSUMPTION,
                    "wallet_type": _PAID,
                    "amount": -paid_deduct,
                    "balance_after": new_paid,
                    "agent_record_id": agent_record_id,
                    "description": f"消耗充值积分 {paid_deduct:.4f} C",
                }
            )

        # Settlement marker (always written, even when actual == frozen)
        # so idempotency check has a single source of truth.
        diff = actual_credits - frozen_for_task
        await tx.credittransaction.create(
            data={
                "tenant_id": tenant_id,
                "tx_type": _SETTLEMENT,
                "wallet_type": _GIFT,  # nominal
                "amount": -actual_credits,
                "balance_after": new_gift + new_paid,
                "agent_record_id": agent_record_id,
                "description": (
                    f"结算 actual={actual_credits:.4f} frozen={frozen_for_task:.4f} "
                    f"diff={diff:+.4f}"
                ),
            }
        )

    verbose_proxy_logger.debug(
        "credit_service.settle: tenant=%s task=%s actual=%.4f frozen=%.4f",
        tenant_id,
        agent_record_id,
        actual_credits,
        frozen_for_task,
    )
    return {
        "settled": True,
        "actual_credits": actual_credits,
        "frozen_amount": frozen_for_task,
        "gift_deduct": gift_deduct,
        "paid_deduct": paid_deduct,
    }


async def rebind_agent_record_id(
    *,
    placeholder: str,
    real_id: str,
) -> int:
    """Swap a placeholder ``agent_record_id`` (used before submit) to the real
    upstream task_id once submit succeeds.

    Rebinds every transaction tagged with the placeholder — currently that's
    just the pre_deduct row but written generally so future flow steps that
    log before submit also stay grouped.

    Returns the number of rows updated.
    """
    if not placeholder or not real_id:
        raise WalletError(
            "credit_service.rebind_agent_record_id: both placeholder and real_id required"
        )
    prisma_client = _get_prisma_client()
    updated = await prisma_client.db.credittransaction.update_many(
        where={"agent_record_id": placeholder},
        data={"agent_record_id": real_id},
    )
    verbose_proxy_logger.debug(
        "credit_service.rebind: placeholder=%s real=%s rows=%s",
        placeholder,
        real_id,
        updated,
    )
    return int(updated or 0)


async def refund(
    *,
    tenant_id: str,
    agent_record_id: str,
    reason: str,
) -> Dict[str, Any]:
    """Unfreeze the previously-frozen amount and log a refund transaction.

    Idempotent — if a settlement or earlier refund already exists for this
    ``agent_record_id``, this is a no-op.
    """
    if not agent_record_id:
        raise WalletError("credit_service.refund: agent_record_id is required")
    prisma_client = _get_prisma_client()
    async with prisma_client.db.tx() as tx:
        if await _already_finalised(tx, agent_record_id, _REFUND):
            return {"refunded": False, "reason": "already_refunded"}
        if await _already_finalised(tx, agent_record_id, _SETTLEMENT):
            return {"refunded": False, "reason": "already_settled"}

        pre_row = await _latest_pre_deduct(tx, agent_record_id)
        if pre_row is None:
            # No freeze on record — nothing to refund. Still write a marker
            # so any future settle/refund knows this task is closed.
            await tx.credittransaction.create(
                data={
                    "tenant_id": tenant_id,
                    "tx_type": _REFUND,
                    "wallet_type": _GIFT,
                    "amount": 0.0,
                    "balance_after": 0.0,
                    "agent_record_id": agent_record_id,
                    "description": f"退款（无预扣记录）: {reason}",
                }
            )
            return {"refunded": True, "amount": 0.0}

        wallet = await _find_wallet(tx, tenant_id)
        if wallet is None:
            raise WalletError(
                f"credit_service.refund: wallet missing for tenant {tenant_id!r}"
            )

        refund_amount = abs(float(pre_row["amount"]))
        frozen = float(wallet.get("frozen_amount") or 0)
        new_frozen = max(frozen - refund_amount, 0.0)

        await tx.creditwallet.update(
            where={"tenant_id": tenant_id},
            data={"frozen_amount": new_frozen},
        )
        await tx.credittransaction.create(
            data={
                "tenant_id": tenant_id,
                "tx_type": _REFUND,
                "wallet_type": _GIFT,
                "amount": refund_amount,
                "balance_after": float(wallet.get("gift_balance") or 0),
                "agent_record_id": agent_record_id,
                "description": f"退还预扣积分: {reason}",
            }
        )

    verbose_proxy_logger.debug(
        "credit_service.refund: tenant=%s task=%s amount=%.4f",
        tenant_id,
        agent_record_id,
        refund_amount,
    )
    return {"refunded": True, "amount": refund_amount}
