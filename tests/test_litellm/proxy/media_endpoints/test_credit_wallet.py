"""Unit tests for the Prisma-backed credit wallet (pre_deduct / settle / refund).

We don't spin up a real Postgres — instead we mock the prisma client so the
tests run in CI without infra. The mock mirrors enough of the real client's
shape (async ``find_unique``, ``find_many``, ``create``, ``update``,
``update_many``, ``tx`` context manager) to exercise the actual logic in
``wallet.py``.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

import pytest

from litellm.exceptions import BudgetExceededError
from litellm.proxy.credit_service import wallet as wallet_mod
from litellm.proxy.credit_service.wallet import (
    pre_deduct,
    rebind_agent_record_id,
    refund,
    settle,
)


# ----------------------------------------------------------------------
# In-memory fake of the Prisma client surface that wallet.py touches.
# ----------------------------------------------------------------------


class _Table:
    def __init__(self, store: List[Dict[str, Any]]) -> None:
        self._store = store

    async def find_unique(self, where: Dict[str, Any]):
        for row in self._store:
            if all(row.get(k) == v for k, v in where.items()):
                return _Row(row)
        return None

    async def find_many(
        self,
        where: Dict[str, Any] | None = None,
        order: Dict[str, Any] | None = None,
        take: int | None = None,
    ):
        where = where or {}
        rows = [r for r in self._store if all(r.get(k) == v for k, v in where.items())]
        if order:
            (key, direction) = next(iter(order.items()))
            rows.sort(key=lambda r: r.get(key), reverse=(direction == "desc"))
        if take is not None:
            rows = rows[:take]
        return [_Row(r) for r in rows]

    async def create(self, data: Dict[str, Any]):
        row = {"id": data.get("id") or str(uuid.uuid4()), **data}
        self._store.append(row)
        return _Row(row)

    async def update(self, where: Dict[str, Any], data: Dict[str, Any]):
        for row in self._store:
            if all(row.get(k) == v for k, v in where.items()):
                row.update(data)
                return _Row(row)
        raise AssertionError(f"no row matches {where}")

    async def update_many(self, where: Dict[str, Any], data: Dict[str, Any]):
        n = 0
        for row in self._store:
            if all(row.get(k) == v for k, v in where.items()):
                row.update(data)
                n += 1
        return n


class _Row:
    """Looks like a Prisma response row — supports both attribute and dict-style."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, k):
        return self._data[k]

    def get(self, k, default=None):
        return self._data.get(k, default)


class _DB:
    def __init__(self) -> None:
        self.wallets: List[Dict[str, Any]] = []
        self.txs: List[Dict[str, Any]] = []
        self.creditwallet = _Table(self.wallets)
        self.credittransaction = _Table(self.txs)

    def tx(self):
        # Single-process fake — we don't simulate isolation; tests don't need it.
        db_self = self

        class _TxCtx:
            async def __aenter__(self):
                return db_self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return _TxCtx()


class _PrismaClient:
    def __init__(self) -> None:
        self.db = _DB()


@pytest.fixture
def fake_prisma(monkeypatch):
    """Install a fake prisma_client global on proxy_server."""
    client = _PrismaClient()
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", client, raising=False)
    return client


async def _seed_wallet(
    fake_prisma: _PrismaClient,
    *,
    tenant_id: str = "team-1",
    gift: float = 0.0,
    paid: float = 0.0,
    frozen: float = 0.0,
) -> None:
    fake_prisma.db.wallets.append(
        {
            "id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "gift_balance": gift,
            "paid_balance": paid,
            "frozen_amount": frozen,
            "total_consumed": 0.0,
        }
    )


def _txs(fake_prisma: _PrismaClient, agent_record_id: str) -> List[Dict[str, Any]]:
    return [
        t for t in fake_prisma.db.txs if t.get("agent_record_id") == agent_record_id
    ]


# ----------------------------------------------------------------------
# pre_deduct
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_deduct_freezes_amount_when_balance_sufficient(fake_prisma):
    await _seed_wallet(fake_prisma, paid=100.0)
    result = await pre_deduct(
        tenant_id="team-1",
        estimated_credits=10.0,
        agent_record_id="task-1",
        model_name="grok-video-3",
    )
    assert result["frozen_amount"] == 10.0
    wallet = fake_prisma.db.wallets[0]
    assert wallet["frozen_amount"] == 10.0
    txs = _txs(fake_prisma, "task-1")
    assert len(txs) == 1
    assert txs[0]["tx_type"] == "pre_deduct"
    assert txs[0]["amount"] == -10.0


@pytest.mark.asyncio
async def test_pre_deduct_raises_when_insufficient(fake_prisma):
    await _seed_wallet(fake_prisma, paid=1.0)
    with pytest.raises(BudgetExceededError):
        await pre_deduct(
            tenant_id="team-1",
            estimated_credits=10.0,
            agent_record_id="task-1",
            model_name="m",
        )
    # No freeze and no tx written when budget check fails
    assert fake_prisma.db.wallets[0]["frozen_amount"] == 0.0
    assert _txs(fake_prisma, "task-1") == []


@pytest.mark.asyncio
async def test_pre_deduct_creates_wallet_on_first_call(fake_prisma):
    """ensure_wallet semantics: first call for a new tenant lazy-creates the row."""
    # Initial empty wallet auto-created with 0/0 → 0 available → budget error
    with pytest.raises(BudgetExceededError):
        await pre_deduct(
            tenant_id="team-1",
            estimated_credits=5.0,
            agent_record_id="task-1",
            model_name="m",
        )
    # The wallet now exists (created during the failed call)
    assert len(fake_prisma.db.wallets) == 1
    assert fake_prisma.db.wallets[0]["tenant_id"] == "team-1"


@pytest.mark.asyncio
async def test_pre_deduct_respects_existing_frozen(fake_prisma):
    """Available = gift + paid - frozen; existing frozen reduces the headroom."""
    await _seed_wallet(fake_prisma, paid=10.0, frozen=8.0)  # 2 available
    with pytest.raises(BudgetExceededError):
        await pre_deduct(
            tenant_id="team-1",
            estimated_credits=5.0,
            agent_record_id="t",
            model_name="m",
        )


# ----------------------------------------------------------------------
# settle
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settle_under_estimate_returns_difference(fake_prisma):
    """Actual cost < frozen → wallet pays only actual, frozen fully released."""
    await _seed_wallet(fake_prisma, paid=100.0)
    await pre_deduct(
        tenant_id="team-1",
        estimated_credits=10.0,
        agent_record_id="task-1",
        model_name="m",
    )
    result = await settle(
        tenant_id="team-1", agent_record_id="task-1", actual_credits=6.0
    )
    assert result["settled"] is True
    wallet = fake_prisma.db.wallets[0]
    assert wallet["paid_balance"] == 94.0  # 100 - 6
    assert wallet["frozen_amount"] == 0.0
    assert wallet["total_consumed"] == 6.0


@pytest.mark.asyncio
async def test_settle_over_estimate_still_charges_full_actual(fake_prisma):
    """Actual cost > frozen → wallet still pays the overshoot."""
    await _seed_wallet(fake_prisma, paid=100.0)
    await pre_deduct(
        tenant_id="team-1",
        estimated_credits=5.0,
        agent_record_id="task-1",
        model_name="m",
    )
    await settle(tenant_id="team-1", agent_record_id="task-1", actual_credits=8.0)
    wallet = fake_prisma.db.wallets[0]
    assert wallet["paid_balance"] == 92.0  # 100 - 8
    assert wallet["frozen_amount"] == 0.0


@pytest.mark.asyncio
async def test_settle_drains_gift_before_paid(fake_prisma):
    await _seed_wallet(fake_prisma, gift=3.0, paid=100.0)
    await pre_deduct(
        tenant_id="team-1",
        estimated_credits=5.0,
        agent_record_id="task-1",
        model_name="m",
    )
    await settle(tenant_id="team-1", agent_record_id="task-1", actual_credits=5.0)
    wallet = fake_prisma.db.wallets[0]
    assert wallet["gift_balance"] == 0.0
    assert wallet["paid_balance"] == 98.0  # 100 - (5 - 3)


@pytest.mark.asyncio
async def test_settle_is_idempotent(fake_prisma):
    """Second settle call with the same task_id must be a no-op."""
    await _seed_wallet(fake_prisma, paid=100.0)
    await pre_deduct(
        tenant_id="team-1",
        estimated_credits=10.0,
        agent_record_id="task-1",
        model_name="m",
    )
    first = await settle(
        tenant_id="team-1", agent_record_id="task-1", actual_credits=5.0
    )
    second = await settle(
        tenant_id="team-1", agent_record_id="task-1", actual_credits=5.0
    )
    assert first["settled"] is True
    assert second["settled"] is False
    # Wallet only debited once
    assert fake_prisma.db.wallets[0]["paid_balance"] == 95.0


# ----------------------------------------------------------------------
# refund
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refund_restores_frozen_amount(fake_prisma):
    await _seed_wallet(fake_prisma, paid=100.0)
    await pre_deduct(
        tenant_id="team-1",
        estimated_credits=10.0,
        agent_record_id="task-1",
        model_name="m",
    )
    assert fake_prisma.db.wallets[0]["frozen_amount"] == 10.0
    result = await refund(tenant_id="team-1", agent_record_id="task-1", reason="failed")
    assert result["refunded"] is True
    assert result["amount"] == 10.0
    wallet = fake_prisma.db.wallets[0]
    assert wallet["frozen_amount"] == 0.0
    assert wallet["paid_balance"] == 100.0


@pytest.mark.asyncio
async def test_refund_is_idempotent(fake_prisma):
    await _seed_wallet(fake_prisma, paid=100.0)
    await pre_deduct(
        tenant_id="team-1",
        estimated_credits=10.0,
        agent_record_id="task-1",
        model_name="m",
    )
    first = await refund(tenant_id="team-1", agent_record_id="task-1", reason="x")
    second = await refund(tenant_id="team-1", agent_record_id="task-1", reason="x")
    assert first["refunded"] is True
    assert second["refunded"] is False


@pytest.mark.asyncio
async def test_refund_skipped_when_already_settled(fake_prisma):
    """Once settled, a stray refund attempt must NOT double-process."""
    await _seed_wallet(fake_prisma, paid=100.0)
    await pre_deduct(
        tenant_id="team-1",
        estimated_credits=10.0,
        agent_record_id="task-1",
        model_name="m",
    )
    await settle(tenant_id="team-1", agent_record_id="task-1", actual_credits=5.0)
    result = await refund(tenant_id="team-1", agent_record_id="task-1", reason="late")
    assert result["refunded"] is False


# ----------------------------------------------------------------------
# rebind_agent_record_id
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebind_swaps_placeholder_to_real_id(fake_prisma):
    await _seed_wallet(fake_prisma, paid=100.0)
    await pre_deduct(
        tenant_id="team-1",
        estimated_credits=10.0,
        agent_record_id="pending:abc",
        model_name="m",
    )
    n = await rebind_agent_record_id(placeholder="pending:abc", real_id="task-42")
    assert n == 1
    # After rebind, look-ups by real_id work
    settled = await settle(
        tenant_id="team-1", agent_record_id="task-42", actual_credits=5.0
    )
    assert settled["settled"] is True
    # The pre_deduct row's agent_record_id now equals "task-42"
    rebound = _txs(fake_prisma, "task-42")
    assert any(t["tx_type"] == "pre_deduct" for t in rebound)
    # Nothing left under the placeholder
    assert _txs(fake_prisma, "pending:abc") == []
