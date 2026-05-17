"""Tests for WalletChargeLogger — the litellm CustomLogger that debits the
prepaid wallet on every successful LLM/media response.

Reuses the in-memory Prisma fake from test_credit_wallet to avoid spinning
up a database.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

import pytest

from litellm.proxy.credit_service.callbacks import WalletChargeLogger
from tests.test_litellm.proxy.media_endpoints.test_credit_wallet import (
    _PrismaClient,
    _seed_wallet,
    _txs,
)


@pytest.fixture
def fake_prisma(monkeypatch):
    """Same fixture as test_credit_wallet — wires our fake prisma into proxy_server."""
    client = _PrismaClient()
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", client, raising=False)
    return client


def _kwargs_for(
    *,
    response_cost: float,
    team_id: str | None = "team-1",
    user_id: str | None = None,
    call_id: str = "call-abc",
    model: str = "gpt-4o",
    call_type: str = "acompletion",
) -> Dict[str, Any]:
    """Build a kwargs dict shaped like litellm's success-callback payload."""
    metadata: Dict[str, Any] = {}
    if team_id:
        metadata["user_api_key_team_id"] = team_id
    if user_id:
        metadata["user_api_key_user_id"] = user_id
    return {
        "standard_logging_object": {
            "response_cost": response_cost,
            "litellm_call_id": call_id,
            "model": model,
            "call_type": call_type,
            "metadata": metadata,
        }
    }


@pytest.mark.asyncio
async def test_callback_debits_wallet_on_success(fake_prisma):
    await _seed_wallet(fake_prisma, paid=100.0)
    logger = WalletChargeLogger()
    await logger.async_log_success_event(
        kwargs=_kwargs_for(response_cost=2.5),
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    wallet = fake_prisma.db.wallets[0]
    assert wallet["paid_balance"] == 97.5
    assert wallet["total_consumed"] == 2.5


@pytest.mark.asyncio
async def test_callback_is_idempotent_on_retry(fake_prisma):
    """Same litellm_call_id firing twice (e.g. retry within a request) → no double-charge."""
    await _seed_wallet(fake_prisma, paid=100.0)
    logger = WalletChargeLogger()
    kwargs = _kwargs_for(response_cost=2.5, call_id="call-abc")
    await logger.async_log_success_event(kwargs, None, None, None)
    await logger.async_log_success_event(kwargs, None, None, None)
    assert fake_prisma.db.wallets[0]["paid_balance"] == 97.5


@pytest.mark.asyncio
async def test_callback_skips_zero_cost(fake_prisma):
    """Cache hits / free models report cost=0 — wallet untouched."""
    await _seed_wallet(fake_prisma, paid=100.0)
    logger = WalletChargeLogger()
    await logger.async_log_success_event(
        kwargs=_kwargs_for(response_cost=0.0),
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    assert fake_prisma.db.wallets[0]["paid_balance"] == 100.0
    assert fake_prisma.db.txs == []


@pytest.mark.asyncio
async def test_callback_skips_when_no_tenant_id(fake_prisma):
    """Master key requests (no team_id, no user_id) skip silently."""
    await _seed_wallet(fake_prisma, paid=100.0, tenant_id="team-1")
    logger = WalletChargeLogger()
    kwargs = _kwargs_for(response_cost=2.5, team_id=None, user_id=None)
    await logger.async_log_success_event(kwargs, None, None, None)
    assert fake_prisma.db.wallets[0]["paid_balance"] == 100.0


@pytest.mark.asyncio
async def test_callback_falls_back_to_user_id_when_no_team(fake_prisma):
    """API keys not attached to a team should still bill (to user_id wallet)."""
    await _seed_wallet(fake_prisma, paid=50.0, tenant_id="user-bob")
    logger = WalletChargeLogger()
    kwargs = _kwargs_for(response_cost=2.5, team_id=None, user_id="user-bob")
    await logger.async_log_success_event(kwargs, None, None, None)
    wallet = next(w for w in fake_prisma.db.wallets if w["tenant_id"] == "user-bob")
    assert wallet["paid_balance"] == 47.5


@pytest.mark.asyncio
async def test_callback_skips_when_prisma_unavailable(monkeypatch):
    """No DB → no crash. WalletError swallowed in the callback."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None, raising=False)
    logger = WalletChargeLogger()
    # Should not raise even though there's no DB
    await logger.async_log_success_event(
        kwargs=_kwargs_for(response_cost=2.5),
        response_obj=None,
        start_time=None,
        end_time=None,
    )


@pytest.mark.asyncio
async def test_callback_preserves_call_id_in_tx_row(fake_prisma):
    """agent_record_id should land in the tx row for traceability."""
    await _seed_wallet(fake_prisma, paid=100.0)
    logger = WalletChargeLogger()
    kwargs = _kwargs_for(response_cost=1.5, call_id="chat-xyz-789")
    await logger.async_log_success_event(kwargs, None, None, None)
    rows = _txs(fake_prisma, "chat-xyz-789")
    assert len(rows) == 1
    assert rows[0]["wallet_type"] == "paid"
    assert rows[0]["amount"] == -1.5
    assert rows[0]["model_name"] == "gpt-4o"


# ----------------------------------------------------------------------
# pre-call hook — reject 402 when wallet is empty / overdrawn
# ----------------------------------------------------------------------

from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.credit_service.wallet import charge as _charge


def _api_key(
    team_id: str | None = "team-1", user_id: str | None = None
) -> UserAPIKeyAuth:
    """Build a minimal UserAPIKeyAuth — only the fields the hook reads."""
    return UserAPIKeyAuth(team_id=team_id, user_id=user_id)


@pytest.mark.asyncio
async def test_pre_call_allows_when_balance_positive(fake_prisma):
    await _seed_wallet(fake_prisma, paid=10.0)
    logger = WalletChargeLogger()
    # No exception = allowed through
    result = await logger.async_pre_call_hook(
        user_api_key_dict=_api_key(),
        cache=None,
        data={},
        call_type="acompletion",
    )
    assert result is None


@pytest.mark.asyncio
async def test_pre_call_rejects_when_balance_zero(fake_prisma):
    """Brand-new wallet with 0 balance → 402."""
    await _seed_wallet(fake_prisma)  # all zero
    logger = WalletChargeLogger()
    with pytest.raises(HTTPException) as exc_info:
        await logger.async_pre_call_hook(
            user_api_key_dict=_api_key(),
            cache=None,
            data={},
            call_type="acompletion",
        )
    assert exc_info.value.status_code == 402


@pytest.mark.asyncio
async def test_pre_call_rejects_after_overdraw(fake_prisma):
    """The canonical loop: charge pushes paid negative → next request blocked."""
    await _seed_wallet(fake_prisma, paid=2.0)
    # First chat overspends — allowed because charge runs post-call.
    await _charge(tenant_id="team-1", actual_credits=10.0, agent_record_id="chat-1")
    assert fake_prisma.db.wallets[0]["paid_balance"] == -8.0

    # Second chat — pre-call hook now sees negative balance and refuses.
    logger = WalletChargeLogger()
    with pytest.raises(HTTPException) as exc_info:
        await logger.async_pre_call_hook(
            user_api_key_dict=_api_key(),
            cache=None,
            data={},
            call_type="acompletion",
        )
    assert exc_info.value.status_code == 402
    assert "余额不足" in exc_info.value.detail["error"]["message"]


@pytest.mark.asyncio
async def test_pre_call_skips_when_no_wallet_row(fake_prisma):
    """Tenant without a wallet row = not on prepaid plan → pass through."""
    logger = WalletChargeLogger()
    assert fake_prisma.db.wallets == []
    result = await logger.async_pre_call_hook(
        user_api_key_dict=_api_key(team_id="team-no-wallet"),
        cache=None,
        data={},
        call_type="acompletion",
    )
    assert result is None
    # And we didn't accidentally create one
    assert fake_prisma.db.wallets == []


@pytest.mark.asyncio
async def test_pre_call_skips_when_no_tenant_id(fake_prisma):
    """Master key (no team_id, no user_id) → pass through."""
    await _seed_wallet(fake_prisma)
    logger = WalletChargeLogger()
    result = await logger.async_pre_call_hook(
        user_api_key_dict=_api_key(team_id=None, user_id=None),
        cache=None,
        data={},
        call_type="acompletion",
    )
    assert result is None


@pytest.mark.asyncio
async def test_pre_call_skips_non_billable_call_types(fake_prisma):
    """/utils/token_counter, /models etc. shouldn't be gated."""
    await _seed_wallet(fake_prisma)  # 0 balance
    logger = WalletChargeLogger()
    # acompletion would be rejected (see test above); pass_through must not.
    result = await logger.async_pre_call_hook(
        user_api_key_dict=_api_key(),
        cache=None,
        data={},
        call_type="pass_through_endpoint",
    )
    assert result is None


@pytest.mark.asyncio
async def test_pre_call_skips_when_prisma_unavailable(monkeypatch):
    """No DB → no crash, request allowed."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None, raising=False)
    logger = WalletChargeLogger()
    result = await logger.async_pre_call_hook(
        user_api_key_dict=_api_key(),
        cache=None,
        data={},
        call_type="acompletion",
    )
    assert result is None


@pytest.mark.asyncio
async def test_pre_call_uses_user_id_when_no_team(fake_prisma):
    """Same fallback as the post-call charge — user-scoped wallet works."""
    await _seed_wallet(fake_prisma, paid=5.0, tenant_id="user-bob")
    logger = WalletChargeLogger()
    result = await logger.async_pre_call_hook(
        user_api_key_dict=_api_key(team_id=None, user_id="user-bob"),
        cache=None,
        data={},
        call_type="acompletion",
    )
    assert result is None  # allowed (balance = 5.0)


@pytest.mark.asyncio
async def test_pre_call_accounts_for_frozen_balance(fake_prisma):
    """Available subtracts frozen — pending video tasks reserve headroom."""
    # 10 paid, 10 frozen by an in-flight video → available = 0 → reject
    await _seed_wallet(fake_prisma, paid=10.0, frozen=10.0)
    logger = WalletChargeLogger()
    with pytest.raises(HTTPException) as exc_info:
        await logger.async_pre_call_hook(
            user_api_key_dict=_api_key(),
            cache=None,
            data={},
            call_type="acompletion",
        )
    assert exc_info.value.status_code == 402
