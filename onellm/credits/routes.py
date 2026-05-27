"""
CREDIT MANAGEMENT

All ``/credit`` endpoints — wallet info, transaction list, recharge, admin
adjust, and recharge package CRUD.

The underlying tables live in the ``app`` schema (separate from upstream
``public``):

  - ``app.credit_wallet``       one row per tenant_id (= LiteLLM_TeamTable.team_id)
  - ``app.credit_transaction``  append-only log
  - ``app.credit_package``      admin-editable recharge packages

Prisma accessors are ``prisma_client.db.creditwallet`` /
``credittransaction`` / ``creditpackage`` (no ``LiteLLM_`` prefix — onellm
extensions only, naming intentionally visible).

Tenant resolution: the wallet is keyed by ``team_id`` when the caller has
one; otherwise by ``user_id`` (personal wallet). Admins can read/write
any tenant by passing ``tenant_id`` explicitly.

Endpoints:

  GET    /credit/wallet/me            current tenant's wallet (auto-init)
  GET    /credit/wallet/list          admin: paginated list of wallets
  POST   /credit/wallet/adjust        admin: adjust gift/paid balance
  GET    /credit/transactions         current tenant's transactions (paged)
  POST   /credit/recharge             simulated recharge → credit a package
  GET    /credit/packages             list active packages
  POST   /credit/packages             admin: create package
  PUT    /credit/packages/{id}        admin: update package
  DELETE /credit/packages/{id}        admin: delete package
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_GIFT = "gift"
_PAID = "paid"

_TX_PURCHASE = "purchase"
_TX_GIFT = "gift"
_TX_ADMIN_ADJUST = "admin_adjust"

# Default tenant fallback when both team_id and user_id are absent — should
# never happen in practice but keeps a single-tenant local-dev experience working.
_DEFAULT_TENANT = "default"

# Default new-user gift on wallet auto-init. Matches the docs/temp reference.
_INITIAL_GIFT_CREDITS = 10.0


def _resolve_tenant_id(
    user_api_key_dict: UserAPIKeyAuth, override: Optional[str] = None
) -> str:
    """Pick the tenant_id for credit operations.

    - Admin callers may pass ``override`` to read/write any tenant.
    - Otherwise fall back to ``team_id`` (preferred) or ``user_id``.
    """
    if override and _is_admin(user_api_key_dict):
        return override
    return user_api_key_dict.team_id or user_api_key_dict.user_id or _DEFAULT_TENANT


def _is_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    role = user_api_key_dict.user_role
    return role == LitellmUserRoles.PROXY_ADMIN


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if not _is_admin(user_api_key_dict):
        raise HTTPException(status_code=403, detail={"error": "Admin role required"})


def _get_prisma_client():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "onellm.credits: prisma_client is not initialised — DATABASE_URL required"
            },
        )
    return prisma_client


def _dump(row: Any) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return row.model_dump() if hasattr(row, "model_dump") else dict(row)


def _coerce_decimal(value: Any) -> float:
    """Postgres NUMERIC columns come back as Decimal — coerce to float for JSON."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _serialize_wallet(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    gift = _coerce_decimal(row.get("gift_balance"))
    paid = _coerce_decimal(row.get("paid_balance"))
    frozen = _coerce_decimal(row.get("frozen_amount"))
    return {
        "id": row.get("id"),
        "tenant_id": row.get("tenant_id"),
        "gift_balance": gift,
        "paid_balance": paid,
        "frozen_amount": frozen,
        "available": gift + paid - frozen,
        "total_consumed": _coerce_decimal(row.get("total_consumed")),
        "total_recharged": _coerce_decimal(row.get("total_recharged")),
        "total_gifted": _coerce_decimal(row.get("total_gifted")),
        "low_balance_threshold": (
            _coerce_decimal(row["low_balance_threshold"])
            if row.get("low_balance_threshold") is not None
            else None
        ),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _serialize_transaction(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "tenant_id": row.get("tenant_id"),
        "tx_type": row.get("tx_type"),
        "wallet_type": row.get("wallet_type"),
        "amount": _coerce_decimal(row.get("amount")),
        "balance_after": _coerce_decimal(row.get("balance_after")),
        "agent_record_id": row.get("agent_record_id"),
        "model_name": row.get("model_name"),
        "description": row.get("description"),
        "metadata": row.get("metadata") or {},
        "created_at": row.get("created_at"),
    }


def _serialize_package(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(row.get("id")) if row.get("id") is not None else None,
        "name": row.get("name"),
        "credits_amount": _coerce_decimal(row.get("credits_amount")),
        "price_cny": _coerce_decimal(row.get("price_cny")),
        "bonus_credits": _coerce_decimal(row.get("bonus_credits")),
        "badge_text": row.get("badge_text"),
        "sort_order": row.get("sort_order") or 0,
        "is_active": bool(row.get("is_active")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class WalletAdjustRequest(BaseModel):
    tenant_id: str
    amount: float = Field(..., description="Positive = credit, negative = debit")
    wallet_type: str = Field(..., description="gift | paid")
    reason: Optional[str] = None


class RechargeRequest(BaseModel):
    package_id: str
    payment_method: Optional[str] = None  # "alipay" | "wechat" — simulated


class PackagePayload(BaseModel):
    name: str
    credits_amount: float
    price_cny: float
    bonus_credits: Optional[float] = 0.0
    badge_text: Optional[str] = None
    sort_order: Optional[int] = 0
    is_active: Optional[bool] = True


# ---------------------------------------------------------------------------
# Wallet endpoints
# ---------------------------------------------------------------------------


async def _ensure_wallet_with_gift(prisma_client, tenant_id: str) -> Dict[str, Any]:
    """Find or auto-init a wallet. First-time init grants ``_INITIAL_GIFT_CREDITS``."""
    existing = await prisma_client.db.creditwallet.find_unique(
        where={"tenant_id": tenant_id}
    )
    if existing is not None:
        return _dump(existing) or {}

    created = await prisma_client.db.creditwallet.create(
        data={
            "tenant_id": tenant_id,
            "gift_balance": _INITIAL_GIFT_CREDITS,
            "total_gifted": _INITIAL_GIFT_CREDITS,
        }
    )
    await prisma_client.db.credittransaction.create(
        data={
            "tenant_id": tenant_id,
            "tx_type": _TX_GIFT,
            "wallet_type": _GIFT,
            "amount": _INITIAL_GIFT_CREDITS,
            "balance_after": _INITIAL_GIFT_CREDITS,
            "description": f"新用户初始赠送 {_INITIAL_GIFT_CREDITS:.0f} 积分",
        }
    )
    return _dump(created) or {}


@router.get(
    "/credit/wallet/me",
    tags=["credit management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_my_wallet(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return the calling tenant's wallet. Auto-creates on first call."""
    prisma_client = _get_prisma_client()
    tenant_id = _resolve_tenant_id(user_api_key_dict)
    wallet = await _ensure_wallet_with_gift(prisma_client, tenant_id)
    return _serialize_wallet(wallet)


@router.get(
    "/credit/wallet/list",
    tags=["credit management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_wallets(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Admin: paginated wallet list. ``search`` matches tenant_id substring."""
    _require_admin(user_api_key_dict)
    prisma_client = _get_prisma_client()

    where: Dict[str, Any] = {}
    if search:
        where["tenant_id"] = {"contains": search}

    total = await prisma_client.db.creditwallet.count(where=where or None)  # type: ignore[arg-type]
    rows = await prisma_client.db.creditwallet.find_many(
        where=where or None,  # type: ignore[arg-type]
        order={"updated_at": "desc"},
        skip=(page - 1) * page_size,
        take=page_size,
    )

    wallets = [_serialize_wallet(_dump(r)) for r in rows]

    # Best-effort enrichment: look up team_alias if tenant_id matches a team.
    tenant_ids = [w["tenant_id"] for w in wallets if w]
    team_alias_by_id: Dict[str, str] = {}
    if tenant_ids:
        try:
            teams = await prisma_client.db.litellm_teamtable.find_many(
                where={"team_id": {"in": tenant_ids}}
            )
            for t in teams:
                d = _dump(t) or {}
                if d.get("team_id"):
                    team_alias_by_id[d["team_id"]] = d.get("team_alias") or ""
        except Exception:
            # Team table not available in some test envs — skip enrichment.
            pass

    for w in wallets:
        if w:
            w["tenant_alias"] = team_alias_by_id.get(w["tenant_id"], None)

    return {
        "data": wallets,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


@router.post(
    "/credit/wallet/adjust",
    tags=["credit management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def adjust_wallet(
    payload: WalletAdjustRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Admin: adjust gift or paid balance and log an ``admin_adjust`` tx."""
    _require_admin(user_api_key_dict)
    if payload.wallet_type not in (_GIFT, _PAID):
        raise HTTPException(
            status_code=400, detail={"error": "wallet_type must be 'gift' or 'paid'"}
        )
    if payload.amount == 0:
        raise HTTPException(status_code=400, detail={"error": "amount cannot be 0"})

    prisma_client = _get_prisma_client()
    wallet = _dump(
        await prisma_client.db.creditwallet.find_unique(
            where={"tenant_id": payload.tenant_id}
        )
    )
    if wallet is None:
        wallet = await _ensure_wallet_with_gift(prisma_client, payload.tenant_id)

    field = "gift_balance" if payload.wallet_type == _GIFT else "paid_balance"
    current = _coerce_decimal(wallet.get(field))
    new_balance = current + payload.amount

    if new_balance < 0:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Insufficient balance: current {field} is {current:.2f}"},
        )

    update_data: Dict[str, Any] = {
        field: new_balance,
        "updated_at": datetime.now(timezone.utc),
    }
    if payload.amount > 0:
        if payload.wallet_type == _GIFT:
            update_data["total_gifted"] = (
                _coerce_decimal(wallet.get("total_gifted")) + payload.amount
            )
        else:
            update_data["total_recharged"] = (
                _coerce_decimal(wallet.get("total_recharged")) + payload.amount
            )

    await prisma_client.db.creditwallet.update(
        where={"tenant_id": payload.tenant_id}, data=update_data
    )
    await prisma_client.db.credittransaction.create(
        data={
            "tenant_id": payload.tenant_id,
            "tx_type": _TX_ADMIN_ADJUST,
            "wallet_type": payload.wallet_type,
            "amount": payload.amount,
            "balance_after": new_balance,
            "description": payload.reason
            or f"管理员调整 {'+' if payload.amount > 0 else ''}{payload.amount} C",
            "metadata": {"adjusted_by": user_api_key_dict.user_id or "admin"},
        }
    )

    return {
        "success": True,
        "tenant_id": payload.tenant_id,
        "balance_after": new_balance,
    }


# ---------------------------------------------------------------------------
# Transactions endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/credit/transactions",
    tags=["credit management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_transactions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    tx_type: Optional[str] = None,
    wallet_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tenant_id: Optional[str] = Query(None, description="Admin only: override tenant"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List the current tenant's transactions. Admins may pass ``tenant_id``."""
    prisma_client = _get_prisma_client()
    resolved = _resolve_tenant_id(user_api_key_dict, override=tenant_id)

    where: Dict[str, Any] = {"tenant_id": resolved, "NOT": {"tx_type": "pre_deduct"}}
    if tx_type:
        where["tx_type"] = tx_type
    if wallet_type:
        where["wallet_type"] = wallet_type

    created_filter: Dict[str, Any] = {}
    if start_date:
        created_filter["gte"] = _parse_date(start_date)
    if end_date:
        created_filter["lte"] = _parse_date(end_date, end_of_day=True)
    if created_filter:
        where["created_at"] = created_filter

    total = await prisma_client.db.credittransaction.count(where=where)
    rows = await prisma_client.db.credittransaction.find_many(
        where=where,
        order={"created_at": "desc"},
        skip=(page - 1) * page_size,
        take=page_size,
    )

    return {
        "data": [_serialize_transaction(_dump(r) or {}) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


def _parse_date(value: str, end_of_day: bool = False) -> datetime:
    """Parse ISO date or yyyy-mm-dd. Falls back to start/end of day if no time."""
    try:
        if "T" in value:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        d = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        if end_of_day:
            d = d.replace(hour=23, minute=59, second=59, microsecond=999999)
        return d
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": f"Invalid date: {value}"})


# ---------------------------------------------------------------------------
# Recharge endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/credit/recharge",
    tags=["credit management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_recharge_order(
    payload: RechargeRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Simulated recharge — credits the paid balance immediately.

    A production integration would issue a payment order and credit the
    wallet on the payment provider's webhook. This endpoint short-circuits
    that for development.
    """
    prisma_client = _get_prisma_client()
    tenant_id = _resolve_tenant_id(user_api_key_dict)

    pkg_row = await prisma_client.db.creditpackage.find_unique(
        where={"id": payload.package_id}
    )
    pkg = _dump(pkg_row)
    if pkg is None:
        raise HTTPException(status_code=404, detail={"error": "Package not found"})
    if not pkg.get("is_active"):
        raise HTTPException(status_code=400, detail={"error": "Package is inactive"})

    credits_amount = _coerce_decimal(pkg.get("credits_amount"))
    bonus = _coerce_decimal(pkg.get("bonus_credits"))
    total_credits = credits_amount + bonus

    wallet = await _ensure_wallet_with_gift(prisma_client, tenant_id)
    new_paid = _coerce_decimal(wallet.get("paid_balance")) + total_credits

    await prisma_client.db.creditwallet.update(
        where={"tenant_id": tenant_id},
        data={
            "paid_balance": new_paid,
            "total_recharged": _coerce_decimal(wallet.get("total_recharged"))
            + total_credits,
            "updated_at": datetime.now(timezone.utc),
        },
    )
    await prisma_client.db.credittransaction.create(
        data={
            "tenant_id": tenant_id,
            "tx_type": _TX_PURCHASE,
            "wallet_type": _PAID,
            "amount": total_credits,
            "balance_after": new_paid,
            "description": f"充值套餐「{pkg.get('name')}」，到账 {total_credits:.2f} C",
            "metadata": {
                "package_id": str(pkg.get("id")),
                "package_name": pkg.get("name"),
                "price_cny": _coerce_decimal(pkg.get("price_cny")),
                "credits_amount": credits_amount,
                "bonus_credits": bonus,
                "payment_method": payload.payment_method,
            },
        }
    )

    return {"success": True, "credits": total_credits, "package_name": pkg.get("name")}


# ---------------------------------------------------------------------------
# Package CRUD
# ---------------------------------------------------------------------------


_DEFAULT_PACKAGES: List[Dict[str, Any]] = [
    {
        "id": "default-1",
        "name": "入门包",
        "credits_amount": 400,
        "price_cny": 30,
        "bonus_credits": 0,
        "badge_text": None,
        "sort_order": 1,
        "is_active": True,
    },
    {
        "id": "default-2",
        "name": "标准包",
        "credits_amount": 1400,
        "price_cny": 98,
        "bonus_credits": 0,
        "badge_text": "热销",
        "sort_order": 2,
        "is_active": True,
    },
    {
        "id": "default-3",
        "name": "大额包",
        "credits_amount": 4500,
        "price_cny": 298,
        "bonus_credits": 0,
        "badge_text": "最划算",
        "sort_order": 3,
        "is_active": True,
    },
]


@router.get(
    "/credit/packages",
    tags=["credit management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_packages():
    """List active packages. Falls back to a hard-coded default set if the
    ``credit_package`` table is empty so the recharge UI has something to show."""
    prisma_client = _get_prisma_client()
    rows = await prisma_client.db.creditpackage.find_many(
        where={"is_active": True},
        order=[{"sort_order": "asc"}, {"price_cny": "asc"}],
    )
    serialized = [_serialize_package(_dump(r) or {}) for r in rows]
    if not serialized:
        return {"data": _DEFAULT_PACKAGES, "default": True}
    return {"data": serialized, "default": False}


@router.post(
    "/credit/packages",
    tags=["credit management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_package(
    payload: PackagePayload,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    prisma_client = _get_prisma_client()
    created = await prisma_client.db.creditpackage.create(
        data={
            "name": payload.name,
            "credits_amount": payload.credits_amount,
            "price_cny": payload.price_cny,
            "bonus_credits": payload.bonus_credits or 0,
            "badge_text": payload.badge_text,
            "sort_order": payload.sort_order or 0,
            "is_active": payload.is_active if payload.is_active is not None else True,
        }
    )
    return _serialize_package(_dump(created) or {})


@router.put(
    "/credit/packages/{package_id}",
    tags=["credit management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_package(
    package_id: str,
    payload: PackagePayload,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    prisma_client = _get_prisma_client()
    updated = await prisma_client.db.creditpackage.update(
        where={"id": package_id},
        data={
            "name": payload.name,
            "credits_amount": payload.credits_amount,
            "price_cny": payload.price_cny,
            "bonus_credits": payload.bonus_credits or 0,
            "badge_text": payload.badge_text,
            "sort_order": payload.sort_order or 0,
            "is_active": payload.is_active if payload.is_active is not None else True,
        },
    )
    return _serialize_package(_dump(updated) or {})


@router.delete(
    "/credit/packages/{package_id}",
    tags=["credit management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_package(
    package_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    prisma_client = _get_prisma_client()
    await prisma_client.db.creditpackage.delete(where={"id": package_id})
    return {"success": True}
