"""
PAY MANAGEMENT

Generic payment platform — channel-agnostic + product-agnostic.

Tables live in the ``app`` schema (separate from upstream ``public``):

  - ``app.pay_order``        canonical order record
  - ``app.pay_notify_log``   append-only channel callback audit log
  - ``app.pay_refund``       refund records (supports partial refunds)

Prisma accessors are ``prisma_client.db.payorder`` /
``paynotifylog`` / ``payrefund`` (no ``LiteLLM_`` prefix — onellm extensions).

Channel adapters live in ``litellm/proxy/pay_service/channels/``. This module
ships a ``stub`` channel that simulates payment success so the recharge flow
works end-to-end without an SDK; real adapters (alipay/wechat/stripe) plug in
via ``pay_service/registry.py``.

Product settlement (e.g. crediting the wallet on a successful ``credits``
order) is dispatched by ``product_type`` — currently only ``credits`` is
implemented (mirrors ``credit_management_endpoints.recharge``).

Endpoints:

  POST   /pay/order                       create + return payment payload
  GET    /pay/order/{id}                  query order (owner or admin)
  POST   /pay/order/{id}/refresh          force-pull status from channel
  GET    /pay/orders                      admin: list orders with filters
  GET    /pay/orders/{id}                 admin: order detail + refunds
  POST   /pay/orders/{id}/refund          admin: refund (partial or full)
  GET    /pay/channels                    list supported channels
  POST   /pay/callback/{channel}          channel async notify (no auth)
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import random
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pay_service import get_channel, is_channel_supported, supported_channels
from litellm.proxy.pay_service.channels.types import CreatePaymentInput

router = APIRouter()


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_MIN = 15
_DEFAULT_TENANT = "default"
_CREDITS_PRODUCT = "credits"
_RECHARGE_CATEGORY = "recharge"

# Order status enum (kept as plain strings to match the SQL design).
_S_PENDING = "pending"
_S_PAID = "paid"
_S_CLOSED = "closed"
_S_REFUNDED = "refunded"
_S_PARTIAL_REFUNDED = "partial_refunded"
_S_FAILED = "failed"

# Refund status enum.
_R_PENDING = "pending"
_R_PROCESSING = "processing"
_R_SUCCESS = "success"
_R_FAILED = "failed"


def _get_prisma_client():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "pay_service: prisma_client is not initialised — DATABASE_URL required"},
        )
    return prisma_client


def _is_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if not _is_admin(user_api_key_dict):
        raise HTTPException(status_code=403, detail={"error": "Admin role required"})


def _resolve_tenant(user_api_key_dict: UserAPIKeyAuth) -> Optional[str]:
    return user_api_key_dict.team_id or user_api_key_dict.user_id or _DEFAULT_TENANT


def _resolve_user_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    if not user_api_key_dict.user_id:
        raise HTTPException(status_code=401, detail={"error": "Authenticated user_id required"})
    return user_api_key_dict.user_id


def _dump(row: Any) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return row.model_dump() if hasattr(row, "model_dump") else dict(row)


def _generate_out_trade_no(prefix: str = "NK") -> str:
    """Match the frontend ``generateOutTradeNo`` shape: prefix + yyyymmddhhmmss + 4-digit rand."""
    now = datetime.now()
    rand = random.randint(1000, 9999)
    return f"{prefix}{now.strftime('%Y%m%d%H%M%S')}{rand}"


def _generate_out_refund_no(out_trade_no: str) -> str:
    return f"R{secrets.token_hex(6)}{out_trade_no[-6:]}"[:64]


# ---------------------------------------------------------------------------
# Pydantic request shapes
# ---------------------------------------------------------------------------


class CreateOrderRequest(BaseModel):
    channel: str = Field(..., description="alipay | wechat | stripe | stub | ...")
    product_type: str = Field(..., description="'credits' | 'subscription' | 'goods' | ...")
    product_id: Optional[str] = None
    product_name: str
    quantity: int = 1
    unit: Optional[str] = None
    billing_cycle: Optional[str] = None
    product_meta: Optional[Dict[str, Any]] = None
    amount_cny: float
    timeout_minutes: Optional[int] = None
    return_url: Optional[str] = None
    notify_url: Optional[str] = None
    remark: Optional[str] = None
    financial_category: Optional[str] = None


class CreateRechargeRequest(BaseModel):
    """Convenience wrapper for product_type=credits."""
    channel: str
    package_id: Optional[str] = None
    custom_amount: Optional[float] = None
    return_url: Optional[str] = None
    notify_url: Optional[str] = None


class RefundRequest(BaseModel):
    amount_cny: float
    reason: Optional[str] = ""


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------


def _serialize_order(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(row.get("id")) if row.get("id") is not None else None,
        "out_trade_no": row.get("out_trade_no"),
        "channel": row.get("channel"),
        "channel_trade_no": row.get("channel_trade_no"),
        "tenant_id": row.get("tenant_id"),
        "user_id": row.get("user_id"),
        "product_type": row.get("product_type"),
        "product_id": row.get("product_id"),
        "product_name": row.get("product_name"),
        "quantity": row.get("quantity"),
        "unit": row.get("unit"),
        "billing_cycle": row.get("billing_cycle"),
        "product_meta": row.get("product_meta") or {},
        "amount_cny": float(row.get("amount_cny") or 0),
        "paid_amount_cny": (
            float(row["paid_amount_cny"]) if row.get("paid_amount_cny") is not None else None
        ),
        "currency": row.get("currency"),
        "status": row.get("status"),
        "paid_at": row.get("paid_at"),
        "expire_at": row.get("expire_at"),
        "pay_account": row.get("pay_account"),
        "financial_category": row.get("financial_category"),
        "remark": row.get("remark"),
        "client_ip": row.get("client_ip"),
        "metadata": row.get("metadata") or {},
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "is_deleted": bool(row.get("is_deleted")),
    }


def _serialize_refund(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(row.get("id")) if row.get("id") is not None else None,
        "order_id": str(row.get("order_id")) if row.get("order_id") is not None else None,
        "out_trade_no": row.get("out_trade_no"),
        "out_refund_no": row.get("out_refund_no"),
        "channel": row.get("channel"),
        "channel_refund_no": row.get("channel_refund_no"),
        "refund_amount_cny": float(row.get("refund_amount_cny") or 0),
        "currency": row.get("currency"),
        "reason": row.get("reason"),
        "status": row.get("status"),
        "error_message": row.get("error_message"),
        "business_settled": bool(row.get("business_settled")),
        "settlement_meta": row.get("settlement_meta") or {},
        "operator_id": row.get("operator_id"),
        "refunded_at": row.get("refunded_at"),
        "created_at": row.get("created_at"),
    }


# ---------------------------------------------------------------------------
# Product settlement — credits
# ---------------------------------------------------------------------------


async def _settle_credits_purchase(prisma_client, order: Dict[str, Any]) -> None:
    """Credit the paid wallet + write a ``purchase`` transaction. Idempotent
    against re-entry: callers must check ``order.status === pending`` before
    calling, since this function unconditionally credits."""
    meta = order.get("product_meta") or {}
    credits_amount = float(meta.get("credits_amount") or 0)
    bonus_credits = float(meta.get("bonus_credits") or 0)
    total = credits_amount + bonus_credits
    if total <= 0:
        return

    tenant_id = order.get("tenant_id") or _DEFAULT_TENANT

    wallet = await prisma_client.db.creditwallet.find_unique(where={"tenant_id": tenant_id})
    if wallet is None:
        wallet_created = await prisma_client.db.creditwallet.create(
            data={"tenant_id": tenant_id, "paid_balance": total, "total_recharged": total}
        )
        wallet = wallet_created
    else:
        w = _dump(wallet) or {}
        new_paid = float(w.get("paid_balance") or 0) + total
        new_recharged = float(w.get("total_recharged") or 0) + total
        await prisma_client.db.creditwallet.update(
            where={"tenant_id": tenant_id},
            data={
                "paid_balance": new_paid,
                "total_recharged": new_recharged,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        wallet = await prisma_client.db.creditwallet.find_unique(where={"tenant_id": tenant_id})

    w = _dump(wallet) or {}
    new_paid_balance = float(w.get("paid_balance") or 0)

    await prisma_client.db.credittransaction.create(
        data={
            "tenant_id": tenant_id,
            "tx_type": "purchase",
            "wallet_type": "paid",
            "amount": total,
            "balance_after": new_paid_balance,
            "description": f"充值 {order.get('product_name')} 到账 {total:.2f} C",
            "metadata": {
                "order_id": str(order.get("id")),
                "out_trade_no": order.get("out_trade_no"),
                "package_id": meta.get("package_id"),
                "credits_amount": credits_amount,
                "bonus_credits": bonus_credits,
            },
        }
    )


async def _settle_credits_refund(prisma_client, order: Dict[str, Any], refund_amount_cny: float) -> Dict[str, Any]:
    """Reverse the corresponding credits from ``paid_balance`` (only). Returns
    a settlement_meta dict to persist on the refund row."""
    tenant_id = order.get("tenant_id") or _DEFAULT_TENANT
    meta = order.get("product_meta") or {}
    credits_amount = float(meta.get("credits_amount") or 0)
    bonus_credits = float(meta.get("bonus_credits") or 0)
    total_credits = credits_amount + bonus_credits

    # Pro-rate refund credits against the original purchase ratio.
    order_amount = float(order.get("amount_cny") or 0) or 1
    refund_credits = round((refund_amount_cny / order_amount) * total_credits, 2)

    wallet = _dump(await prisma_client.db.creditwallet.find_unique(where={"tenant_id": tenant_id}))
    if wallet is None:
        return {"refund_credits": refund_credits, "deducted": 0, "shortfall": refund_credits}

    paid = float(wallet.get("paid_balance") or 0)
    deducted = min(paid, refund_credits)
    shortfall = max(0.0, refund_credits - deducted)
    new_paid = paid - deducted

    await prisma_client.db.creditwallet.update(
        where={"tenant_id": tenant_id},
        data={"paid_balance": new_paid, "updated_at": datetime.now(timezone.utc)},
    )
    await prisma_client.db.credittransaction.create(
        data={
            "tenant_id": tenant_id,
            "tx_type": "refund",
            "wallet_type": "paid",
            "amount": -deducted,
            "balance_after": new_paid,
            "description": f"订单 {order.get('out_trade_no')} 退款 {deducted:.2f} C",
            "metadata": {
                "order_id": str(order.get("id")),
                "out_trade_no": order.get("out_trade_no"),
                "refund_amount_cny": refund_amount_cny,
            },
        }
    )
    return {"refund_credits": refund_credits, "deducted": deducted, "shortfall": shortfall}


async def _settle_by_product_type(prisma_client, order: Dict[str, Any]) -> None:
    if order.get("product_type") == _CREDITS_PRODUCT:
        await _settle_credits_purchase(prisma_client, order)


# ---------------------------------------------------------------------------
# Internal: mark order paid (used by both create-immediate flow and callbacks)
# ---------------------------------------------------------------------------


async def _mark_paid_and_settle(
    prisma_client,
    order: Dict[str, Any],
    *,
    paid_amount_cny: Optional[float] = None,
    channel_trade_no: Optional[str] = None,
    pay_account: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically mark an order paid (idempotent) and settle the product."""
    if order.get("status") != _S_PENDING:
        # Already settled — no-op.
        return order

    patch: Dict[str, Any] = {
        "status": _S_PAID,
        "paid_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    if paid_amount_cny is not None:
        patch["paid_amount_cny"] = paid_amount_cny
    if channel_trade_no:
        patch["channel_trade_no"] = channel_trade_no
    if pay_account:
        patch["pay_account"] = pay_account

    updated = await prisma_client.db.payorder.update(where={"id": order["id"]}, data=patch)
    order_updated = _dump(updated) or order
    await _settle_by_product_type(prisma_client, order_updated)
    return order_updated


# ---------------------------------------------------------------------------
# Endpoints — user-facing
# ---------------------------------------------------------------------------


@router.get(
    "/pay/channels",
    tags=["pay management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_channels():
    return {"channels": supported_channels()}


@router.post(
    "/pay/order",
    tags=["pay management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_order(
    payload: CreateOrderRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Create a generic pay order and return the channel payment payload."""
    if not is_channel_supported(payload.channel):
        raise HTTPException(status_code=400, detail={"error": f"Unsupported channel: {payload.channel}"})
    if payload.amount_cny <= 0:
        raise HTTPException(status_code=400, detail={"error": "amount_cny must be positive"})

    prisma_client = _get_prisma_client()
    user_id = _resolve_user_id(user_api_key_dict)
    tenant_id = _resolve_tenant(user_api_key_dict)
    out_trade_no = _generate_out_trade_no()
    timeout = payload.timeout_minutes or _DEFAULT_TIMEOUT_MIN
    expire_at = datetime.now(timezone.utc) + timedelta(minutes=timeout)
    client_ip = (request.client.host if request.client else None) or None

    created = await prisma_client.db.payorder.create(
        data={
            "out_trade_no": out_trade_no,
            "channel": payload.channel,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "product_type": payload.product_type,
            "product_id": payload.product_id,
            "product_name": payload.product_name,
            "quantity": payload.quantity,
            "unit": payload.unit,
            "billing_cycle": payload.billing_cycle,
            "product_meta": payload.product_meta or {},
            "amount_cny": payload.amount_cny,
            "currency": "CNY",
            "status": _S_PENDING,
            "expire_at": expire_at,
            "financial_category": payload.financial_category,
            "remark": payload.remark,
            "client_ip": client_ip,
            "metadata": {},
            "created_by": user_id,
        }
    )
    order = _dump(created) or {}

    channel = get_channel(payload.channel)
    pay_result = channel.create_payment(
        CreatePaymentInput(
            out_trade_no=out_trade_no,
            amount_cny=payload.amount_cny,
            subject=payload.product_name,
            notify_url=payload.notify_url,
            return_url=payload.return_url,
            timeout_minutes=timeout,
            client_ip=client_ip,
        )
    )

    return {
        "order_id": str(order.get("id")),
        "out_trade_no": out_trade_no,
        "payment_type": pay_result.type,
        "payload": pay_result.payload,
        "expire_at": order.get("expire_at"),
    }


# Default recharge packages — surfaced when ``credit_package`` table is empty.
_DEFAULT_PACKAGES: List[Dict[str, Any]] = [
    {"id": "default-30",   "name": "体验包", "price_cny": 30,   "credits_amount": 30,   "bonus_credits": 0, "badge_text": None},
    {"id": "default-100",  "name": "标准包", "price_cny": 100,  "credits_amount": 100,  "bonus_credits": 0, "badge_text": "常用"},
    {"id": "default-500",  "name": "大额包", "price_cny": 500,  "credits_amount": 500,  "bonus_credits": 0, "badge_text": None},
    {"id": "default-1000", "name": "超大包", "price_cny": 1000, "credits_amount": 1000, "bonus_credits": 0, "badge_text": None},
]


async def _resolve_package(prisma_client, package_id: str) -> Optional[Dict[str, Any]]:
    if package_id.startswith("default-"):
        for p in _DEFAULT_PACKAGES:
            if p["id"] == package_id:
                return dict(p)
        return None
    row = await prisma_client.db.creditpackage.find_unique(where={"id": package_id})
    return _dump(row)


@router.post(
    "/pay/recharge",
    tags=["pay management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_recharge_order(
    payload: CreateRechargeRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Convenience wrapper around ``/pay/order`` for product_type=credits."""
    if not is_channel_supported(payload.channel):
        raise HTTPException(status_code=400, detail={"error": f"Unsupported channel: {payload.channel}"})

    prisma_client = _get_prisma_client()

    amount_cny: float
    credits_amount: float
    bonus_credits: float = 0.0
    package_id: Optional[str] = None
    subject: str

    if payload.package_id:
        pkg = await _resolve_package(prisma_client, payload.package_id)
        if not pkg:
            raise HTTPException(status_code=404, detail={"error": "Package not found or inactive"})
        amount_cny = float(pkg["price_cny"])
        credits_amount = float(pkg["credits_amount"])
        bonus_credits = float(pkg.get("bonus_credits") or 0)
        package_id = str(pkg["id"])
        subject = f"充值套餐：{pkg['name']}"
    elif payload.custom_amount is not None:
        amt = int(payload.custom_amount)
        if amt < 1 or amt > 5000:
            raise HTTPException(status_code=400, detail={"error": "custom_amount must be 1..5000"})
        amount_cny = float(amt)
        credits_amount = float(amt)
        subject = f"充值 {amt} 积分"
    else:
        raise HTTPException(status_code=400, detail={"error": "Either package_id or custom_amount is required"})

    return await create_order(
        CreateOrderRequest(
            channel=payload.channel,
            product_type=_CREDITS_PRODUCT,
            product_id=package_id,
            product_name=subject,
            quantity=1,
            unit="积分",
            product_meta={
                "credits_amount": credits_amount,
                "bonus_credits": bonus_credits,
                "package_id": package_id,
            },
            amount_cny=amount_cny,
            financial_category=_RECHARGE_CATEGORY,
            notify_url=payload.notify_url,
            return_url=payload.return_url,
        ),
        request,
        user_api_key_dict,
    )


@router.get(
    "/pay/order/{order_id}",
    tags=["pay management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_order(
    order_id: str,
    refresh: bool = Query(False),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    prisma_client = _get_prisma_client()
    row = _dump(await prisma_client.db.payorder.find_unique(where={"id": order_id}))
    if not row or row.get("is_deleted"):
        raise HTTPException(status_code=404, detail={"error": "Order not found"})

    is_admin = _is_admin(user_api_key_dict)
    if not is_admin and row.get("user_id") != user_api_key_dict.user_id:
        raise HTTPException(status_code=403, detail={"error": "Not your order"})

    if refresh and row.get("status") == _S_PENDING:
        try:
            channel = get_channel(row["channel"])
            result = channel.query_order(row["out_trade_no"])
            if result and result.trade_status.lower() in ("paid", "trade_success", "trade_finished"):
                row = await _mark_paid_and_settle(
                    prisma_client,
                    row,
                    paid_amount_cny=result.paid_amount_cny,
                    channel_trade_no=result.channel_trade_no,
                    pay_account=result.pay_account,
                )
        except Exception as exc:  # noqa: BLE001
            # Refresh is best-effort; surface failure as 200 with stale state.
            return {"order": _serialize_order(row), "refresh_error": str(exc)}

    return {"order": _serialize_order(row)}


# ---------------------------------------------------------------------------
# Endpoints — admin
# ---------------------------------------------------------------------------


@router.get(
    "/pay/orders",
    tags=["pay management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_orders(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    status: Optional[str] = None,
    channel: Optional[str] = None,
    product_type: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    out_trade_no: Optional[str] = None,
    include_deleted: bool = False,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    prisma_client = _get_prisma_client()

    where: Dict[str, Any] = {}
    if not include_deleted:
        where["is_deleted"] = False
    if status:
        where["status"] = status
    if channel:
        where["channel"] = channel
    if product_type:
        where["product_type"] = product_type
    if user_id:
        where["user_id"] = user_id
    if tenant_id:
        where["tenant_id"] = tenant_id
    if out_trade_no:
        where["out_trade_no"] = out_trade_no

    total = await prisma_client.db.payorder.count(where=where or None)  # type: ignore[arg-type]
    rows = await prisma_client.db.payorder.find_many(
        where=where or None,  # type: ignore[arg-type]
        order={"created_at": "desc"},
        skip=(page - 1) * page_size,
        take=page_size,
    )
    return {
        "data": [_serialize_order(_dump(r) or {}) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


@router.get(
    "/pay/orders/{order_id}",
    tags=["pay management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_order_detail(
    order_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    prisma_client = _get_prisma_client()

    order = _dump(await prisma_client.db.payorder.find_unique(where={"id": order_id}))
    if not order or order.get("is_deleted"):
        raise HTTPException(status_code=404, detail={"error": "Order not found"})

    refunds = await prisma_client.db.payrefund.find_many(
        where={"order_id": order_id, "is_deleted": False},
        order={"created_at": "desc"},
    )
    refund_rows = [_serialize_refund(_dump(r) or {}) for r in refunds]
    refunded_amount = sum(
        float(r["refund_amount_cny"]) for r in refund_rows if r["status"] == _R_SUCCESS
    )

    return {
        "order": _serialize_order(order),
        "refunds": refund_rows,
        "refunded_amount": refunded_amount,
        "remaining_amount": max(0.0, float(order.get("amount_cny") or 0) - refunded_amount),
    }


@router.post(
    "/pay/orders/{order_id}/refund",
    tags=["pay management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def refund_order(
    order_id: str,
    payload: RefundRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    prisma_client = _get_prisma_client()

    order = _dump(await prisma_client.db.payorder.find_unique(where={"id": order_id}))
    if not order or order.get("is_deleted"):
        raise HTTPException(status_code=404, detail={"error": "Order not found"})
    if order.get("status") not in (_S_PAID, _S_PARTIAL_REFUNDED):
        raise HTTPException(status_code=400, detail={"error": f"Cannot refund order in status {order.get('status')}"})

    if payload.amount_cny <= 0:
        raise HTTPException(status_code=400, detail={"error": "Refund amount must be positive"})

    # Compute remaining refundable amount.
    existing_refunds = await prisma_client.db.payrefund.find_many(
        where={"order_id": order_id, "is_deleted": False}
    )
    refunded_total = sum(
        float((_dump(r) or {}).get("refund_amount_cny") or 0)
        for r in existing_refunds
        if (_dump(r) or {}).get("status") == _R_SUCCESS
    )
    amount_cny = float(order["amount_cny"])
    remaining = amount_cny - refunded_total
    if payload.amount_cny > remaining + 1e-6:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Refund amount exceeds remaining (¥{remaining:.2f})"},
        )

    out_refund_no = _generate_out_refund_no(order["out_trade_no"])
    refund_created = await prisma_client.db.payrefund.create(
        data={
            "order_id": order_id,
            "out_trade_no": order["out_trade_no"],
            "out_refund_no": out_refund_no,
            "channel": order["channel"],
            "refund_amount_cny": payload.amount_cny,
            "currency": order.get("currency") or "CNY",
            "reason": payload.reason or "",
            "status": _R_PROCESSING,
            "operator_id": user_api_key_dict.user_id or "admin",
            "created_by": user_api_key_dict.user_id,
        }
    )
    refund = _dump(refund_created) or {}

    # Call channel refund API.
    try:
        channel = get_channel(order["channel"])
        channel_result = channel.refund(
            out_trade_no=order["out_trade_no"],
            out_refund_no=out_refund_no,
            amount_cny=payload.amount_cny,
            reason=payload.reason or "",
        )
        if (channel_result.get("fund_change") or "").upper() != "Y":
            raise RuntimeError(f"Channel refused refund: {channel_result}")
    except Exception as exc:  # noqa: BLE001
        await prisma_client.db.payrefund.update(
            where={"id": refund["id"]},
            data={
                "status": _R_FAILED,
                "error_message": str(exc),
                "updated_at": datetime.now(timezone.utc),
            },
        )
        raise HTTPException(status_code=500, detail={"error": f"Refund failed: {exc}"})

    # Channel succeeded — settle business side.
    settlement_meta: Dict[str, Any] = {}
    if order.get("product_type") == _CREDITS_PRODUCT:
        settlement_meta = await _settle_credits_refund(prisma_client, order, payload.amount_cny)

    await prisma_client.db.payrefund.update(
        where={"id": refund["id"]},
        data={
            "status": _R_SUCCESS,
            "channel_refund_no": channel_result.get("channel_refund_no"),
            "business_settled": True,
            "settlement_meta": settlement_meta,
            "refunded_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        },
    )

    # Update order status (refunded vs partial_refunded).
    new_refunded = refunded_total + payload.amount_cny
    new_status = _S_REFUNDED if abs(new_refunded - amount_cny) < 1e-6 else _S_PARTIAL_REFUNDED
    await prisma_client.db.payorder.update(
        where={"id": order_id},
        data={"status": new_status, "updated_at": datetime.now(timezone.utc)},
    )

    return {
        "success": True,
        "out_refund_no": out_refund_no,
        "refund_amount_cny": payload.amount_cny,
        "settlement_meta": settlement_meta,
    }


# ---------------------------------------------------------------------------
# Channel callback — no auth (signature-validated within handler)
# ---------------------------------------------------------------------------


@router.post(
    "/pay/callback/{channel_code}",
    tags=["pay management"],
)
async def channel_callback(
    channel_code: str,
    request: Request,
):
    """Generic async notify endpoint. The channel adapter is responsible for
    verifying signatures; this handler logs the raw payload either way.
    Returns plain text per most channel conventions (alipay expects
    ``success``)."""
    prisma_client = _get_prisma_client()

    try:
        channel = get_channel(channel_code)
    except ValueError:
        raise HTTPException(status_code=404, detail={"error": f"Unknown channel: {channel_code}"})

    # Accept either application/json or application/x-www-form-urlencoded.
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        params: Dict[str, Any] = await request.json()
    else:
        form = await request.form()
        params = {k: v for k, v in form.items()}

    signature_valid = False
    try:
        signature_valid = channel.verify_notify(params)
    except Exception:
        signature_valid = False

    client_ip = request.client.host if request.client else None

    # Best-effort parse — used to back-link the log entry to an order.
    parsed = None
    try:
        parsed = channel.parse_notify(params)
    except Exception:
        parsed = None

    # Match the order by out_trade_no (if present).
    order_row = None
    out_trade_no = parsed.out_trade_no if parsed else None
    if out_trade_no:
        order_row = _dump(
            await prisma_client.db.payorder.find_unique(where={"out_trade_no": out_trade_no})
        )

    # Always log the callback.
    await prisma_client.db.paynotifylog.create(
        data={
            "channel": channel_code,
            "order_id": order_row["id"] if order_row else None,
            "out_trade_no": out_trade_no,
            "channel_trade_no": parsed.channel_trade_no if parsed else None,
            "trade_status": parsed.trade_status if parsed else None,
            "raw_body": params,
            "signature_valid": signature_valid,
            "processed": False,
            "client_ip": client_ip,
        }
    )

    if not signature_valid:
        raise HTTPException(status_code=400, detail={"error": "Invalid signature"})

    if not order_row:
        return "success"  # Log written; nothing to settle.

    # Channel-specific "is paid" check. For the stub channel we accept any
    # truthy status; for real adapters this should be tightened.
    if parsed and parsed.trade_status.lower() in ("paid", "trade_success", "trade_finished"):
        await _mark_paid_and_settle(
            prisma_client,
            order_row,
            paid_amount_cny=parsed.paid_amount_cny,
            channel_trade_no=parsed.channel_trade_no,
            pay_account=parsed.pay_account,
        )
        await prisma_client.db.paynotifylog.update_many(
            where={"out_trade_no": out_trade_no, "processed": False},
            data={"processed": True},
        )

    return "success"
