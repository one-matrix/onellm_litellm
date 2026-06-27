"""Resolve the OneLLM tenant that billing/pay endpoints should use.

Credit wallets are keyed by the LiteLLM team id derived from a OneLLM
``sys_tenants`` row. The browser sends the active OneLLM tenant id as
``X-Tenant-Id``; the Next.js proxy also forwards the server-side OneLLM access
token in ``X-OneLLM-Access-Token`` so this module can verify membership before
trusting that header.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from onellm.auth.jwt import decode_access_token
from onellm.exceptions import TokenInvalid

ONELLM_ACCESS_TOKEN_HEADER = "X-OneLLM-Access-Token"
ONELLM_AUTH_PROVIDER_HEADER = "X-OneLLM-Auth-Provider"
ONELLM_TENANT_HEADER = "X-Tenant-Id"


def is_litellm_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN


def _dump(row: Any) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


async def _find_tenant(db: Any, tenant_id: str) -> Dict[str, Any]:
    tenant = _dump(await db.systenant.find_unique(where={"id": tenant_id}))
    if tenant is None and hasattr(db.systenant, "find_first"):
        tenant = _dump(
            await db.systenant.find_first(where={"litellm_team_id": tenant_id})
        )
    if tenant is None or tenant.get("is_deleted") or not tenant.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Tenant not found"},
        )
    return tenant


async def _assert_onellm_tenant_access(
    db: Any,
    *,
    request: Request,
    tenant: Dict[str, Any],
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    access_token = request.headers.get(ONELLM_ACCESS_TOKEN_HEADER)
    auth_provider = request.headers.get(ONELLM_AUTH_PROVIDER_HEADER)

    # The local master-key login has no OneLLM JWT, but it is intentionally a
    # platform-admin session. The proxy overwrites this header, so it is not
    # accepted from the browser as-is.
    if not access_token:
        if auth_provider == "master-key" and is_litellm_admin(user_api_key_dict):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "OneLLM access token required for tenant-scoped billing"},
        )

    try:
        payload = decode_access_token(access_token)
    except TokenInvalid as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "OneLLM access token missing subject"},
        )

    user = _dump(await db.sysuser.find_unique(where={"id": user_id}))
    if user is None or user.get("is_deleted") or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "OneLLM user is inactive or removed"},
        )
    if payload.get("stamp") and payload["stamp"] != user.get("security_stamp"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "OneLLM access token has been superseded"},
        )

    if user.get("role_global") == "root":
        return

    membership = await db.sysuserrole.find_first(
        where={"user_id": user_id, "tenant_id": tenant["id"]}
    )
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "User is not a member of this tenant"},
        )


async def resolve_credit_tenant_id(
    *,
    prisma_client: Any,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    override: Optional[str] = None,
    default_tenant: str = "default",
) -> str:
    """Return the wallet key for the active billing tenant.

    Priority:
    1. explicit admin override (existing admin APIs)
    2. current OneLLM tenant header, after membership validation
    3. legacy LiteLLM team/user fallback
    """
    if override and is_litellm_admin(user_api_key_dict):
        return override

    header_tenant_id = request.headers.get(ONELLM_TENANT_HEADER)
    if header_tenant_id:
        tenant = await _find_tenant(prisma_client.db, header_tenant_id)
        await _assert_onellm_tenant_access(
            prisma_client.db,
            request=request,
            tenant=tenant,
            user_api_key_dict=user_api_key_dict,
        )
        return str(tenant.get("litellm_team_id") or tenant["id"])

    return user_api_key_dict.team_id or user_api_key_dict.user_id or default_tenant
