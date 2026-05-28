"""HTTP routes for tenants and tenant memberships."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, status

from onellm.db import get_prisma
from onellm.deps import CurrentIdentity, get_current_user, require_tenant_scope
from onellm.exceptions import PermissionDenied
from onellm.schemas.tenant import (
    TenantCreateIn,
    TenantMemberIn,
    TenantMemberUpdateIn,
    TenantOut,
)
from onellm.tenant import service as tenant_service

router = APIRouter(prefix="/tenants", tags=["onellm-tenant"])


def _tenant_to_out(t: Any) -> TenantOut:
    return TenantOut(
        id=t.id,
        name=t.name,
        code=t.code,
        domain=t.domain,
        plan_code=t.plan_code,
        default_price_markup=t.default_price_markup,
        is_active=t.is_active,
        litellm_team_id=t.litellm_team_id,
        created_at=t.created_at,
    )


async def _require_tenant_role(
    identity: CurrentIdentity, tenant_id: str, allowed: set[str]
) -> None:
    """Inline guard — checks the caller has one of ``allowed`` codes in ``tenant_id``.

    Used by member-management endpoints where the tenant_id comes from the
    path, not from the X-Tenant-Id header (so require_permission isn't enough).
    """
    if identity.role_global == "root":
        return
    db = get_prisma()
    memberships = await db.sysuserrole.find_many(
        where={"user_id": identity.user_id, "tenant_id": tenant_id},
        include={"role": True},
    )
    codes = {m.role.code for m in memberships if m.role is not None}
    if not (codes & allowed):
        raise PermissionDenied(f"Requires one of tenant roles: {sorted(allowed)}")


@router.get("", response_model=List[Dict[str, Any]])
async def list_my_tenants(
    identity: CurrentIdentity = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    db = get_prisma()
    return await tenant_service.list_my_memberships(db, user_id=identity.user_id)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=TenantOut)
async def create_tenant(
    payload: TenantCreateIn,
    identity: CurrentIdentity = Depends(get_current_user),
) -> TenantOut:
    # Any authenticated user can create a new tenant. The creator becomes the
    # tenant_admin (full rights including tenant.delete) and pays for the
    # seat the same way the registration flow creates the user's first tenant.
    db = get_prisma()
    tenant = await tenant_service.create_tenant(
        db,
        name=payload.name,
        code=payload.code,
        plan_code=payload.plan_code or "free",
        owner_user_id=identity.user_id,
    )
    return _tenant_to_out(tenant)


@router.get("/{tenant_id}/members")
async def list_members(
    tenant_id: str,
    identity: CurrentIdentity = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    await _require_tenant_role(
        identity,
        tenant_id,
        {"tenant_admin", "billing", "viewer", "user"},
    )
    db = get_prisma()
    return await tenant_service.list_members(db, tenant_id=tenant_id)


@router.post("/{tenant_id}/members", status_code=status.HTTP_201_CREATED)
async def invite_member(
    tenant_id: str,
    payload: TenantMemberIn,
    identity: CurrentIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    await _require_tenant_role(identity, tenant_id, {"tenant_admin"})
    db = get_prisma()
    return await tenant_service.invite_member(
        db, tenant_id=tenant_id, email=payload.email, role_code=payload.role_code
    )


@router.patch("/{tenant_id}/members/{user_id}")
async def update_member(
    tenant_id: str,
    user_id: str,
    payload: TenantMemberUpdateIn,
    identity: CurrentIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    await _require_tenant_role(identity, tenant_id, {"tenant_admin"})
    db = get_prisma()
    return await tenant_service.update_member_roles(
        db,
        tenant_id=tenant_id,
        user_id=user_id,
        role_codes=payload.role_codes,
        actor_id=identity.user_id,
    )


@router.delete("/{tenant_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    tenant_id: str,
    user_id: str,
    identity: CurrentIdentity = Depends(get_current_user),
) -> None:
    await _require_tenant_role(identity, tenant_id, {"tenant_admin"})
    db = get_prisma()
    await tenant_service.remove_member(
        db, tenant_id=tenant_id, user_id=user_id, actor_id=identity.user_id
    )


# A thin namespace check used by Phase 6 frontend so it can prefetch the
# active tenant's display fields when the X-Tenant-Id header is set.
@router.get("/current", response_model=TenantOut)
async def current_tenant(
    identity: CurrentIdentity = Depends(require_tenant_scope()),
) -> TenantOut:
    db = get_prisma()
    tenant = await db.systenant.find_unique(where={"id": identity.tenant_id})
    if tenant is None:
        raise PermissionDenied("Tenant scope is invalid")
    return _tenant_to_out(tenant)
