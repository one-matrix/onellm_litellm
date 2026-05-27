"""HTTP routes for register / login / refresh / logout / me / change-password."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, status

from onellm.auth import service as auth_service
from onellm.db import get_prisma
from onellm.deps import CurrentIdentity, get_current_user
from onellm.schemas.token import LoginIn, RefreshIn, RegisterIn, TokenOut
from onellm.schemas.user import ChangePasswordIn, UserOut

router = APIRouter(prefix="/auth", tags=["onellm-auth"])


def _user_to_out(user: Any) -> UserOut:
    return UserOut(
        id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        user_name=user.user_name,
        name=user.name,
        role_global=user.role_global,
        email_confirmed=user.email_confirmed,
        is_active=user.is_active,
        two_factor_enabled=user.two_factor_enabled,
        created_at=user.created_at,
    )


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=TokenOut)
async def register(payload: RegisterIn) -> TokenOut:
    db = get_prisma()
    _, _, tokens = await auth_service.register_password_user(
        db,
        email=payload.email,
        password=payload.password,
        name=payload.name,
        tenant_name=payload.tenant_name,
        tenant_code=payload.tenant_code,
    )
    return tokens


@router.post("/login", response_model=TokenOut)
async def login(payload: LoginIn) -> TokenOut:
    db = get_prisma()
    _, tokens = await auth_service.login_password(
        db, email=payload.email, password=payload.password
    )
    return tokens


@router.post("/refresh", response_model=TokenOut)
async def refresh(payload: RefreshIn) -> TokenOut:
    db = get_prisma()
    return await auth_service.refresh_session(db, refresh_token=payload.refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    identity: CurrentIdentity = Depends(get_current_user),
    payload: Optional[RefreshIn] = Body(default=None),
) -> None:
    db = get_prisma()
    await auth_service.logout(
        db,
        user_id=identity.user_id,
        refresh_token=payload.refresh_token if payload else None,
    )


@router.get("/me")
async def me(identity: CurrentIdentity = Depends(get_current_user)) -> Dict[str, Any]:
    db = get_prisma()
    tenant_payload: Optional[Dict[str, Any]] = None
    if identity.tenant_id:
        tenant = await db.systenant.find_unique(where={"id": identity.tenant_id})
        if tenant is not None:
            tenant_payload = {
                "id": tenant.id,
                "name": tenant.name,
                "code": tenant.code,
                "plan_code": tenant.plan_code,
                "litellm_team_id": tenant.litellm_team_id,
            }
    return {
        "user": _user_to_out(identity.user).model_dump(),
        "tenant": tenant_payload,
        "permissions": sorted(identity.permission_codes),
    }


@router.post("/me/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_my_password(
    payload: ChangePasswordIn,
    identity: CurrentIdentity = Depends(get_current_user),
) -> None:
    db = get_prisma()
    await auth_service.change_password(
        db,
        user_id=identity.user_id,
        old_password=payload.old_password,
        new_password=payload.new_password,
    )


@router.get("/tenants/mine")
async def list_my_tenants(
    identity: CurrentIdentity = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Tenants the caller has a role in.

    Lives under /auth (not /tenants) so a fresh access token can fetch the
    list before the caller has chosen an active tenant.
    """
    db = get_prisma()
    memberships = await db.sysuserrole.find_many(
        where={"user_id": identity.user_id},
        include={"tenant": True, "role": True},
    )
    grouped: Dict[str, Dict[str, Any]] = {}
    for m in memberships:
        if m.tenant is None:
            continue
        bucket = grouped.setdefault(
            m.tenant.id,
            {
                "id": m.tenant.id,
                "name": m.tenant.name,
                "code": m.tenant.code,
                "plan_code": m.tenant.plan_code,
                "roles": [],
            },
        )
        if m.role and m.role.code not in bucket["roles"]:
            bucket["roles"].append(m.role.code)
    return list(grouped.values())
