"""User profile + admin user-listing routes.

Self-service endpoints (me / change-password) live in onellm.auth.routes;
this module is for global root listing of all users + per-user admin ops.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, status

from onellm.db import get_prisma
from onellm.deps import CurrentIdentity, get_current_user, require_global_role
from onellm.exceptions import NotFound
from onellm.schemas.user import UserOut, UserUpdateSelfIn

router = APIRouter(prefix="/users", tags=["onellm-user"])


def _user_to_out(u: Any) -> UserOut:
    return UserOut(
        id=u.id,
        tenant_id=u.tenant_id,
        email=u.email,
        user_name=u.user_name,
        name=u.name,
        role_global=u.role_global,
        email_confirmed=u.email_confirmed,
        is_active=u.is_active,
        two_factor_enabled=u.two_factor_enabled,
        created_at=u.created_at,
    )


@router.get("", response_model=List[UserOut])
async def list_users(
    q: Optional[str] = Query(default=None, max_length=128),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: Optional[str] = Query(default=None),
    _: CurrentIdentity = Depends(require_global_role("root", "admin")),
) -> List[UserOut]:
    """Paginated global user list — root/admin only."""
    db = get_prisma()
    where: Dict[str, Any] = {"is_deleted": False}
    if q:
        where["OR"] = [
            {"email": {"contains": q, "mode": "insensitive"}},
            {"user_name": {"contains": q, "mode": "insensitive"}},
            {"name": {"contains": q, "mode": "insensitive"}},
        ]
    kwargs: Dict[str, Any] = {"where": where, "take": limit, "order": {"id": "asc"}}
    if cursor:
        kwargs["cursor"] = {"id": cursor}
        kwargs["skip"] = 1
    rows = await db.sysuser.find_many(**kwargs)
    return [_user_to_out(u) for u in rows]


@router.patch("/me", response_model=UserOut)
async def update_self(
    payload: UserUpdateSelfIn,
    identity: CurrentIdentity = Depends(get_current_user),
) -> UserOut:
    db = get_prisma()
    data: Dict[str, Any] = {}
    if payload.name is not None:
        data["name"] = payload.name
    if payload.user_name is not None:
        data["user_name"] = payload.user_name
        data["normalized_user_name"] = payload.user_name.upper()
    if not data:
        return _user_to_out(identity.user)
    updated = await db.sysuser.update(where={"id": identity.user_id}, data=data)
    return _user_to_out(updated)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def soft_delete_user(
    user_id: str,
    _: CurrentIdentity = Depends(require_global_role("root")),
) -> None:
    db = get_prisma()
    existing = await db.sysuser.find_unique(where={"id": user_id})
    if existing is None or existing.is_deleted:
        raise NotFound("User not found")
    await db.sysuser.update(
        where={"id": user_id},
        data={"is_deleted": True, "is_active": False},
    )
