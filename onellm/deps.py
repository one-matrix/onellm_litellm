"""FastAPI dependencies for the OneLLM control plane.

Three layers of guard:
- ``get_current_user``     -> any authenticated caller
- ``require_global_role``  -> root / admin (cross-tenant ops)
- ``require_tenant_role``  -> within-tenant role check
- ``require_permission``   -> permission code check (e.g. "key.create")
"""

from typing import Any, Iterable, Optional, Set

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from onellm.auth.jwt import decode_access_token
from onellm.auth.service import list_user_permission_codes
from onellm.db import get_prisma
from onellm.exceptions import PermissionDenied, TokenInvalid

_bearer_scheme = HTTPBearer(auto_error=False)


class CurrentIdentity:
    """Resolved caller context — user row + tenant scope + permission codes."""

    __slots__ = ("user", "tenant_id", "permission_codes")

    def __init__(
        self,
        user: Any,
        tenant_id: Optional[str],
        permission_codes: Iterable[str] = (),
    ) -> None:
        self.user = user
        self.tenant_id = tenant_id
        self.permission_codes = set(permission_codes)

    @property
    def role_global(self) -> str:
        return self.user.role_global

    @property
    def user_id(self) -> str:
        return self.user.id


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-Id"),
) -> CurrentIdentity:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise TokenInvalid("Missing or malformed Authorization header")
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise TokenInvalid("Token missing subject")

    db = get_prisma()
    user = await db.sysuser.find_unique(where={"id": user_id})
    if user is None or user.is_deleted or not user.is_active:
        raise TokenInvalid("User is inactive or removed")
    # Stamp rotation invalidates every outstanding access token.
    if payload.get("stamp") and payload["stamp"] != user.security_stamp:
        raise TokenInvalid("Token superseded by password change")

    tenant_id = x_tenant_id or payload.get("tid") or user.tenant_id
    request.state.onellm_user = user
    request.state.onellm_tenant_id = tenant_id

    permission_codes: Set[str] = set()
    if tenant_id:
        permission_codes = set(
            await list_user_permission_codes(db, user_id=user.id, tenant_id=tenant_id)
        )
    return CurrentIdentity(
        user=user, tenant_id=tenant_id, permission_codes=permission_codes
    )


def require_global_role(*roles: str):
    allowed = set(roles)

    async def _dep(
        identity: CurrentIdentity = Depends(get_current_user),
    ) -> CurrentIdentity:
        if identity.role_global not in allowed and "root" not in allowed | {
            identity.role_global
        }:
            # root is always allowed unless we want a stricter list
            if identity.role_global != "root":
                raise PermissionDenied(
                    f"Requires one of global roles: {sorted(allowed)}"
                )
        return identity

    return _dep


def require_permission(*codes: str):
    """Allow if the caller holds ANY of the listed permission codes (or is global root)."""
    needed = set(codes)

    async def _dep(
        identity: CurrentIdentity = Depends(get_current_user),
    ) -> CurrentIdentity:
        if identity.role_global == "root":
            return identity
        if not (identity.permission_codes & needed):
            raise PermissionDenied(f"Missing permission: any of {sorted(needed)}")
        return identity

    return _dep


def require_tenant_scope():
    """Caller must have a resolved tenant_id (either from token or X-Tenant-Id)."""

    async def _dep(
        identity: CurrentIdentity = Depends(get_current_user),
    ) -> CurrentIdentity:
        if not identity.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-Tenant-Id header is required for this endpoint",
            )
        return identity

    return _dep
