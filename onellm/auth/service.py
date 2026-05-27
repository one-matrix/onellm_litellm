"""Identity service: registration, password login, token rotation, logout.

This module owns the cross-table consistency rules (user+tenant+role+token)
and keeps them inside a single Prisma transaction. Routes layer only does
request/response shape; everything below is here.
"""

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple

from onellm.auth.jwt import issue_access_token, new_refresh_token
from onellm.auth.lockout import is_locked, record_failure, record_success
from onellm.auth.password import hash_password, new_security_stamp, verify_password
from onellm.config import SETTINGS
from onellm.exceptions import (
    AccountInactive,
    AccountLocked,
    EmailAlreadyRegistered,
    InvalidCredentials,
    NotFound,
    TenantCodeTaken,
    TokenInvalid,
)
from onellm.schemas.token import TokenOut
from onellm.sync.litellm_shadow import (
    upsert_litellm_team_shadow,
    upsert_litellm_user_shadow,
)


REFRESH_PROVIDER = "OneLLM"
REFRESH_NAME = "refresh"

_TENANT_CODE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,49}$")
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _normalize(text: Optional[str]) -> Optional[str]:
    return text.upper().strip() if text else None


def _hash_refresh(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _slugify(name: str) -> str:
    slug = _SLUG_PATTERN.sub("-", name.lower()).strip("-")
    return slug[:50] or "tenant"


async def _ensure_role(db: Any, code: str) -> Any:
    role = await db.sysrole.find_unique(where={"code": code})
    if role is None:
        raise NotFound(f"Role {code!r} not found — did the seed migration run?")
    return role


async def resolve_unique_tenant_code(db: Any, desired: str) -> str:
    """Append numeric suffix until we find an unused tenant code."""
    code = desired
    suffix = 0
    while await db.systenant.find_unique(where={"code": code}) is not None:
        suffix += 1
        code = f"{desired}-{suffix}"[:50]
        if suffix > 50:
            raise TenantCodeTaken(
                "Could not generate a unique tenant code; please supply one."
            )
    return code


async def build_tokens(
    db: Any,
    user_id: str,
    tenant_id: Optional[str],
    role_global: str,
    security_stamp: Optional[str],
) -> TokenOut:
    access_token, expires_in = issue_access_token(
        user_id=user_id,
        tenant_id=tenant_id,
        role_global=role_global,
        security_stamp=security_stamp,
    )
    refresh = new_refresh_token()
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=SETTINGS.refresh_token_ttl_seconds
    )
    await db.sysusertoken.upsert(
        where={
            "user_id_login_provider_name": {
                "user_id": user_id,
                "login_provider": REFRESH_PROVIDER,
                "name": REFRESH_NAME,
            }
        },
        data={
            "create": {
                "user_id": user_id,
                "login_provider": REFRESH_PROVIDER,
                "name": REFRESH_NAME,
                "value": _hash_refresh(refresh),
                "expires_at": expires_at,
            },
            "update": {
                "value": _hash_refresh(refresh),
                "expires_at": expires_at,
            },
        },
    )
    return TokenOut(
        access_token=access_token,
        refresh_token=refresh,
        expires_in=expires_in,
    )


async def register_password_user(
    db: Any,
    *,
    email: str,
    password: str,
    name: Optional[str],
    tenant_name: Optional[str],
    tenant_code: Optional[str],
) -> Tuple[Any, Any, TokenOut]:
    """Create tenant + owner user atomically, then issue tokens.

    The new user is the owner of a brand-new tenant — this matches the
    self-service signup flow described in docs/onellm.md §5.2.
    """
    normalized_email = _normalize(email)
    if normalized_email is None:
        raise InvalidCredentials("Email is required")

    existing = await db.sysuser.find_unique(
        where={"normalized_email": normalized_email}
    )
    if existing is not None:
        raise EmailAlreadyRegistered()

    desired_code = tenant_code or _slugify(tenant_name or email.split("@")[0])
    if not _TENANT_CODE_PATTERN.match(desired_code):
        desired_code = _slugify(desired_code)

    async with db.tx() as tx:
        final_code = await resolve_unique_tenant_code(tx, desired_code)
        tenant = await tx.systenant.create(
            data={
                "name": tenant_name or (name or email.split("@")[0]),
                "code": final_code,
                "plan_code": "free",
            }
        )

        password_hash = hash_password(password)
        security_stamp = new_security_stamp()
        user = await tx.sysuser.create(
            data={
                "tenant_id": tenant.id,
                "name": name,
                "user_name": email,
                "normalized_user_name": _normalize(email),
                "email": email,
                "normalized_email": normalized_email,
                "email_confirmed": False,
                "password_hash": password_hash,
                "security_stamp": security_stamp,
                "role_global": "user",
            }
        )

        owner_role = await _ensure_role(tx, "tenant_owner")
        await tx.sysuserrole.create(
            data={
                "user_id": user.id,
                "role_id": owner_role.id,
                "tenant_id": tenant.id,
            }
        )

        litellm_team_id = await upsert_litellm_team_shadow(tx, tenant=tenant)
        if litellm_team_id != tenant.litellm_team_id:
            tenant = await tx.systenant.update(
                where={"id": tenant.id},
                data={"litellm_team_id": litellm_team_id},
            )

        litellm_user_id = await upsert_litellm_user_shadow(
            tx,
            user=user,
            tenant=tenant,
            password_hash=password_hash,
        )
        if litellm_user_id != user.litellm_user_id:
            user = await tx.sysuser.update(
                where={"id": user.id},
                data={"litellm_user_id": litellm_user_id},
            )

        tokens = await build_tokens(
            tx,
            user_id=user.id,
            tenant_id=tenant.id,
            role_global=user.role_global,
            security_stamp=user.security_stamp,
        )
    return user, tenant, tokens


async def login_password(db: Any, *, email: str, password: str) -> Tuple[Any, TokenOut]:
    normalized_email = _normalize(email)
    user = (
        None
        if normalized_email is None
        else await db.sysuser.find_unique(where={"normalized_email": normalized_email})
    )
    if user is None or user.is_deleted:
        raise InvalidCredentials()
    if not user.is_active:
        raise AccountInactive()
    if is_locked(user):
        raise AccountLocked()
    if not user.password_hash or not verify_password(password, user.password_hash):
        locked = await record_failure(db, user)
        if locked:
            raise AccountLocked()
        raise InvalidCredentials()

    await record_success(db, user.id)
    tokens = await build_tokens(
        db,
        user_id=user.id,
        tenant_id=user.tenant_id,
        role_global=user.role_global,
        security_stamp=user.security_stamp,
    )
    return user, tokens


async def refresh_session(db: Any, *, refresh_token: str) -> TokenOut:
    hashed = _hash_refresh(refresh_token)
    rows = await db.sysusertoken.find_many(
        where={
            "login_provider": REFRESH_PROVIDER,
            "name": REFRESH_NAME,
            "value": hashed,
        },
        take=1,
    )
    if not rows:
        raise TokenInvalid("Refresh token not recognized")
    record = rows[0]
    if record.expires_at is not None:
        expires = record.expires_at
        expires_aware = (
            expires if expires.tzinfo else expires.replace(tzinfo=timezone.utc)
        )
        if expires_aware <= datetime.now(timezone.utc):
            raise TokenInvalid("Refresh token expired")
    user = await db.sysuser.find_unique(where={"id": record.user_id})
    if user is None or user.is_deleted or not user.is_active:
        raise TokenInvalid("User is inactive")
    return await build_tokens(
        db,
        user_id=user.id,
        tenant_id=user.tenant_id,
        role_global=user.role_global,
        security_stamp=user.security_stamp,
    )


async def logout(db: Any, *, user_id: str, refresh_token: Optional[str]) -> None:
    """Best-effort revoke; absent token is not an error."""
    if not refresh_token:
        await db.sysusertoken.delete_many(
            where={
                "user_id": user_id,
                "login_provider": REFRESH_PROVIDER,
                "name": REFRESH_NAME,
            }
        )
        return
    await db.sysusertoken.delete_many(
        where={
            "user_id": user_id,
            "login_provider": REFRESH_PROVIDER,
            "name": REFRESH_NAME,
            "value": _hash_refresh(refresh_token),
        }
    )


async def change_password(
    db: Any, *, user_id: str, old_password: str, new_password: str
) -> None:
    user = await db.sysuser.find_unique(where={"id": user_id})
    if user is None:
        raise NotFound("User not found")
    if not user.password_hash or not verify_password(old_password, user.password_hash):
        raise InvalidCredentials("Current password is incorrect")
    new_hash = hash_password(new_password)
    await db.sysuser.update(
        where={"id": user_id},
        data={
            "password_hash": new_hash,
            "security_stamp": new_security_stamp(),
        },
    )
    # Wipe all refresh tokens — forces re-login on every other device.
    await db.sysusertoken.delete_many(
        where={
            "user_id": user_id,
            "login_provider": REFRESH_PROVIDER,
            "name": REFRESH_NAME,
        }
    )


async def list_user_permission_codes(
    db: Any, *, user_id: str, tenant_id: Optional[str]
) -> List[str]:
    """Aggregate permission codes the user has within a tenant scope.

    Falls back to the empty set when tenant_id is missing — the caller can
    still rely on role_global checks (root/admin/user) for global routes.
    """
    if tenant_id is None:
        return []
    memberships = await db.sysuserrole.find_many(
        where={"user_id": user_id, "tenant_id": tenant_id},
        include={
            "role": {"include": {"role_permissions": {"include": {"permission": True}}}}
        },
    )
    codes: set = set()
    for membership in memberships:
        role = getattr(membership, "role", None)
        if role is None:
            continue
        for rp in role.role_permissions or []:
            permission = getattr(rp, "permission", None)
            if permission is not None and permission.code:
                codes.add(permission.code)
    return sorted(codes)
