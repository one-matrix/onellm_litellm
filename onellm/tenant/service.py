"""Tenant + member service: account creation, role updates, removals.

Membership rule: a user is a "member" of a tenant if any sys_user_roles row
exists with that (user_id, tenant_id). The same user may carry multiple
role codes inside one tenant (sys_user_roles is many-to-many).
"""

from datetime import datetime
from typing import Any, Iterable, List, Optional

from onellm.auth.password import hash_password, new_security_stamp
from onellm.db import ONELLM_TX_OPTIONS
from onellm.exceptions import (
    EmailAlreadyRegistered,
    NotFound,
    PermissionDenied,
    TenantCodeTaken,
)
from onellm.sync.litellm_shadow import (
    upsert_litellm_team_shadow,
    upsert_litellm_user_shadow,
)

PROTECTED_TENANT_ADMIN_ROLE = "tenant_admin"
ROOT_ROLE = "root"


def _normalize(text: Optional[str]) -> Optional[str]:
    return text.upper().strip() if text else None


def _role_code_set(role_codes: Iterable[str]) -> set[str]:
    return {code for code in role_codes if code}


def _ensure_member_role_codes_can_be_assigned(role_codes: Iterable[str]) -> None:
    codes = _role_code_set(role_codes)
    if ROOT_ROLE in codes:
        raise PermissionDenied("Cannot assign the root role through tenant membership")
    if PROTECTED_TENANT_ADMIN_ROLE in codes:
        raise PermissionDenied(
            "tenant_admin can only be assigned by registration or tenant creation"
        )


def _ensure_target_member_is_not_tenant_admin(role_codes: Iterable[str]) -> None:
    if PROTECTED_TENANT_ADMIN_ROLE in _role_code_set(role_codes):
        raise PermissionDenied(
            "tenant_admin accounts and roles cannot be changed or removed"
        )


async def _get_member_role_codes(db: Any, *, tenant_id: str, user_id: str) -> set[str]:
    memberships = await db.sysuserrole.find_many(
        where={"user_id": user_id, "tenant_id": tenant_id},
        include={"role": True},
    )
    return {m.role.code for m in memberships if m.role is not None}


async def list_my_memberships(db: Any, *, user_id: str) -> List[dict]:
    memberships = await db.sysuserrole.find_many(
        where={"user_id": user_id},
        include={"tenant": True, "role": True},
    )
    grouped: dict = {}
    for m in memberships:
        if m.tenant is None or m.tenant.is_deleted:
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


async def create_tenant(
    db: Any,
    *,
    name: str,
    code: str,
    plan_code: str,
    owner_user_id: str,
) -> Any:
    existing = await db.systenant.find_unique(where={"code": code})
    if existing is not None:
        raise TenantCodeTaken()

    async with db.tx(**ONELLM_TX_OPTIONS) as tx:
        tenant = await tx.systenant.create(
            data={"name": name, "code": code, "plan_code": plan_code or "free"}
        )
        admin_role = await tx.sysrole.find_unique(where={"code": "tenant_admin"})
        if admin_role is None:
            raise NotFound("tenant_admin role missing — re-run identity migration")
        await tx.sysuserrole.create(
            data={
                "user_id": owner_user_id,
                "role_id": admin_role.id,
                "tenant_id": tenant.id,
            }
        )
        team_id = await upsert_litellm_team_shadow(tx, tenant=tenant)
        if team_id != tenant.litellm_team_id:
            tenant = await tx.systenant.update(
                where={"id": tenant.id}, data={"litellm_team_id": team_id}
            )
    return tenant


async def list_members(db: Any, *, tenant_id: str) -> List[dict]:
    memberships = await db.sysuserrole.find_many(
        where={"tenant_id": tenant_id},
        include={"user": True, "role": True},
    )
    grouped: dict = {}
    for m in memberships:
        if m.user is None or m.user.is_deleted:
            continue
        bucket = grouped.setdefault(
            m.user.id,
            {
                "user": {
                    "id": m.user.id,
                    "email": m.user.email,
                    "name": m.user.name,
                    "role_global": m.user.role_global,
                },
                "roles": [],
                "joined_at": m.created_at,
            },
        )
        if m.role and m.role.code not in bucket["roles"]:
            bucket["roles"].append(m.role.code)
        # keep the earliest membership timestamp as the join date
        if m.created_at and m.created_at < bucket["joined_at"]:
            bucket["joined_at"] = m.created_at
    return list(grouped.values())


async def _find_or_create_user(
    db: Any,
    *,
    email: str,
    password: str,
    tenant_id: str,
    name: Optional[str],
    user_name: Optional[str],
) -> tuple[Any, Optional[str]]:
    """Look up a user by email; if missing, create a password-login account.

    Existing users are not password-reset here. The supplied password is only
    written when a placeholder account still has no password hash.
    """
    normalized = _normalize(email)
    if not normalized:
        raise EmailAlreadyRegistered("Invalid email")
    normalized_user_name = _normalize(user_name or email)
    password_hash = hash_password(password)
    existing = await db.sysuser.find_unique(where={"normalized_email": normalized})
    if existing is not None:
        data: dict = {}
        if name is not None and name != existing.name:
            data["name"] = name
        if user_name is not None and user_name != existing.user_name:
            data["user_name"] = user_name
            data["normalized_user_name"] = normalized_user_name
        if existing.tenant_id is None:
            data["tenant_id"] = tenant_id
        written_password_hash = None
        if not existing.password_hash:
            data["password_hash"] = password_hash
            data["security_stamp"] = new_security_stamp()
            written_password_hash = password_hash
        if data:
            existing = await db.sysuser.update(where={"id": existing.id}, data=data)
        return existing, written_password_hash
    user = await db.sysuser.create(
        data={
            "tenant_id": tenant_id,
            "name": name,
            "email": email,
            "normalized_email": normalized,
            "user_name": user_name or email,
            "normalized_user_name": normalized_user_name,
            "email_confirmed": False,
            "password_hash": password_hash,
            "security_stamp": new_security_stamp(),
            "role_global": "user",
            "is_active": True,
        }
    )
    return user, password_hash


async def create_member(
    db: Any,
    *,
    tenant_id: str,
    email: str,
    password: str,
    role_code: str,
    name: Optional[str] = None,
    user_name: Optional[str] = None,
) -> dict:
    tenant = await db.systenant.find_unique(where={"id": tenant_id})
    if tenant is None or tenant.is_deleted:
        raise NotFound("Tenant not found")
    role = await db.sysrole.find_unique(where={"code": role_code})
    if role is None:
        raise NotFound(f"Role {role_code!r} not found")
    _ensure_member_role_codes_can_be_assigned((role.code,))

    async with db.tx(**ONELLM_TX_OPTIONS) as tx:
        user, password_hash = await _find_or_create_user(
            tx,
            email=email,
            password=password,
            tenant_id=tenant_id,
            name=name,
            user_name=user_name,
        )
        await tx.sysuserrole.upsert(
            where={
                "user_id_role_id_tenant_id": {
                    "user_id": user.id,
                    "role_id": role.id,
                    "tenant_id": tenant_id,
                }
            },
            data={
                "create": {
                    "user_id": user.id,
                    "role_id": role.id,
                    "tenant_id": tenant_id,
                },
                "update": {},
            },
        )
        # Make sure the LiteLLM shadow row exists for this user so spend
        # tracking still has a row to attach against.
        await upsert_litellm_user_shadow(
            tx,
            user=user,
            tenant=tenant,
            password_hash=password_hash,
            tenant_role_codes=(role.code,),
        )
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role_global": user.role_global,
        },
        "roles": [role.code],
        "joined_at": datetime.utcnow(),
    }


async def update_member_roles(
    db: Any,
    *,
    tenant_id: str,
    user_id: str,
    role_codes: List[str],
    actor_id: str,
) -> dict:
    existing_role_codes = await _get_member_role_codes(
        db, tenant_id=tenant_id, user_id=user_id
    )
    _ensure_target_member_is_not_tenant_admin(existing_role_codes)

    if user_id == actor_id and PROTECTED_TENANT_ADMIN_ROLE not in role_codes:
        # Admins would otherwise be able to demote themselves out of the role
        # they need to manage members — surface as a clear error early.
        raise PermissionDenied("Cannot strip your own tenant_admin role")

    roles = await db.sysrole.find_many(where={"code": {"in": role_codes}})
    found_codes = {r.code for r in roles}
    missing = set(role_codes) - found_codes
    if missing:
        raise NotFound(f"Unknown role codes: {sorted(missing)}")
    _ensure_member_role_codes_can_be_assigned(found_codes)

    user = await db.sysuser.find_unique(where={"id": user_id})
    if user is None or user.is_deleted:
        raise NotFound("User not found")

    async with db.tx(**ONELLM_TX_OPTIONS) as tx:
        await tx.sysuserrole.delete_many(
            where={"user_id": user_id, "tenant_id": tenant_id}
        )
        for role in roles:
            await tx.sysuserrole.create(
                data={
                    "user_id": user_id,
                    "role_id": role.id,
                    "tenant_id": tenant_id,
                }
            )

    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role_global": user.role_global,
        },
        "roles": sorted(found_codes),
        "joined_at": user.created_at,
    }


async def remove_member(
    db: Any, *, tenant_id: str, user_id: str, actor_id: str
) -> None:
    existing_role_codes = await _get_member_role_codes(
        db, tenant_id=tenant_id, user_id=user_id
    )
    _ensure_target_member_is_not_tenant_admin(existing_role_codes)

    if user_id == actor_id:
        raise PermissionDenied("Cannot remove yourself; transfer ownership first.")
    deleted = await db.sysuserrole.delete_many(
        where={"user_id": user_id, "tenant_id": tenant_id}
    )
    if deleted == 0:
        raise NotFound("Membership not found")
