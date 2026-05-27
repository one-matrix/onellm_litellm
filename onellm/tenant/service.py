"""Tenant + member service: invites, role updates, removals.

Membership rule: a user is a "member" of a tenant if any sys_user_roles row
exists with that (user_id, tenant_id). The same user may carry multiple
role codes inside one tenant (sys_user_roles is many-to-many).
"""

from datetime import datetime
from typing import Any, List, Optional

from onellm.auth.password import new_security_stamp
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


def _normalize(text: Optional[str]) -> Optional[str]:
    return text.upper().strip() if text else None


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

    async with db.tx() as tx:
        tenant = await tx.systenant.create(
            data={"name": name, "code": code, "plan_code": plan_code or "free"}
        )
        owner_role = await tx.sysrole.find_unique(where={"code": "tenant_owner"})
        if owner_role is None:
            raise NotFound("tenant_owner role missing — re-run identity migration")
        await tx.sysuserrole.create(
            data={
                "user_id": owner_user_id,
                "role_id": owner_role.id,
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


async def _find_or_invite_user(db: Any, *, email: str) -> Any:
    """Look up a user by email; if missing, create a shadow account.

    The invited user has no password yet — they activate via OAuth or by
    using the "forgot password" flow once that exists. For now we mark the
    account inactive until the user finishes setup (MVP: any login attempt
    on this account fails until the user resets password).
    """
    normalized = _normalize(email)
    if not normalized:
        raise EmailAlreadyRegistered("Invalid email")
    existing = await db.sysuser.find_unique(where={"normalized_email": normalized})
    if existing is not None:
        return existing
    return await db.sysuser.create(
        data={
            "email": email,
            "normalized_email": normalized,
            "user_name": email,
            "normalized_user_name": normalized,
            "security_stamp": new_security_stamp(),
            "role_global": "user",
            "is_active": True,  # they can be invited even if they have no pwd
        }
    )


async def invite_member(db: Any, *, tenant_id: str, email: str, role_code: str) -> dict:
    tenant = await db.systenant.find_unique(where={"id": tenant_id})
    if tenant is None or tenant.is_deleted:
        raise NotFound("Tenant not found")
    role = await db.sysrole.find_unique(where={"code": role_code})
    if role is None:
        raise NotFound(f"Role {role_code!r} not found")
    if role.code == "root":
        raise PermissionDenied("Cannot assign the root role through tenant invitation")

    async with db.tx() as tx:
        user = await _find_or_invite_user(tx, email=email)
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
        await upsert_litellm_user_shadow(tx, user=user, tenant=tenant)
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
    if user_id == actor_id and "tenant_owner" not in role_codes:
        # Owners would otherwise be able to demote themselves out of the role
        # they need to manage members — surface as a clear error early.
        raise PermissionDenied("Cannot strip your own tenant_owner role")

    roles = await db.sysrole.find_many(where={"code": {"in": role_codes}})
    found_codes = {r.code for r in roles}
    missing = set(role_codes) - found_codes
    if missing:
        raise NotFound(f"Unknown role codes: {sorted(missing)}")
    if "root" in found_codes:
        raise PermissionDenied("root role cannot be granted inside a tenant scope")

    user = await db.sysuser.find_unique(where={"id": user_id})
    if user is None or user.is_deleted:
        raise NotFound("User not found")

    async with db.tx() as tx:
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
    if user_id == actor_id:
        raise PermissionDenied("Cannot remove yourself; transfer ownership first.")
    deleted = await db.sysuserrole.delete_many(
        where={"user_id": user_id, "tenant_id": tenant_id}
    )
    if deleted == 0:
        raise NotFound("Membership not found")
