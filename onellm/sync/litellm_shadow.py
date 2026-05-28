"""Shadow-write OneLLM tenants/users into LiteLLM's native tables.

Why: LiteLLM proxy already has spend tracking, key management, and team
routing keyed off ``LiteLLM_TeamTable.team_id`` / ``LiteLLM_UserTable.user_id``.
To keep those subsystems working, we mirror the OneLLM ids into LiteLLM as
shadow rows. OneLLM remains authoritative; LiteLLM rows are derived state.

Sync direction is OneLLM -> LiteLLM only. Never write back from LiteLLM
into ``sys_*``; if a LiteLLM-only mutation slipped through, the next OneLLM
update will overwrite it on the next upsert.

Role projection
---------------
The OneLLM RBAC model (``role_global`` + tenant-scoped role codes + a
permission tree) is strictly richer than LiteLLM's flat ``LitellmUserRoles``
enum, so the shadow translates *down*. The mapping below is intentional and
matches the semantics LiteLLM applies in `proxy_server.py` for key/spend
routes — keep it in sync if LiteLLM's enum grows.

    OneLLM identity                          → LiteLLM user_role
    ─────────────────────────────────────────────────────────────
    role_global == 'root'                    → proxy_admin
    role_global == 'admin'                   → proxy_admin_viewer
    tenant role == 'tenant_admin'            → org_admin
    tenant role == 'user'                    → internal_user
    tenant role in {billing, viewer}         → internal_user_viewer
    (no tenant role / unknown)               → internal_user_viewer

If a user holds multiple tenant roles we pick the highest-privileged one
(precedence follows the table order).
"""

from typing import Any, Iterable, List, Optional, Tuple

from prisma import Json


# Global-scope short-circuit. role_global='user' falls through so the tenant
# role determines the projection.
_GLOBAL_ROLE_TO_LITELLM_ROLE = {
    "root": "proxy_admin",
    "admin": "proxy_admin_viewer",
}

# Ordered by precedence — first match wins.
_TENANT_ROLE_TO_LITELLM_ROLE: List[Tuple[str, str]] = [
    ("tenant_admin", "org_admin"),
    ("user", "internal_user"),
    ("billing", "internal_user_viewer"),
    ("viewer", "internal_user_viewer"),
]

_DEFAULT_LITELLM_ROLE = "internal_user_viewer"


def resolve_litellm_user_role(
    *, role_global: str, tenant_role_codes: Iterable[str] = ()
) -> str:
    """Project an OneLLM identity into the closest ``LitellmUserRoles`` value.

    See module docstring for the precedence table. Callers in transactions
    should prefer passing ``tenant_role_codes`` explicitly; otherwise
    ``upsert_litellm_user_shadow`` will resolve them via a single query.
    """
    mapped = _GLOBAL_ROLE_TO_LITELLM_ROLE.get(role_global)
    if mapped is not None:
        return mapped
    codes = set(tenant_role_codes or ())
    for code, litellm_role in _TENANT_ROLE_TO_LITELLM_ROLE:
        if code in codes:
            return litellm_role
    return _DEFAULT_LITELLM_ROLE


async def _fetch_tenant_role_codes(
    tx: Any, *, user_id: str, tenant_id: str
) -> set:
    """All sys_roles.code held by ``user_id`` inside ``tenant_id``."""
    rows = await tx.sysuserrole.find_many(
        where={"user_id": user_id, "tenant_id": tenant_id},
        include={"role": True},
    )
    codes: set = set()
    for row in rows:
        role = getattr(row, "role", None)
        if role is not None and role.code:
            codes.add(role.code)
    return codes


async def upsert_litellm_team_shadow(tx: Any, *, tenant: Any) -> str:
    """Create or update ``LiteLLM_TeamTable`` keyed by the OneLLM tenant id.

    Returns the team_id (== tenant.id) so the caller can persist it back
    onto ``sys_tenants.litellm_team_id``.
    """
    team_id = str(tenant.id)
    team_metadata = Json(
        {
            "onellm_tenant_id": team_id,
            "onellm_tenant_code": tenant.code,
            "plan_code": tenant.plan_code,
        }
    )
    await tx.litellm_teamtable.upsert(
        where={"team_id": team_id},
        data={
            "create": {
                "team_id": team_id,
                "team_alias": tenant.name,
                "metadata": team_metadata,
            },
            "update": {
                "team_alias": tenant.name,
                "metadata": team_metadata,
            },
        },
    )
    return team_id


async def upsert_litellm_user_shadow(
    tx: Any,
    *,
    user: Any,
    tenant: Optional[Any],
    password_hash: Optional[str] = None,
    tenant_role_codes: Optional[Iterable[str]] = None,
) -> str:
    """Create or update ``LiteLLM_UserTable`` keyed by the OneLLM user id.

    ``tenant_role_codes`` lets callers supply the role codes they just
    assigned (avoids a redundant query). When ``None`` and a tenant is
    given, we query ``sys_user_roles`` so the shadow always reflects the
    user's current standing in that tenant.
    """
    user_id = str(user.id)
    if tenant_role_codes is None and tenant is not None:
        tenant_role_codes = await _fetch_tenant_role_codes(
            tx, user_id=user_id, tenant_id=str(tenant.id)
        )
    litellm_role = resolve_litellm_user_role(
        role_global=user.role_global,
        tenant_role_codes=tenant_role_codes or (),
    )
    teams_list = [str(tenant.id)] if tenant is not None else []
    primary_team = str(tenant.id) if tenant is not None else None

    user_metadata = Json({"onellm_user_id": user_id})
    create_payload = {
        "user_id": user_id,
        "user_email": user.email,
        "user_alias": user.name or user.user_name,
        "user_role": litellm_role,
        "team_id": primary_team,
        "teams": teams_list,
        "password": password_hash,
        "metadata": user_metadata,
    }
    update_payload = {
        "user_email": user.email,
        "user_alias": user.name or user.user_name,
        "user_role": litellm_role,
        "team_id": primary_team,
        "teams": teams_list,
        "metadata": user_metadata,
    }
    if password_hash is not None:
        update_payload["password"] = password_hash

    await tx.litellm_usertable.upsert(
        where={"user_id": user_id},
        data={"create": create_payload, "update": update_payload},
    )
    return user_id


__all__ = [
    "resolve_litellm_user_role",
    "upsert_litellm_team_shadow",
    "upsert_litellm_user_shadow",
]
