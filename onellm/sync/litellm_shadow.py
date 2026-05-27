"""Shadow-write OneLLM tenants/users into LiteLLM's native tables.

Why: LiteLLM proxy already has spend tracking, key management, and team
routing keyed off ``LiteLLM_TeamTable.team_id`` / ``LiteLLM_UserTable.user_id``.
To keep those subsystems working, we mirror the OneLLM ids into LiteLLM as
shadow rows. OneLLM remains authoritative; LiteLLM rows are derived state.

Sync direction is OneLLM -> LiteLLM only. Never write back from LiteLLM
into ``sys_*``; if a LiteLLM-only mutation slipped through, the next OneLLM
update will overwrite it on the next upsert.
"""

from typing import Any, Optional

_GLOBAL_ROLE_TO_LITELLM_ROLE = {
    "root": "proxy_admin",
    "admin": "proxy_admin",
    "user": "internal_user",
}


async def upsert_litellm_team_shadow(tx: Any, *, tenant: Any) -> str:
    """Create or update ``LiteLLM_TeamTable`` keyed by the OneLLM tenant id.

    Returns the team_id (== tenant.id) so the caller can persist it back
    onto ``sys_tenants.litellm_team_id``.
    """
    team_id = str(tenant.id)
    await tx.litellm_teamtable.upsert(
        where={"team_id": team_id},
        data={
            "create": {
                "team_id": team_id,
                "team_alias": tenant.name,
                "metadata": {
                    "onellm_tenant_id": team_id,
                    "onellm_tenant_code": tenant.code,
                    "plan_code": tenant.plan_code,
                },
            },
            "update": {
                "team_alias": tenant.name,
                "metadata": {
                    "onellm_tenant_id": team_id,
                    "onellm_tenant_code": tenant.code,
                    "plan_code": tenant.plan_code,
                },
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
) -> str:
    """Create or update ``LiteLLM_UserTable`` keyed by the OneLLM user id."""
    user_id = str(user.id)
    litellm_role = _GLOBAL_ROLE_TO_LITELLM_ROLE.get(user.role_global, "internal_user")
    teams_list = [str(tenant.id)] if tenant is not None else []
    primary_team = str(tenant.id) if tenant is not None else None

    create_payload = {
        "user_id": user_id,
        "user_email": user.email,
        "user_alias": user.name or user.user_name,
        "user_role": litellm_role,
        "team_id": primary_team,
        "teams": teams_list,
        "password": password_hash,
        "metadata": {"onellm_user_id": user_id},
    }
    update_payload = {
        "user_email": user.email,
        "user_alias": user.name or user.user_name,
        "user_role": litellm_role,
        "team_id": primary_team,
        "teams": teams_list,
        "metadata": {"onellm_user_id": user_id},
    }
    if password_hash is not None:
        update_payload["password"] = password_hash

    await tx.litellm_usertable.upsert(
        where={"user_id": user_id},
        data={"create": create_payload, "update": update_payload},
    )
    return user_id
