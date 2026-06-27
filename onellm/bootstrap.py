"""Idempotent provisioning of the platform superadmin (role_global='root').

Invoked from ``litellm/proxy/proxy_server.py`` after ``prisma_client`` is
initialized. Safe to call repeatedly: if the configured email already maps to
a user, we leave them untouched.

The superadmin lives inside a dedicated "system" tenant so the standard
RBAC machinery still applies — they hold the seeded ``root`` role, which the
identity migration binds to every action permission.
"""

from typing import Any, Optional

from litellm._logging import verbose_proxy_logger

from onellm.auth.password import hash_password, new_security_stamp
from onellm.config import SETTINGS
from onellm.db import ONELLM_TX_OPTIONS, get_prisma
from onellm.exceptions import OneLLMError
from onellm.sync.litellm_shadow import (
    upsert_litellm_team_shadow,
    upsert_litellm_user_shadow,
)


SUPERADMIN_ROLE_CODE = "root"


def _normalized(email: str) -> str:
    return email.upper().strip()


async def _ensure_system_tenant(tx: Any) -> Any:
    tenant = await tx.systenant.find_unique(
        where={"code": SETTINGS.superadmin_tenant_code}
    )
    if tenant is not None:
        return tenant
    return await tx.systenant.create(
        data={
            "name": SETTINGS.superadmin_tenant_name,
            "code": SETTINGS.superadmin_tenant_code,
            "plan_code": "free",
        }
    )


async def _ensure_role_assignment(
    tx: Any, *, user_id: str, role_id: str, tenant_id: str
) -> None:
    existing = await tx.sysuserrole.find_unique(
        where={
            "user_id_role_id_tenant_id": {
                "user_id": user_id,
                "role_id": role_id,
                "tenant_id": tenant_id,
            }
        }
    )
    if existing is None:
        await tx.sysuserrole.create(
            data={"user_id": user_id, "role_id": role_id, "tenant_id": tenant_id}
        )


async def ensure_superadmin() -> Optional[str]:
    """Create or verify the platform superadmin user.

    Returns the user_id when a user exists (created or pre-existing), or None
    if bootstrap was skipped (e.g. prisma not ready). Never raises — failures
    log a warning so they cannot block proxy startup.
    """
    try:
        db = get_prisma()
    except OneLLMError as exc:
        verbose_proxy_logger.warning(
            "OneLLM superadmin bootstrap skipped: %s", exc.detail
        )
        return None

    email = SETTINGS.superadmin_email
    normalized_email = _normalized(email)

    try:
        existing = await db.sysuser.find_unique(
            where={"normalized_email": normalized_email}
        )
        root_role = await db.sysrole.find_unique(
            where={"code": SUPERADMIN_ROLE_CODE}
        )
        if root_role is None:
            verbose_proxy_logger.error(
                "OneLLM superadmin bootstrap: '%s' role missing — identity "
                "migration has not been applied. Skipping.",
                SUPERADMIN_ROLE_CODE,
            )
            return None

        if existing is not None:
            # Idempotent path: keep the user but make sure the role binding is
            # still in place inside the system tenant. Do NOT overwrite their
            # password or rotate their security stamp — operators may have
            # rotated credentials manually.
            tenant = await db.systenant.find_unique(
                where={"code": SETTINGS.superadmin_tenant_code}
            )
            if tenant is None:
                async with db.tx(**ONELLM_TX_OPTIONS) as tx:
                    tenant = await _ensure_system_tenant(tx)
            await _ensure_role_assignment(
                db,
                user_id=existing.id,
                role_id=root_role.id,
                tenant_id=tenant.id,
            )
            verbose_proxy_logger.debug(
                "OneLLM superadmin already provisioned (user_id=%s)", existing.id
            )
            return existing.id

        password_hash = hash_password(SETTINGS.superadmin_password)
        security_stamp = new_security_stamp()

        async with db.tx(**ONELLM_TX_OPTIONS) as tx:
            tenant = await _ensure_system_tenant(tx)
            user = await tx.sysuser.create(
                data={
                    "tenant_id": tenant.id,
                    "name": SETTINGS.superadmin_name,
                    "user_name": email,
                    "normalized_user_name": normalized_email,
                    "email": email,
                    "normalized_email": normalized_email,
                    "email_confirmed": True,
                    "password_hash": password_hash,
                    "security_stamp": security_stamp,
                    "role_global": "root",
                }
            )
            await _ensure_role_assignment(
                tx,
                user_id=user.id,
                role_id=root_role.id,
                tenant_id=tenant.id,
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
                tenant_role_codes=(SUPERADMIN_ROLE_CODE,),
            )
            if litellm_user_id != user.litellm_user_id:
                user = await tx.sysuser.update(
                    where={"id": user.id},
                    data={"litellm_user_id": litellm_user_id},
                )

        verbose_proxy_logger.info(
            "OneLLM superadmin provisioned: email=%s tenant=%s user_id=%s. "
            "Rotate ONELLM_SUPERADMIN_PASSWORD before production deployment.",
            email,
            SETTINGS.superadmin_tenant_code,
            user.id,
        )
        return user.id
    except Exception as exc:
        verbose_proxy_logger.warning(
            "OneLLM superadmin bootstrap failed: %s", exc, exc_info=True
        )
        return None


__all__ = ["ensure_superadmin"]
