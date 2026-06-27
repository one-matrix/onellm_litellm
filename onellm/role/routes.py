"""Role + role-permission management routes.

Reads are open to any authenticated caller (the tenant admin UI needs the
role list to populate dropdowns). Mutations are root-only because changing
the permission set of a role affects every tenant globally.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends, status

from onellm.db import ONELLM_TX_OPTIONS, get_prisma
from onellm.deps import CurrentIdentity, get_current_user, require_global_role
from onellm.exceptions import NotFound
from onellm.schemas.role import RoleOut

router = APIRouter(prefix="/roles", tags=["onellm-role"])


async def _hydrate(db: Any, role: Any) -> RoleOut:
    rp_rows = await db.sysrolepermission.find_many(
        where={"role_id": role.id}, include={"permission": True}
    )
    codes = sorted({rp.permission.code for rp in rp_rows if rp.permission is not None})
    return RoleOut(
        id=role.id,
        code=role.code,
        name=role.name,
        description=role.description,
        permission_codes=codes,
    )


@router.get("", response_model=List[RoleOut])
async def list_roles(_: CurrentIdentity = Depends(get_current_user)) -> List[RoleOut]:
    db = get_prisma()
    roles = await db.sysrole.find_many(where={"is_deleted": False})
    return [await _hydrate(db, r) for r in roles]


@router.get("/{role_id}", response_model=RoleOut)
async def get_role(
    role_id: str, _: CurrentIdentity = Depends(get_current_user)
) -> RoleOut:
    db = get_prisma()
    role = await db.sysrole.find_unique(where={"id": role_id})
    if role is None or role.is_deleted:
        raise NotFound("Role not found")
    return await _hydrate(db, role)


@router.put("/{role_id}/permissions", response_model=RoleOut)
async def replace_role_permissions(
    role_id: str,
    permission_codes: List[str] = Body(..., embed=True),
    _: CurrentIdentity = Depends(require_global_role("root")),
) -> RoleOut:
    """Replace the role's permission set with the supplied codes (atomic)."""
    db = get_prisma()
    role = await db.sysrole.find_unique(where={"id": role_id})
    if role is None or role.is_deleted:
        raise NotFound("Role not found")

    permissions = await db.syspermission.find_many(
        where={"code": {"in": permission_codes}}
    )
    found_codes = {p.code for p in permissions}
    missing = set(permission_codes) - found_codes
    if missing:
        raise NotFound(f"Unknown permission codes: {sorted(missing)}")

    async with db.tx(**ONELLM_TX_OPTIONS) as tx:
        await tx.sysrolepermission.delete_many(where={"role_id": role_id})
        for p in permissions:
            await tx.sysrolepermission.create(
                data={"role_id": role_id, "permission_id": p.id}
            )

    return await _hydrate(db, role)
