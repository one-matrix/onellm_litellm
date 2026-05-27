"""Permission tree query route.

Returns the full permission hierarchy as nested ``PermissionNode`` objects.
The tree is small (seven groups, ~20 leaves) so a single fetch + in-memory
build is fine.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from onellm.db import get_prisma
from onellm.deps import CurrentIdentity, get_current_user
from onellm.schemas.permission import PermissionNode

router = APIRouter(prefix="/permissions", tags=["onellm-permission"])


def _build_tree(rows: List[Any]) -> List[PermissionNode]:
    nodes: Dict[str, PermissionNode] = {
        r.id: PermissionNode(
            id=r.id,
            code=r.code,
            name=r.name,
            type=r.type,
            sort_order=r.sort_order,
            children=[],
        )
        for r in rows
    }
    roots: List[PermissionNode] = []
    for r in rows:
        node = nodes[r.id]
        if r.parent_id and r.parent_id in nodes:
            nodes[r.parent_id].children.append(node)
        else:
            roots.append(node)

    def sort_recursive(items: List[PermissionNode]) -> None:
        items.sort(key=lambda n: (n.sort_order, n.code))
        for it in items:
            sort_recursive(it.children)

    sort_recursive(roots)
    return roots


@router.get("", response_model=List[PermissionNode])
async def get_permission_tree(
    _: CurrentIdentity = Depends(get_current_user),
) -> List[PermissionNode]:
    db = get_prisma()
    rows = await db.syspermission.find_many(where={"is_deleted": False})
    return _build_tree(rows)
