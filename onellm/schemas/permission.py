from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class PermissionNode(BaseModel):
    """Hierarchical permission node — children built recursively in service layer."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    code: str
    name: str
    type: str  # "group" | "action"
    sort_order: int
    children: List["PermissionNode"] = []


PermissionNode.model_rebuild()
