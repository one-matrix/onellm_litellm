from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class RoleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    code: str
    name: Optional[str]
    description: Optional[str]
    permission_codes: List[str] = []
