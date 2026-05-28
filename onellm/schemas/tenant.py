from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from onellm.schemas.user import UserBrief


class TenantBrief(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    code: str
    plan_code: str


class TenantOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    code: str
    domain: Optional[str]
    plan_code: str
    default_price_markup: float
    is_active: bool
    litellm_team_id: Optional[str]
    created_at: datetime


class TenantCreateIn(BaseModel):
    name: str = Field(max_length=100)
    code: str = Field(max_length=50, pattern=r"^[a-z0-9][a-z0-9_-]{2,49}$")
    plan_code: Optional[str] = Field(default="free", max_length=50)


class TenantMemberIn(BaseModel):
    email: EmailStr
    role_code: str = Field(description="e.g. tenant_admin, user, billing, viewer")


class TenantMemberUpdateIn(BaseModel):
    role_codes: List[str] = Field(min_length=1)


class TenantMemberOut(BaseModel):
    user: UserBrief
    roles: List[str]
    joined_at: datetime
