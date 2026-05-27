from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserBrief(BaseModel):
    """Minimal user shape — used in membership lists and audit trails."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    email: Optional[EmailStr]
    name: Optional[str]
    role_global: str


class UserOut(BaseModel):
    """Full user profile returned by /auth/me and admin endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: Optional[str]
    email: Optional[EmailStr]
    user_name: Optional[str]
    name: Optional[str]
    role_global: str
    email_confirmed: bool
    is_active: bool
    two_factor_enabled: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None


class UserUpdateSelfIn(BaseModel):
    name: Optional[str] = Field(default=None, max_length=100)
    user_name: Optional[str] = Field(default=None, max_length=256)


class ChangePasswordIn(BaseModel):
    old_password: str
    new_password: str = Field(min_length=8, max_length=128)
