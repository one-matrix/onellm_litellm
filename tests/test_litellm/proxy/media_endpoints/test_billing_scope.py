from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from onellm import billing_scope
from onellm.billing_scope import resolve_credit_tenant_id


class _Row:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._data)


class _Table:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    async def find_unique(self, where: Dict[str, Any]):
        return self._find(where)

    async def find_first(self, where: Dict[str, Any]):
        return self._find(where)

    def _find(self, where: Dict[str, Any]) -> Optional[_Row]:
        for row in self.rows:
            if all(row.get(k) == v for k, v in where.items()):
                return _Row(row)
        return None


class _PrismaClient:
    def __init__(self) -> None:
        self.db = SimpleNamespace(
            systenant=_Table(
                [
                    {
                        "id": "tenant-1",
                        "litellm_team_id": "team-tenant-1",
                        "is_active": True,
                        "is_deleted": False,
                    },
                    {
                        "id": "tenant-2",
                        "litellm_team_id": "team-tenant-2",
                        "is_active": True,
                        "is_deleted": False,
                    },
                ]
            ),
            sysuser=_Table(
                [
                    {
                        "id": "user-1",
                        "is_active": True,
                        "is_deleted": False,
                        "security_stamp": "stamp-1",
                        "role_global": "user",
                    }
                ]
            ),
            sysuserrole=_Table(
                [{"user_id": "user-1", "tenant_id": "tenant-1"}]
            ),
        )


def _request(headers: Dict[str, str]):
    return SimpleNamespace(headers=headers)


@pytest.mark.asyncio
async def test_resolve_credit_tenant_uses_active_onellm_tenant(monkeypatch):
    monkeypatch.setattr(
        billing_scope,
        "decode_access_token",
        lambda token: {"sub": "user-1", "stamp": "stamp-1"},
    )

    resolved = await resolve_credit_tenant_id(
        prisma_client=_PrismaClient(),
        request=_request(
            {
                "X-Tenant-Id": "tenant-1",
                "X-OneLLM-Access-Token": "jwt",
                "X-OneLLM-Auth-Provider": "onellm",
            }
        ),
        user_api_key_dict=UserAPIKeyAuth(
            user_id="fallback-user",
            team_id="fallback-team",
            user_role=LitellmUserRoles.INTERNAL_USER,
        ),
    )

    assert resolved == "team-tenant-1"


@pytest.mark.asyncio
async def test_resolve_credit_tenant_rejects_non_member(monkeypatch):
    monkeypatch.setattr(
        billing_scope,
        "decode_access_token",
        lambda token: {"sub": "user-1", "stamp": "stamp-1"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await resolve_credit_tenant_id(
            prisma_client=_PrismaClient(),
            request=_request(
                {
                    "X-Tenant-Id": "tenant-2",
                    "X-OneLLM-Access-Token": "jwt",
                    "X-OneLLM-Auth-Provider": "onellm",
                }
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="fallback-user",
                team_id="fallback-team",
                user_role=LitellmUserRoles.INTERNAL_USER,
            ),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_resolve_credit_tenant_keeps_admin_override():
    resolved = await resolve_credit_tenant_id(
        prisma_client=_PrismaClient(),
        request=_request({}),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        override="tenant-from-admin-filter",
    )

    assert resolved == "tenant-from-admin-filter"
