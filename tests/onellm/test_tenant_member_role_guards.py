import pytest

from onellm.exceptions import PermissionDenied
from onellm.tenant import service


def test_member_role_assignment_rejects_tenant_admin() -> None:
    with pytest.raises(PermissionDenied, match="tenant_admin"):
        service._ensure_member_role_codes_can_be_assigned(["tenant_admin"])


def test_member_role_assignment_rejects_root() -> None:
    with pytest.raises(PermissionDenied, match="root"):
        service._ensure_member_role_codes_can_be_assigned(["root"])


def test_member_role_assignment_allows_regular_roles() -> None:
    service._ensure_member_role_codes_can_be_assigned(["user", "billing", "viewer"])


def test_tenant_admin_target_member_is_protected() -> None:
    with pytest.raises(PermissionDenied, match="tenant_admin"):
        service._ensure_target_member_is_not_tenant_admin(["tenant_admin", "billing"])


def test_regular_target_member_is_not_protected() -> None:
    service._ensure_target_member_is_not_tenant_admin(["user", "billing"])
