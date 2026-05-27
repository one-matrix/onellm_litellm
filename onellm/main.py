"""Top-level router aggregating all OneLLM control-plane endpoints.

Mounted from ``litellm/proxy/proxy_server.py`` with::

    from onellm.main import router as onellm_router
    app.include_router(onellm_router)

The aggregator router declares ``prefix="/onellm"``; each sub-router adds
its own segment (e.g. ``/auth``, ``/tenants``) so final paths read
naturally when grep'd.
"""

from fastapi import APIRouter

from onellm.auth.oauth_google import router as oauth_google_router
from onellm.auth.routes import router as auth_router
from onellm.permission.routes import router as permission_router
from onellm.role.routes import router as role_router
from onellm.tenant.routes import router as tenant_router
from onellm.user.routes import router as user_router

router = APIRouter(prefix="/onellm")
router.include_router(auth_router)
router.include_router(oauth_google_router)
router.include_router(tenant_router)
router.include_router(user_router)
router.include_router(role_router)
router.include_router(permission_router)

__all__ = ["router"]
