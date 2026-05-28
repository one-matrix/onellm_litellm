"""Runtime configuration for the OneLLM control plane.

All values are read from environment variables at import time. Sensible
fall-backs exist so a developer can boot the proxy without setting any
ONELLM_* env var (in which case LITELLM_MASTER_KEY is reused as the JWT
secret — fine for local dev, must be separated before production).
"""

import os
from dataclasses import dataclass
from typing import Optional


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class OneLLMSettings:
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = 15 * 60
    refresh_token_ttl_seconds: int = 30 * 24 * 60 * 60

    lockout_max_failures: int = 5
    lockout_duration_seconds: int = 15 * 60

    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    google_redirect_uri: Optional[str] = None

    # Bootstrap: platform superadmin (role_global='root') auto-provisioned at startup.
    # Defaults are safe for local dev but MUST be overridden in production via
    # ONELLM_SUPERADMIN_* env vars — the first launch persists them into the DB.
    #
    # NOTE: do NOT use reserved TLDs like .local / .localhost / .test /
    # .example / .invalid. Pydantic's EmailStr (via the email-validator
    # package) refuses them per RFC 6761/6762, which would silently break
    # login with a 422 even though the row exists in the database.
    superadmin_email: str = "superadmin@onellm.io"
    superadmin_password: str = "OneLLM-Super-Admin-Change-Me!"
    superadmin_name: str = "Super Admin"
    superadmin_tenant_code: str = "system"
    superadmin_tenant_name: str = "System"


def load_settings() -> OneLLMSettings:
    secret = (
        _env("ONELLM_JWT_SECRET")
        or _env("LITELLM_MASTER_KEY")
        or "onellm-insecure-dev-secret"
    )
    if not secret:
        raise RuntimeError(
            "OneLLM: no JWT secret available. Set ONELLM_JWT_SECRET or LITELLM_MASTER_KEY."
        )
    return OneLLMSettings(
        jwt_secret=secret,
        access_token_ttl_seconds=int(_env("ONELLM_ACCESS_TTL_SECONDS", "900") or 900),
        refresh_token_ttl_seconds=int(
            _env("ONELLM_REFRESH_TTL_SECONDS", str(30 * 24 * 60 * 60))
            or 30 * 24 * 60 * 60
        ),
        lockout_max_failures=int(_env("ONELLM_LOCKOUT_MAX", "5") or 5),
        lockout_duration_seconds=int(_env("ONELLM_LOCKOUT_SECONDS", "900") or 900),
        google_client_id=_env("ONELLM_GOOGLE_CLIENT_ID"),
        google_client_secret=_env("ONELLM_GOOGLE_CLIENT_SECRET"),
        google_redirect_uri=_env("ONELLM_GOOGLE_REDIRECT_URI"),
        superadmin_email=_env("ONELLM_SUPERADMIN_EMAIL", "superadmin@onellm.io")
        or "superadmin@onellm.io",
        superadmin_password=_env(
            "ONELLM_SUPERADMIN_PASSWORD", "OneLLM-Super-Admin-Change-Me!"
        )
        or "OneLLM-Super-Admin-Change-Me!",
        superadmin_name=_env("ONELLM_SUPERADMIN_NAME", "Super Admin") or "Super Admin",
        superadmin_tenant_code=_env("ONELLM_SUPERADMIN_TENANT_CODE", "system")
        or "system",
        superadmin_tenant_name=_env("ONELLM_SUPERADMIN_TENANT_NAME", "System")
        or "System",
    )


SETTINGS = load_settings()
