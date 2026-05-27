"""JWT helpers for OneLLM access tokens.

Refresh tokens are opaque random strings persisted in sys_user_tokens (see
auth/service.py); JWTs are only used for short-lived access tokens, with the
security_stamp embedded so a password change instantly revokes them all.
"""

import secrets
import time
from typing import Any, Dict, Optional, Tuple

import jwt

from onellm.config import SETTINGS
from onellm.exceptions import TokenInvalid


def issue_access_token(
    user_id: str,
    tenant_id: Optional[str],
    role_global: str,
    security_stamp: Optional[str],
) -> Tuple[str, int]:
    """Return (access_token, expires_in_seconds)."""
    now = int(time.time())
    exp = now + SETTINGS.access_token_ttl_seconds
    payload: Dict[str, Any] = {
        "sub": user_id,
        "tid": tenant_id,
        "role": role_global,
        "stamp": security_stamp,
        "iat": now,
        "exp": exp,
        "jti": secrets.token_urlsafe(16),
        "typ": "access",
    }
    token = jwt.encode(payload, SETTINGS.jwt_secret, algorithm=SETTINGS.jwt_algorithm)
    return token, SETTINGS.access_token_ttl_seconds


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token, SETTINGS.jwt_secret, algorithms=[SETTINGS.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenInvalid("Access token expired") from exc
    except jwt.PyJWTError as exc:
        raise TokenInvalid(f"Access token invalid: {exc}") from exc
    if payload.get("typ") != "access":
        raise TokenInvalid("Wrong token type")
    return payload


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)
