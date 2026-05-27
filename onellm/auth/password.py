"""Password hashing.

Reuses litellm's scrypt implementation so OneLLM users and LiteLLM users
share a hash format. That way the LiteLLM shadow row can mirror the same
password_hash without an additional re-encode step.
"""

import secrets

from litellm.proxy.utils import hash_password as _scrypt_hash
from litellm.proxy.utils import verify_password as _scrypt_verify


def hash_password(plaintext: str) -> str:
    return _scrypt_hash(plaintext)


def verify_password(plaintext: str, stored: str) -> bool:
    if not stored:
        return False
    return _scrypt_verify(plaintext, stored)


def new_security_stamp() -> str:
    """Rotated whenever the password changes — invalidates all access tokens."""
    return secrets.token_urlsafe(32)
