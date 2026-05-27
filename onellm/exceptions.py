"""Domain-specific exceptions for the OneLLM control plane.

Mapped to HTTP status codes in a single FastAPI exception handler at
onellm.main, so service code can raise these without knowing about HTTP.
"""

from fastapi import HTTPException, status


class OneLLMError(HTTPException):
    code: str = "onellm_error"


class InvalidCredentials(OneLLMError):
    code = "invalid_credentials"

    def __init__(self, detail: str = "Invalid email or password") -> None:
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class AccountLocked(OneLLMError):
    code = "account_locked"

    def __init__(self, detail: str = "Account is temporarily locked") -> None:
        super().__init__(status_code=status.HTTP_423_LOCKED, detail=detail)


class AccountInactive(OneLLMError):
    code = "account_inactive"

    def __init__(self, detail: str = "Account is inactive") -> None:
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class EmailAlreadyRegistered(OneLLMError):
    code = "email_taken"

    def __init__(self, detail: str = "Email is already registered") -> None:
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)


class TenantCodeTaken(OneLLMError):
    code = "tenant_code_taken"

    def __init__(self, detail: str = "Tenant code is already in use") -> None:
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)


class NotFound(OneLLMError):
    code = "not_found"

    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class PermissionDenied(OneLLMError):
    code = "permission_denied"

    def __init__(self, detail: str = "Permission denied") -> None:
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class TokenInvalid(OneLLMError):
    code = "token_invalid"

    def __init__(self, detail: str = "Token is invalid or expired") -> None:
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)
