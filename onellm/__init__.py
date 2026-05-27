"""OneLLM control plane: user, tenant, RBAC, OAuth.

Lives outside backend/litellm/ so upstream merges never collide with the
OneLLM-specific identity layer. All tables live in the `app` schema; see
litellm-proxy-extras/.../migrations/20260527084700_add_onellm_identity for
the underlying SQL.
"""

__all__ = ["main"]
