"""Database access shim for OneLLM.

Re-exports the global Prisma client that proxy_server.py initializes at
startup. We do NOT spin up a second client — the litellm proxy already owns
the connection pool, and reusing it keeps transactions consistent across
OneLLM + LiteLLM tables (e.g. when shadow-syncing sys_users -> LiteLLM_UserTable).
"""

from datetime import timedelta
from typing import Any

from onellm.exceptions import OneLLMError


ONELLM_TX_OPTIONS = {
    "max_wait": timedelta(seconds=10),
    "timeout": timedelta(seconds=60),
}


def get_prisma() -> Any:
    """Return the live PrismaClient, or raise if the proxy hasn't started."""
    from litellm.proxy import proxy_server  # local import to avoid cycle at boot

    client = getattr(proxy_server, "prisma_client", None)
    if client is None:
        raise OneLLMError(
            status_code=503,
            detail="Database is not initialized yet. The proxy is still booting.",
        )
    return client.db
