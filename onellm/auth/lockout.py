"""Account lockout: throttles brute-force password attempts.

The counter and the lockout window live in sys_users (access_failed_count
and lockout_end). We follow the ASP.NET Identity semantics: count failures,
flip to "locked" once the threshold is hit, clear both fields on a
successful login.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from onellm.config import SETTINGS


def is_locked(user: Any) -> bool:
    end = getattr(user, "lockout_end", None)
    if not end:
        return False
    end_aware = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
    return end_aware > datetime.now(timezone.utc)


async def record_failure(db: Any, user: Any) -> bool:
    """Increment failure counter; return True if account is now locked."""
    next_count = (user.access_failed_count or 0) + 1
    data: dict = {"access_failed_count": next_count}
    locked = False
    if next_count >= SETTINGS.lockout_max_failures and user.lockout_enabled:
        data["lockout_end"] = datetime.now(timezone.utc) + timedelta(
            seconds=SETTINGS.lockout_duration_seconds
        )
        data["access_failed_count"] = 0  # reset counter once locked
        locked = True
    await db.sysuser.update(where={"id": user.id}, data=data)
    return locked


async def record_success(db: Any, user_id: str) -> None:
    await db.sysuser.update(
        where={"id": user_id},
        data={"access_failed_count": 0, "lockout_end": None},
    )
