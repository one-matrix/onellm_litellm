"""Payment channel registry — channel-code → adapter instance.

A real channel adapter implements ``create_payment / verify_notify / parse_notify /
query_order / close_order / refund``. The stub channel ships in this repo and
auto-marks orders as paid so end-to-end recharge flows work without an SDK.
"""

from __future__ import annotations

from typing import Dict, List

from .channels.types import PayChannel
from .channels.stub import StubChannel

_REGISTRY: Dict[str, PayChannel] = {
    "stub": StubChannel(),
}

# Channels visible to the user in the recharge UI. Real adapters (alipay,
# wechat, ...) should add themselves here once their SDKs are wired up.
_SUPPORTED: List[str] = ["stub"]


def get_channel(code: str) -> PayChannel:
    if code not in _REGISTRY:
        raise ValueError(f"Unsupported payment channel: {code}")
    return _REGISTRY[code]


def is_channel_supported(code: str) -> bool:
    return code in _SUPPORTED


def supported_channels() -> List[str]:
    return list(_SUPPORTED)
