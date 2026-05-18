"""Onellm payment subsystem.

Wraps the generic ``app.pay_order`` / ``pay_refund`` / ``pay_notify_log`` tables
with channel adapters (alipay / wechat / stripe / ...) and product settlers
(credits / subscription / ...).

Currently ships a single ``stub`` channel that simulates payment success
immediately — useful for development and for the credit recharge flow until a
real channel SDK is wired in. The architecture (PayChannel protocol, registry,
per-product settler) mirrors ``frontend/features/pay`` so future migration to
real alipay/wechat SDKs is a drop-in.
"""

from .registry import get_channel, is_channel_supported, supported_channels  # noqa: F401
