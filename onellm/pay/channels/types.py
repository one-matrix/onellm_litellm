"""Payment channel adapter protocol.

Each channel (alipay, wechat, stripe, paypal, manual, ...) implements this
protocol. Endpoints stay channel-agnostic by always going through
``registry.get_channel(code)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol


PaymentType = Literal["form", "url", "qrcode", "manual"]


@dataclass
class CreatePaymentInput:
    out_trade_no: str
    amount_cny: float
    subject: str
    notify_url: Optional[str] = None
    return_url: Optional[str] = None
    timeout_minutes: int = 15
    client_ip: Optional[str] = None


@dataclass
class CreatePaymentResult:
    type: PaymentType
    payload: str
    """For ``form``: full <form> HTML; for ``url``: a URL string;
    for ``qrcode``: the data to encode; for ``manual``: a human note."""


@dataclass
class NotifyParseResult:
    out_trade_no: str
    channel_trade_no: Optional[str]
    trade_status: str
    paid_amount_cny: float
    pay_account: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class PayChannel(Protocol):
    code: str

    def create_payment(self, payload: CreatePaymentInput) -> CreatePaymentResult: ...

    def verify_notify(self, params: Dict[str, Any]) -> bool: ...

    def parse_notify(self, params: Dict[str, Any]) -> NotifyParseResult: ...

    def query_order(self, out_trade_no: str) -> Optional[NotifyParseResult]: ...

    def close_order(self, out_trade_no: str) -> None: ...

    def refund(
        self,
        *,
        out_trade_no: str,
        out_refund_no: str,
        amount_cny: float,
        reason: str,
    ) -> Dict[str, Any]: ...
