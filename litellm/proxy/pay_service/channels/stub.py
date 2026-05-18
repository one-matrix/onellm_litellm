"""Stub payment channel.

Simulates a payment provider for development. ``create_payment`` returns a
self-submitting HTML form that POSTs back to a callback URL we own, so the
existing PaymentResultModal polling loop works end-to-end without an SDK.

For now we simplify even further: the create_payment result is just a "fake
gateway" HTML page that, when opened, shows a button the user can click to
mark the order as paid manually. This makes local development clear without
auto-completing payments (which would defeat the purpose of testing the
polling UI).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .types import (
    CreatePaymentInput,
    CreatePaymentResult,
    NotifyParseResult,
    PayChannel,
)


class StubChannel:
    code: str = "stub"

    def create_payment(self, payload: CreatePaymentInput) -> CreatePaymentResult:
        # A minimal HTML page that lets the developer manually finalize the
        # order (POSTs to our own callback). Real channels return a redirect
        # form, here we keep the same shape (`type=form`, payload=HTML).
        html = f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>Simulated Gateway</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; padding: 48px; max-width: 640px; margin: 0 auto; color: #333; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 16px; padding: 32px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); }}
  h1 {{ font-size: 20px; margin: 0 0 8px; }}
  .amt {{ font-size: 36px; font-weight: 800; color: #ff4d4f; margin: 16px 0; }}
  button {{ background: #1677ff; color: #fff; border: 0; padding: 12px 24px; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer; }}
  button.cancel {{ background: #f0f0f0; color: #333; margin-left: 12px; }}
  .meta {{ color: #999; font-size: 13px; margin-top: 24px; }}
</style>
</head>
<body>
  <div class="card">
    <h1>模拟支付网关 (stub)</h1>
    <div>订单 <code>{payload.out_trade_no}</code></div>
    <div>{payload.subject}</div>
    <div class="amt">¥{payload.amount_cny:.2f}</div>
    <button onclick="pay('paid')">模拟支付成功</button>
    <button class="cancel" onclick="pay('closed')">取消支付</button>
    <div class="meta">本页面仅在开发环境使用。生产环境请配置真实支付渠道。</div>
  </div>
<script>
async function pay(status) {{
  const url = (location.origin.replace(/:\\d+$/, ':4000')) + '/pay/callback/stub';
  try {{
    await fetch(url, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ out_trade_no: '{payload.out_trade_no}', status }}),
    }});
  }} catch (e) {{}}
  document.body.innerHTML = '<div style="text-align:center;padding:64px;font-size:18px;color:#52c41a">已提交，您可以关闭此页面</div>';
  setTimeout(function() {{ try {{ window.close(); }} catch (e) {{}} }}, 800);
}}
</script>
</body></html>
""".strip()
        return CreatePaymentResult(type="form", payload=html)

    def verify_notify(self, params: Dict[str, Any]) -> bool:
        # Stub: signature always valid.
        return True

    def parse_notify(self, params: Dict[str, Any]) -> NotifyParseResult:
        return NotifyParseResult(
            out_trade_no=str(params.get("out_trade_no") or ""),
            channel_trade_no=str(params.get("channel_trade_no") or "") or None,
            trade_status=str(params.get("status") or "paid"),
            paid_amount_cny=float(params.get("paid_amount_cny") or 0.0),
            pay_account=params.get("pay_account"),
            raw=dict(params),
        )

    def query_order(self, out_trade_no: str) -> Optional[NotifyParseResult]:
        # Stub channel has no external state — refresh is a no-op.
        return None

    def close_order(self, out_trade_no: str) -> None:
        return None

    def refund(
        self,
        *,
        out_trade_no: str,
        out_refund_no: str,
        amount_cny: float,
        reason: str,
    ) -> Dict[str, Any]:
        # Stub: refunds always succeed instantly.
        return {
            "out_trade_no": out_trade_no,
            "out_refund_no": out_refund_no,
            "amount_cny": amount_cny,
            "fund_change": "Y",
            "channel_refund_no": f"stub-rf-{out_refund_no}",
        }
