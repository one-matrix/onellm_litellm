"""Credit wallet + pricing for async media generation.

Two public surfaces:

  - ``estimator.estimate_media_credits`` — compute the estimated credit cost
    for a media task from ``model_info`` (base_price / option_prices /
    price_markup) and the post-mapping AI6700 params dict.

  - ``wallet.ensure_wallet`` / ``pre_deduct`` / ``settle`` / ``refund`` —
    Prisma-backed three-step billing keyed by ``tenant_id`` (= team_id).

See [docs/onellm_extend.md](../../../../docs/onellm_extend.md) for the wider
async-media architecture and [docs/sql/credits.sql](../../../../docs/sql/credits.sql)
for the SQL shape these tables descend from.
"""

from litellm.proxy.credit_service.estimator import (
    MIN_CHARGE_CREDITS,
    EstimateResult,
    estimate_media_credits,
)
from litellm.proxy.credit_service.wallet import (
    WalletError,
    ensure_wallet,
    pre_deduct,
    rebind_agent_record_id,
    refund,
    settle,
)

__all__ = [
    "MIN_CHARGE_CREDITS",
    "EstimateResult",
    "estimate_media_credits",
    "WalletError",
    "ensure_wallet",
    "pre_deduct",
    "rebind_agent_record_id",
    "settle",
    "refund",
]
