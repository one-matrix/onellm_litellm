"""AI6700 cost calculator.

The *primary* cost-writeback path is via ``_hidden_params["additional_headers"]
["llm_provider-x-litellm-response-cost"]`` (set by ``AI6700Helper.build_hidden_params``),
which makes ``litellm.cost_calculator.response_cost_calculator`` short-circuit
on the value our submit+poll loop already retrieved from
``/v1/skills/task-status``.

This module provides a *secondary* path for callers who:
  - pass an AI6700 response object to ``litellm.completion_cost(...)``
    after the fact (no logging chain involved), or
  - bypass the standard hidden_params header convention entirely.

In both cases we look at ``_hidden_params.response_cost`` (already × markup),
then fall back to ``raw_cost × markup`` from the model_info, then 0.0.

Note: ai6700's task-status returns the *real* charged amount, so any value
computed locally from ``model_info.base_price`` is at best an estimate.
Prefer the value sitting on the response object — it came from the wire.
"""

from __future__ import annotations

from typing import Any, Optional

import litellm

__all__ = ["cost_calculator"]


def _hidden_params(response: Any) -> dict:
    hp = getattr(response, "_hidden_params", None)
    if isinstance(hp, dict):
        return hp
    # Pydantic v2 stores via __dict__ in some places
    if hasattr(response, "__dict__"):
        d = response.__dict__.get("_hidden_params")
        if isinstance(d, dict):
            return d
    return {}


def _from_additional_headers(hp: dict) -> Optional[float]:
    headers = hp.get("additional_headers") or {}
    val = headers.get("llm_provider-x-litellm-response-cost")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _from_response_attrs(response: Any) -> Optional[float]:
    """For our own response types (MediaResponse, AI6700AudioResponse)."""
    for attr in ("cost", "response_cost"):
        val = getattr(response, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def cost_calculator(model: str, response: Any) -> float:
    """Post-hoc cost lookup for an AI6700 response.

    Looks in this order:
      1. ``_hidden_params["response_cost"]`` (canonical post-call source)
      2. ``_hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"]``
         (the convention used by ``response_cost_calculator``)
      3. attributes on the response object itself (``cost``, ``response_cost``)
      4. ``0.0`` (no cost info available — caller should treat as unknown)
    """
    hp = _hidden_params(response)

    cost = hp.get("response_cost")
    if cost is not None:
        try:
            return float(cost)
        except (TypeError, ValueError):
            pass

    cost = _from_additional_headers(hp)
    if cost is not None:
        return cost

    cost = _from_response_attrs(response)
    if cost is not None:
        return cost

    # Best-effort: try model_info × raw_cost from hidden_params.
    raw = hp.get("ai6700_raw_cost")
    if raw is not None:
        markup = hp.get("ai6700_price_markup")
        if markup is None:
            try:
                info = litellm.get_model_info(model=model, custom_llm_provider="ai6700")
                markup = info.get("price_markup")
            except Exception:
                markup = None
        try:
            return float(raw) * float(markup or 1.0)
        except (TypeError, ValueError):
            pass

    return 0.0
