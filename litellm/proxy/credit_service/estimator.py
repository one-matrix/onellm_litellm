"""Estimate the credit cost of a media-generation task from ``model_info``.

Pricing inputs all live in ``deployment["model_info"]`` (loaded from
``model_list`` / config.yaml). Layout:

  base_price            float   per-unit cost before options/markup
  cost_per_second       float?  per-second price for video (preferred over
                                base_price when billing_method == "按秒")
  price_markup          float   our retail markup (default 1.0)
  billing_method        str     "按次" / "按张" / "按秒" / "按token" / "per_call"...
  option_prices         list    [{param_name, option_value,
                                  price_multiplier, price_addition, ...}]

Formula::

    per_unit = base × ∏ option.price_multiplier (where param matches)
                  + Σ option.price_addition       (where param matches)
    estimated = max(per_unit × quantity × markup, MIN_CHARGE_CREDITS)

``MIN_CHARGE_CREDITS = 0.1`` mirrors the OpenNotebook reference implementation
([docs/temp/credit_service.py](../../../../docs/temp/credit_service.py)).

This module is pure (no DB / no IO) and safe to call before submitting the
upstream task — the estimate becomes the frozen amount on the wallet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

MIN_CHARGE_CREDITS = 0.1

# Billing method aliases — accept both CN labels from AI6700 and the canonical
# OpenNotebook strings, so the same estimator works for either source.
_PER_CALL = {"按次", "per_call"}
_PER_IMAGE = {"按张", "per_image"}
_PER_SECOND = {"按秒", "per_second"}
_PER_MINUTE = {"按分钟", "per_minute"}

_DEFAULT_AUTO_SECONDS = 15.0  # conservative pre-freeze when duration='auto'
_DEFAULT_SECONDS = 5.0


@dataclass
class EstimateResult:
    estimated_credits: float
    base_price: float
    quantity: float
    multiplier: float
    addition: float
    markup: float
    billing_method: str
    matched_options: List[Dict[str, Any]] = field(default_factory=list)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _to_positive_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
        return v if v > 0 else None
    return None


def _duration_for(params: Dict[str, Any]) -> float:
    """Resolve seconds for video tasks.

    AI6700 native keys: ``audio_duration``, ``duration``, ``seconds``. The
    string ``"auto"`` means "model picks length" — we conservatively freeze
    ``_DEFAULT_AUTO_SECONDS`` and settle to actual on completion.
    """
    for key in ("seconds", "audio_duration", "duration", "durationSeconds"):
        raw = params.get(key)
        if raw is None:
            continue
        if isinstance(raw, str) and raw.strip().lower() == "auto":
            return _DEFAULT_AUTO_SECONDS
        positive = _to_positive_float(raw)
        if positive is not None:
            return positive
    return _DEFAULT_SECONDS


def _image_count(params: Dict[str, Any], top_level_count: Optional[int]) -> float:
    if top_level_count is not None and top_level_count > 0:
        return float(top_level_count)
    for key in ("count", "n", "num_images", "imageCount"):
        raw = params.get(key)
        positive = _to_positive_float(raw)
        if positive is not None:
            return positive
    return 1.0


def _iter_option_prices(model_info: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    entries = model_info.get("option_prices")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def _resolve_option_adjustments(
    option_prices: Iterable[Dict[str, Any]],
    params: Dict[str, Any],
) -> tuple[float, float, List[Dict[str, Any]]]:
    """Match request params against option_prices entries.

    For each entry whose ``param_name`` is present in params and whose
    ``option_value`` matches the supplied value (string-compared), apply
    its ``price_multiplier`` (multiplicative) and ``price_addition``
    (additive). Returns ``(multiplier_product, addition_sum, matched_entries)``.
    """
    multiplier = 1.0
    addition = 0.0
    matched: List[Dict[str, Any]] = []
    for entry in option_prices:
        param_name = entry.get("param_name")
        if not param_name or param_name not in params:
            continue
        option_value = entry.get("option_value")
        if option_value is None:
            continue
        if _to_str(params[param_name]) != _to_str(option_value):
            continue
        mult = _to_positive_float(entry.get("price_multiplier"))
        if mult is not None:
            multiplier *= mult
        try:
            addition += float(entry.get("price_addition") or 0.0)
        except (TypeError, ValueError):
            pass
        matched.append(entry)
    return multiplier, addition, matched


def _resolve_base_price(model_info: Dict[str, Any], billing_method: str) -> float:
    """Pick the right per-unit base from model_info.

    For per-second billing, ``cost_per_second`` is preferred when present;
    falls back to ``base_price`` so configs that only set base_price still
    estimate (just less precisely).
    """
    if billing_method in _PER_SECOND:
        cps = _to_positive_float(model_info.get("cost_per_second"))
        if cps is not None:
            return cps
    base = _to_positive_float(model_info.get("base_price"))
    if base is not None:
        return base
    # Some configs use cost_per_request as the synonym for base_price.
    cpr = _to_positive_float(model_info.get("cost_per_request"))
    if cpr is not None:
        return cpr
    return 0.0


def _resolve_billing_method(model_info: Dict[str, Any], media_type: str) -> str:
    """Best-effort billing method — fall back to a media_type sensible default."""
    bm = model_info.get("billing_method")
    if isinstance(bm, str) and bm.strip():
        return bm.strip()
    if media_type == "video":
        return "按秒"
    if media_type == "image":
        return "按张"
    return "按次"


def _quantity_for(
    billing_method: str,
    media_type: str,
    ai6700_params: Dict[str, Any],
    top_level_count: Optional[int],
) -> float:
    if billing_method in _PER_CALL:
        return 1.0
    if billing_method in _PER_IMAGE:
        return _image_count(ai6700_params, top_level_count)
    if billing_method in _PER_SECOND:
        return _duration_for(ai6700_params)
    if billing_method in _PER_MINUTE:
        seconds = _duration_for(ai6700_params)
        return seconds / 60.0
    # Unknown billing method — best-effort by media_type
    if media_type == "image":
        return _image_count(ai6700_params, top_level_count)
    if media_type == "video":
        return _duration_for(ai6700_params)
    return 1.0


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def estimate_media_credits(
    *,
    model_info: Dict[str, Any],
    media_type: str,
    ai6700_params: Optional[Dict[str, Any]] = None,
    count: Optional[int] = None,
) -> EstimateResult:
    """Compute the pre-freeze credit estimate for a media task.

    Args:
        model_info: ``deployment["model_info"]`` from the router. Reads
            ``base_price`` / ``cost_per_second`` / ``billing_method`` /
            ``option_prices`` / ``price_markup``.
        media_type: ``image`` / ``video`` / ``audio`` / ``tts`` / ``music``.
        ai6700_params: the AI6700-native param dict (post mapping from
            OpenAI-style fields). Pass the request body's ``params`` here.
        count: top-level batch count (image / video). Falls back to
            ``params.count`` / ``params.n`` / ``params.num_images``.
    """
    info = dict(model_info or {})
    params = dict(ai6700_params or {})

    billing_method = _resolve_billing_method(info, media_type)
    base_price = _resolve_base_price(info, billing_method)
    multiplier, addition, matched = _resolve_option_adjustments(
        _iter_option_prices(info), params
    )
    quantity = _quantity_for(billing_method, media_type, params, count)
    markup = _to_positive_float(info.get("price_markup")) or 1.0

    per_unit = base_price * multiplier + addition
    estimated = per_unit * quantity * markup
    estimated = max(estimated, MIN_CHARGE_CREDITS)

    return EstimateResult(
        estimated_credits=estimated,
        base_price=base_price,
        quantity=quantity,
        multiplier=multiplier,
        addition=addition,
        markup=markup,
        billing_method=billing_method,
        matched_options=matched,
    )
