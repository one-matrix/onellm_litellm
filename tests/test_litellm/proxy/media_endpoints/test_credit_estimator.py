"""Unit tests for the pure-function estimator in proxy.credit_service.

Covers all three billing methods used in production (按张 / 按秒 / 按次),
option_prices matching, markup, and the MIN_CHARGE_CREDITS floor.
"""

from typing import Any, Dict

import pytest

from litellm.proxy.credit_service.estimator import (
    MIN_CHARGE_CREDITS,
    estimate_media_credits,
)


def _video_info(**overrides: Any) -> Dict[str, Any]:
    """A realistic video model_info shape (matches docs/AI6700/litellm_config.yaml)."""
    info: Dict[str, Any] = {
        "ai6700_media_type": "video",
        "billing_method": "按秒",
        "base_price": 0.1518,
        "cost_per_second": 0.1518,
        "price_markup": 1.1,
        "option_prices": [
            {
                "param_name": "resolution",
                "option_value": "720p",
                "price_multiplier": 2.1625,
                "price_addition": 0,
            },
            {
                "param_name": "resolution",
                "option_value": "1080p",
                "price_multiplier": 4.8625,
                "price_addition": 0,
            },
            {
                "param_name": "generate_audio",
                "option_value": "false",
                "price_multiplier": 0.6,
                "price_addition": 0,
            },
        ],
    }
    info.update(overrides)
    return info


def _image_info(**overrides: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "ai6700_media_type": "image",
        "billing_method": "按张",
        "base_price": 0.5,
        "price_markup": 1.1,
        "option_prices": [],
    }
    info.update(overrides)
    return info


# ----- 按张 / image -----


def test_image_per_image_billing_default_count():
    result = estimate_media_credits(
        model_info=_image_info(),
        media_type="image",
        ai6700_params={},
    )
    # 0.5 base × 1.1 markup × 1 image = 0.55
    assert result.quantity == 1.0
    assert result.estimated_credits == pytest.approx(0.55)
    assert result.markup == 1.1


def test_image_per_image_billing_explicit_count():
    result = estimate_media_credits(
        model_info=_image_info(),
        media_type="image",
        ai6700_params={"count": 3},
        count=3,
    )
    # 0.5 × 1.1 × 3 = 1.65
    assert result.quantity == 3.0
    assert result.estimated_credits == pytest.approx(1.65)


def test_image_per_image_falls_back_to_n_in_params():
    result = estimate_media_credits(
        model_info=_image_info(),
        media_type="image",
        ai6700_params={"n": 2},
    )
    assert result.quantity == 2.0
    assert result.estimated_credits == pytest.approx(1.1)


# ----- 按秒 / video -----


def test_video_per_second_default_5s():
    result = estimate_media_credits(
        model_info=_video_info(),
        media_type="video",
        ai6700_params={},  # nothing → default 5s
    )
    # 0.1518 × 5 × 1.1 = 0.8349
    assert result.quantity == 5.0
    assert result.estimated_credits == pytest.approx(0.8349)


def test_video_per_second_explicit_seconds():
    result = estimate_media_credits(
        model_info=_video_info(),
        media_type="video",
        ai6700_params={"audio_duration": 10},
    )
    # 0.1518 × 10 × 1.1 = 1.6698
    assert result.quantity == 10.0
    assert result.estimated_credits == pytest.approx(1.6698)


def test_video_per_second_auto_freezes_15s():
    """``audio_duration='auto'`` should freeze the conservative 15s default."""
    result = estimate_media_credits(
        model_info=_video_info(),
        media_type="video",
        ai6700_params={"audio_duration": "auto"},
    )
    assert result.quantity == 15.0


def test_video_resolution_multiplier_applied():
    result = estimate_media_credits(
        model_info=_video_info(),
        media_type="video",
        ai6700_params={"audio_duration": 5, "resolution": "720p"},
    )
    # per_unit = 0.1518 × 2.1625 = 0.328267(5)
    # total    = 0.328267 × 5 × 1.1 = 1.80547
    assert result.multiplier == pytest.approx(2.1625)
    assert result.estimated_credits == pytest.approx(0.1518 * 2.1625 * 5 * 1.1)


def test_video_multiple_options_compose_multiplicatively():
    """Both resolution and generate_audio should compose."""
    result = estimate_media_credits(
        model_info=_video_info(),
        media_type="video",
        ai6700_params={
            "audio_duration": 5,
            "resolution": "1080p",
            "generate_audio": "false",  # x0.6
        },
    )
    expected = 0.1518 * 4.8625 * 0.6 * 5 * 1.1
    assert result.multiplier == pytest.approx(4.8625 * 0.6)
    assert result.estimated_credits == pytest.approx(expected)
    assert len(result.matched_options) == 2


def test_video_option_no_match_falls_back_to_base():
    result = estimate_media_credits(
        model_info=_video_info(),
        media_type="video",
        ai6700_params={"audio_duration": 5, "resolution": "4k"},  # not in option_prices
    )
    # No multiplier match → base only
    assert result.multiplier == 1.0
    assert result.estimated_credits == pytest.approx(0.1518 * 5 * 1.1)


# ----- markup + base resolution -----


def test_video_uses_cost_per_second_when_billing_method_per_second():
    """cost_per_second preferred over base_price for per-second billing."""
    result = estimate_media_credits(
        model_info=_video_info(base_price=999, cost_per_second=0.1518),
        media_type="video",
        ai6700_params={"audio_duration": 5},
    )
    assert result.base_price == 0.1518


def test_markup_default_one_when_missing():
    info = _image_info()
    del info["price_markup"]
    result = estimate_media_credits(
        model_info=info,
        media_type="image",
        ai6700_params={},
    )
    assert result.markup == 1.0


# ----- 按次 / per-call + edge cases -----


def test_per_call_billing_quantity_always_one():
    result = estimate_media_credits(
        model_info={
            "billing_method": "按次",
            "base_price": 1.0,
            "price_markup": 1.0,
        },
        media_type="audio",
        ai6700_params={"voice": "bv001"},
    )
    assert result.quantity == 1.0
    assert result.estimated_credits == pytest.approx(1.0)


def test_min_charge_floor_applied():
    """Tiny base × tiny quantity should still hit MIN_CHARGE_CREDITS."""
    result = estimate_media_credits(
        model_info={
            "billing_method": "按张",
            "base_price": 0.001,
            "price_markup": 1.0,
        },
        media_type="image",
        ai6700_params={"count": 1},
    )
    assert result.estimated_credits == MIN_CHARGE_CREDITS


def test_empty_model_info_defaults_safely():
    """Unknown model → no crash, returns MIN_CHARGE."""
    result = estimate_media_credits(
        model_info={},
        media_type="image",
        ai6700_params={},
    )
    assert result.estimated_credits == MIN_CHARGE_CREDITS
    assert result.base_price == 0.0


def test_billing_method_inferred_when_missing():
    """Empty billing_method → fall back to media-type-appropriate default."""
    result = estimate_media_credits(
        model_info={"base_price": 0.2, "price_markup": 1.0},
        media_type="video",
        ai6700_params={"audio_duration": 5},
    )
    assert result.billing_method == "按秒"
    assert result.quantity == 5.0


def test_boolean_option_value_normalized():
    """generate_audio supplied as a python bool should match the 'false' string in option_prices."""
    result = estimate_media_credits(
        model_info=_video_info(),
        media_type="video",
        ai6700_params={"audio_duration": 5, "generate_audio": False},
    )
    # x0.6 multiplier applied
    assert result.multiplier == pytest.approx(0.6)
