"""Tests for AI6700 post-hoc cost_calculator + litellm-standard short-circuit."""

import pytest

from litellm.cost_calculator import (
    get_response_cost_from_hidden_params,
    response_cost_calculator,
)
from litellm.llms.ai6700 import AI6700Helper, cost_calculator
from litellm.llms.ai6700.image_generation import AI6700ImageConfig
from litellm.types.media import MediaAsset, MediaResponse


@pytest.fixture
def terminal_status() -> dict:
    return {
        "task_id": 1,
        "is_final": True,
        "status_group": "已完成",
        "result_url": "https://cdn/x.png",
        "result_type": "image",
        "cost": 1.5,
        "channel_group": "default",
        "created_at": "2026-03-17T10:00:00Z",
    }


@pytest.fixture
def hidden_params_with_cost(terminal_status) -> dict:
    return AI6700Helper.build_hidden_params(
        status=terminal_status, bare_model="m", price_markup=1.1
    )


class TestLiteLLMShortCircuit:
    """The critical path: response_cost_calculator picks up our cost via header."""

    def test_get_response_cost_from_hidden_params(self, hidden_params_with_cost):
        val = get_response_cost_from_hidden_params(hidden_params_with_cost)
        assert val == pytest.approx(1.65)

    def test_response_cost_calculator_short_circuits(self, terminal_status):
        """End-to-end: standard litellm.response_cost_calculator uses our cost."""
        cfg = AI6700ImageConfig()
        resp = cfg.transform_task_to_image_response(
            status=terminal_status, model="ai6700/m", price_markup=1.1
        )
        got = response_cost_calculator(
            response_object=resp,
            model="ai6700/m",
            custom_llm_provider="ai6700",
            call_type="aimage_generation",
            optional_params={},
        )
        assert got == pytest.approx(1.65)

    def test_no_recompute_via_completion_cost(self, terminal_status):
        """Verify short-circuit means we never reach completion_cost
        (which would try to look up litellm.model_cost and fail).
        """
        cfg = AI6700ImageConfig()
        resp = cfg.transform_task_to_image_response(
            status=terminal_status,
            model="ai6700/totally-unregistered",
            price_markup=2.0,
        )
        # Even with a bogus model, short-circuit returns the header value.
        got = response_cost_calculator(
            response_object=resp,
            model="ai6700/totally-unregistered",
            custom_llm_provider="ai6700",
            call_type="aimage_generation",
            optional_params={},
        )
        assert got == pytest.approx(3.0)


class TestPostHocCostCalculator:
    """litellm/llms/ai6700/cost_calculator.py fallback chain."""

    def test_reads_response_cost_from_hidden_params(self, terminal_status):
        cfg = AI6700ImageConfig()
        resp = cfg.transform_task_to_image_response(
            status=terminal_status, model="ai6700/m", price_markup=1.1
        )
        cost = cost_calculator(model="ai6700/m", response=resp)
        assert cost == pytest.approx(1.65)

    def test_reads_additional_headers_when_response_cost_missing(self):
        class FakeResp:
            _hidden_params = {
                "additional_headers": {
                    "llm_provider-x-litellm-response-cost": 0.42,
                }
            }

        assert cost_calculator(model="ai6700/m", response=FakeResp()) == 0.42

    def test_reads_cost_attribute_when_hidden_params_missing(self):
        class FakeResp:
            cost = 0.99
            _hidden_params: dict = {}

        assert cost_calculator(model="ai6700/m", response=FakeResp()) == 0.99

    def test_reads_response_cost_attribute(self):
        class FakeResp:
            response_cost = 0.55
            _hidden_params: dict = {}

        assert cost_calculator(model="ai6700/m", response=FakeResp()) == 0.55

    def test_zero_for_unknown_response(self):
        class Bare:
            pass

        assert cost_calculator(model="ai6700/m", response=Bare()) == 0.0

    def test_handles_media_response(self):
        resp = MediaResponse(
            task_id="1",
            model="m",
            media_type="video",
            data=[MediaAsset(url="https://x")],
            cost=5.5,
            raw_cost=5.0,
            price_markup=1.1,
        )
        # MediaResponse has .cost attribute → fallback path
        assert cost_calculator(model="ai6700/m", response=resp) == 5.5

    def test_raw_cost_with_markup_fallback(self):
        """If only ai6700_raw_cost is in hidden_params (response_cost wiped),
        fall back to raw × markup."""

        class FakeResp:
            _hidden_params = {
                "ai6700_raw_cost": 2.0,
                "ai6700_price_markup": 1.5,
            }

        assert cost_calculator(model="ai6700/m", response=FakeResp()) == pytest.approx(
            3.0
        )
