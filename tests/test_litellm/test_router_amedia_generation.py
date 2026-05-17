"""Tests for Router.amedia_generation and its forward-kwargs filter.

The Router path translates a public ``model_name`` (e.g. ``kwvideo-v2``) to
the provider-prefixed deployment model (``ai6700/kwvideo-v2``) before calling
``litellm.amedia_generation``. Without this routing the proxy passes the raw
alias straight to litellm core, which fails with
"LLM Provider NOT provided" (see media/generations bug report).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.router import _build_amedia_forward_kwargs


def test_build_amedia_forward_kwargs_allowlists_litellm_params():
    """Cost fields and other deployment-only metadata must not leak upstream."""
    forward = _build_amedia_forward_kwargs(
        deployment_params={
            "model": "ai6700/kwvideo-v2",
            "api_base": "https://api.lk888.ai/api",
            "api_key": "sk-test",
            "timeout": 900,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 5.6e-05,
        },
        deployment_model_info={},
        request_kwargs={},
    )
    assert forward == {
        "api_base": "https://api.lk888.ai/api",
        "api_key": "sk-test",
        "timeout": 900,
    }


def test_build_amedia_forward_kwargs_injects_type_from_model_info():
    """Proxy aliases inherit ``type`` from model_info so callers don't need it."""
    forward = _build_amedia_forward_kwargs(
        deployment_params={"api_base": "x", "api_key": "k"},
        deployment_model_info={"ai6700_media_type": "video", "price_markup": 1.1},
        request_kwargs={},
    )
    assert forward["type"] == "video"
    assert forward["price_markup"] == 1.1


def test_build_amedia_forward_kwargs_mode_fallback():
    """When ``ai6700_media_type`` is absent, fall back to litellm-standard mode."""
    forward = _build_amedia_forward_kwargs(
        deployment_params={},
        deployment_model_info={"mode": "image_generation"},
        request_kwargs={},
    )
    assert forward["type"] == "image"


def test_build_amedia_forward_kwargs_caller_overrides_deployment():
    """Caller-supplied ``type``/``price_markup`` win over deployment defaults."""
    forward = _build_amedia_forward_kwargs(
        deployment_params={},
        deployment_model_info={"ai6700_media_type": "video", "price_markup": 1.1},
        request_kwargs={"type": "image", "price_markup": 2.0},
    )
    assert forward["type"] == "image"
    assert forward["price_markup"] == 2.0


def test_build_amedia_forward_kwargs_strips_router_internals():
    """Router/proxy-internal plumbing keys never reach amedia_generation."""
    forward = _build_amedia_forward_kwargs(
        deployment_params={},
        deployment_model_info={},
        request_kwargs={
            "size": "1280x720",
            "seconds": 5,
            "ratio": "16:9",  # unknown openai key — must still pass through
            "metadata": {"user": "abc"},
            "litellm_call_id": "x",
            "litellm_trace_id": "y",
            "proxy_server_request": object(),
            "user_api_key": "sk",
            "user_api_key_team_id": "t",
            "original_function": lambda: None,
            "num_retries": 3,
            "model_info": {"id": "1"},
            "model_group": "g",
            "caching": True,
            "specific_deployment": True,
        },
    )
    assert forward["size"] == "1280x720"
    assert forward["seconds"] == 5
    assert forward["ratio"] == "16:9"
    # All dropped:
    for dropped in (
        "metadata",
        "litellm_call_id",
        "litellm_trace_id",
        "proxy_server_request",
        "user_api_key",
        "user_api_key_team_id",
        "original_function",
        "num_retries",
        "model_info",
        "model_group",
        "caching",
        "specific_deployment",
    ):
        assert dropped not in forward


@pytest.mark.asyncio
async def test_router_amedia_generation_resolves_model_alias():
    """Router.amedia_generation must rewrite ``kwvideo-v2`` → ``ai6700/kwvideo-v2``.

    This is the regression: before the fix, the proxy bypassed the router and
    handed ``kwvideo-v2`` (no provider prefix) directly to ``litellm.amedia_generation``,
    which rejects unknown providers with "LLM Provider NOT provided".
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "kwvideo-v2",
                "litellm_params": {
                    "model": "ai6700/kwvideo-v2",
                    "api_base": "https://api.lk888.ai/api",
                    "api_key": "sk-test",
                    "timeout": 900,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 5.6e-05,
                },
                "model_info": {
                    "ai6700_media_type": "video",
                    "mode": "video_generation",
                    "price_markup": 1.1,
                },
            }
        ]
    )

    fake_response = MagicMock(name="MediaResponse")

    async def fake_amedia_generation(**kwargs):
        return fake_response

    with patch.object(
        litellm, "amedia_generation", side_effect=fake_amedia_generation
    ) as mock_amg:
        result = await router.amedia_generation(
            prompt="a cat",
            model="kwvideo-v2",
            size="1280x720",
            seconds=5,
        )

    assert result is fake_response
    mock_amg.assert_called_once()
    call_kwargs = mock_amg.call_args.kwargs

    # The most important assertion — provider prefix is applied.
    assert call_kwargs["model"] == "ai6700/kwvideo-v2"
    assert call_kwargs["prompt"] == "a cat"
    # Deployment credentials forwarded.
    assert call_kwargs["api_base"] == "https://api.lk888.ai/api"
    assert call_kwargs["api_key"] == "sk-test"
    assert call_kwargs["timeout"] == 900
    # model_info bridging.
    assert call_kwargs["type"] == "video"
    assert call_kwargs["price_markup"] == 1.1
    # Caller params preserved.
    assert call_kwargs["size"] == "1280x720"
    assert call_kwargs["seconds"] == 5
    # Cost fields and router internals did NOT leak.
    assert "input_cost_per_token" not in call_kwargs
    assert "output_cost_per_token" not in call_kwargs
    assert "metadata" not in call_kwargs
    assert "original_function" not in call_kwargs
    assert "num_retries" not in call_kwargs


@pytest.mark.asyncio
async def test_route_request_resolves_both_alias_and_prefixed_model_for_media():
    """Parity check with /v1/chat/completions alias resolution.

    Whether the caller sends ``kwvideo-v2`` (public ``model_name`` alias) or
    ``ai6700/kwvideo-v2`` (the provider-prefixed ``litellm_params.model``),
    /v1/media/generations must route through the Router. The prefixed form
    matches via ``deployment_names`` and is dispatched with
    ``specific_deployment=True`` — same branch acompletion uses.
    """
    from litellm.proxy.route_llm_request import route_request

    router = litellm.Router(
        model_list=[
            {
                "model_name": "kwvideo-v2",
                "litellm_params": {
                    "model": "ai6700/kwvideo-v2",
                    "api_base": "https://api.lk888.ai/api",
                    "api_key": "sk-test",
                    "timeout": 900,
                },
                "model_info": {"ai6700_media_type": "video"},
            }
        ]
    )

    sentinel = MagicMock(name="MediaResponse")

    async def fake_amedia_generation(**kwargs):
        return sentinel

    # Case 1: public alias — resolves via model_names branch.
    with patch.object(litellm, "amedia_generation", side_effect=fake_amedia_generation):
        call = await route_request(
            data={"model": "kwvideo-v2", "prompt": "a cat"},
            llm_router=router,
            user_model=None,
            route_type="amedia_generation",
        )
        result = await call
    assert result is sentinel

    # Case 2: provider-prefixed — resolves via deployment_names branch.
    assert "ai6700/kwvideo-v2" in router.deployment_names
    with patch.object(litellm, "amedia_generation", side_effect=fake_amedia_generation):
        call = await route_request(
            data={"model": "ai6700/kwvideo-v2", "prompt": "a cat"},
            llm_router=router,
            user_model=None,
            route_type="amedia_generation",
        )
        result = await call
    assert result is sentinel
