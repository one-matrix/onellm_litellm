"""Tests for the /v1/media/models* routes (registered-models view).

The routes pull from ``llm_router.get_model_list()`` and never call upstream,
so we monkeypatch the global ``llm_router`` with a tiny stub.
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from fastapi import HTTPException

from litellm.proxy.media_endpoints import endpoints as media_endpoints
from litellm.proxy.media_endpoints.endpoints import (
    _channel_view,
    _is_media_deployment,
    _strip_tier,
    media_model_detail,
    media_model_pricing,
    media_models_list,
)


def _dep(name: str, *, mtype: str = "video", **info: Any) -> Dict[str, Any]:
    """Build a fake deployment dict in the shape llm_router.get_model_list returns."""
    base_info = {
        "ai6700_media_type": mtype,
        "channel_group": info.pop("channel_group", "default"),
        "billing_method": info.pop("billing_method", "按秒"),
        "is_active": info.pop("is_active", True),
        "base_price": info.pop("base_price", 0.1),
        "price_markup": info.pop("price_markup", 1.1),
    }
    base_info.update(info)
    return {
        "model_name": name,
        "litellm_params": {"model": f"ai6700/{_strip_tier(name)}"},
        "model_info": base_info,
    }


def _install_router(monkeypatch, deployments: List[Dict[str, Any]]):
    """Pretend the proxy has an llm_router with these deployments registered."""
    fake_router = SimpleNamespace(get_model_list=lambda: deployments)
    # The route imports llm_router lazily from proxy_server inside _list_media_deployments
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", fake_router, raising=False)


class TestPureHelpers:
    def test_strip_tier_removes_suffix(self):
        assert _strip_tier("foo-tier1") == "foo"
        assert _strip_tier("foo-tier12") == "foo"

    def test_strip_tier_keeps_non_tier_suffixes(self):
        # 'tier' without integer suffix isn't a tier
        assert _strip_tier("foo-tier") == "foo-tier"
        assert _strip_tier("foo") == "foo"
        # Internal "-tier1-" must not be touched
        assert _strip_tier("foo-tier1-bar") == "foo-tier1-bar"

    def test_is_media_deployment_requires_ai6700_media_type(self):
        assert _is_media_deployment({"model_info": {"ai6700_media_type": "video"}})
        # Just having mode is not enough — must be on AI6700's async-poll surface.
        assert not _is_media_deployment({"model_info": {"mode": "image_generation"}})
        assert not _is_media_deployment({"model_info": {}})
        assert not _is_media_deployment({})

    def test_channel_view_extracts_pricing_fields(self):
        dep = _dep(
            "grok-video-3-tier1",
            channel_group="高级渠道",
            base_price=0.5,
            price_markup=1.2,
            option_prices=[{"foo": "bar"}],
        )
        view = _channel_view(dep)
        assert view["name"] == "grok-video-3-tier1"
        assert view["channel_group"] == "高级渠道"
        assert view["base_price"] == 0.5
        assert view["price_markup"] == 1.2
        assert view["option_prices"] == [{"foo": "bar"}]


class TestMediaModelsList:
    async def test_empty_when_no_router(self, monkeypatch):
        import litellm.proxy.proxy_server as proxy_server

        monkeypatch.setattr(proxy_server, "llm_router", None, raising=False)
        result = await media_models_list(user_api_key_dict=None)  # type: ignore[arg-type]
        assert result == {"object": "list", "data": []}

    async def test_groups_tiers_by_base_name(self, monkeypatch):
        _install_router(
            monkeypatch,
            [
                _dep("grok-video-3", mtype="video", display_name="Grok V3"),
                _dep("grok-video-3-tier1", mtype="video"),
                _dep("grok-video-3-tier2", mtype="video"),
                _dep("doubao-tts", mtype="audio", display_name="豆包 TTS"),
            ],
        )
        result = await media_models_list(user_api_key_dict=None)  # type: ignore[arg-type]
        data = {row["name"]: row for row in result["data"]}
        assert set(data) == {"grok-video-3", "doubao-tts"}
        assert data["grok-video-3"]["channel_count"] == 3
        assert data["grok-video-3"]["type"] == "video"
        # display_name comes from the bare-name deployment, not the tiers
        assert data["grok-video-3"]["display_name"] == "Grok V3"
        assert data["doubao-tts"]["channel_count"] == 1

    async def test_filters_out_non_media(self, monkeypatch):
        _install_router(
            monkeypatch,
            [
                _dep("grok-video-3", mtype="video"),
                # No ai6700_media_type — a chat model, should be filtered.
                {
                    "model_name": "claude-sonnet-4-6",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-6"},
                    "model_info": {"mode": "chat"},
                },
            ],
        )
        result = await media_models_list(user_api_key_dict=None)  # type: ignore[arg-type]
        names = [r["name"] for r in result["data"]]
        assert names == ["grok-video-3"]


class TestMediaModelDetail:
    async def test_404_when_unknown(self, monkeypatch):
        _install_router(monkeypatch, [_dep("grok-video-3")])
        with pytest.raises(HTTPException) as exc:
            await media_model_detail(
                model="does-not-exist", user_api_key_dict=None  # type: ignore[arg-type]
            )
        assert exc.value.status_code == 404

    async def test_returns_bare_deployment_when_present(self, monkeypatch):
        _install_router(
            monkeypatch,
            [
                _dep("grok-video-3", display_name="Grok V3"),
                _dep("grok-video-3-tier1", display_name="Grok V3 cheap"),
            ],
        )
        info = await media_model_detail(
            model="grok-video-3", user_api_key_dict=None  # type: ignore[arg-type]
        )
        assert info["name"] == "grok-video-3"
        assert info["type"] == "video"
        # Bare-name wins, not the tier.
        assert info["display_name"] == "Grok V3"
        assert info["litellm_model"] == "ai6700/grok-video-3"

    async def test_falls_back_to_tier_if_no_bare(self, monkeypatch):
        _install_router(
            monkeypatch,
            [_dep("grok-video-3-tier1", display_name="tier-only")],
        )
        info = await media_model_detail(
            model="grok-video-3", user_api_key_dict=None  # type: ignore[arg-type]
        )
        # Name is normalized to the queried base, not the underlying tier
        assert info["name"] == "grok-video-3"
        assert info["display_name"] == "tier-only"


class TestMediaModelPricing:
    async def test_404_when_unknown(self, monkeypatch):
        _install_router(monkeypatch, [_dep("grok-video-3")])
        with pytest.raises(HTTPException) as exc:
            await media_model_pricing(
                model="does-not-exist", user_api_key_dict=None  # type: ignore[arg-type]
            )
        assert exc.value.status_code == 404

    async def test_returns_all_tiers_with_bare_first(self, monkeypatch):
        _install_router(
            monkeypatch,
            [
                _dep("grok-video-3-tier2", base_price=0.3),
                _dep("grok-video-3", base_price=0.1),
                _dep("grok-video-3-tier1", base_price=0.2),
                _dep("other-model"),
            ],
        )
        result = await media_model_pricing(
            model="grok-video-3", user_api_key_dict=None  # type: ignore[arg-type]
        )
        assert result["model"] == "grok-video-3"
        assert result["type"] == "video"
        names = [c["name"] for c in result["channels"]]
        assert names == ["grok-video-3", "grok-video-3-tier1", "grok-video-3-tier2"]
        # Pricing fields plumbed through
        prices = [c["base_price"] for c in result["channels"]]
        assert prices == [0.1, 0.2, 0.3]
