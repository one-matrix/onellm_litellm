"""Tests for AI6700 image transformation."""

import pytest

from litellm.llms.ai6700 import AI6700Error
from litellm.llms.ai6700.image_generation import AI6700ImageConfig
from litellm.types.utils import ImageResponse


@pytest.fixture
def cfg() -> AI6700ImageConfig:
    return AI6700ImageConfig()


@pytest.fixture
def terminal_status() -> dict:
    return {
        "task_id": 7777,
        "status": "生成完成",
        "status_group": "已完成",
        "is_final": True,
        "progress": "100%",
        "result_url": "https://cdn/img.png",
        "result_type": "image",
        "cost": 0.2,
        "channel_group": "默认",
        "created_at": "2026-03-17T10:00:00Z",
        "duration_seconds": 12,
    }


class TestEnvironment:
    def test_validates_api_key(self, cfg):
        h = cfg.validate_environment({}, "ai6700/x", api_key="sk-test")
        assert h["Authorization"] == "Bearer sk-test"

    def test_missing_api_key_raises(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.validate_environment({}, "x", api_key=None)
        assert exc.value.status_code == 401


class TestSizeCoercion:
    @pytest.mark.parametrize(
        "size,expected",
        [
            ("1024x1024", "1K"),
            ("1024x1792", "1K"),
            ("2048x2048", "2K"),
            ("4096x4096", "4K"),
            ("2K", "2K"),  # already shaped
            ("2k", "2K"),  # normalize case
            ("anything-else", "anything-else"),  # pass-through
        ],
    )
    def test_coerce(self, size, expected):
        assert AI6700ImageConfig._coerce_size(size) == expected

    @pytest.mark.parametrize("v", [None, "", 1024])
    def test_invalid(self, v):
        assert AI6700ImageConfig._coerce_size(v) is None


class TestCountCoercion:
    @pytest.mark.parametrize(
        "candidates,expected",
        [
            ((None, None), None),
            ((None, 3), 3),
            (("2",), 2),
            ((0,), None),  # zero invalid
            ((-1,), None),
            ((None, None, 5), 5),
            (("bad",), None),
        ],
    )
    def test_coerce(self, candidates, expected):
        assert AI6700ImageConfig._coerce_count(*candidates) == expected


class TestExtractCount:
    def test_n(self, cfg):
        assert cfg.extract_count({"n": 2}) == 2

    def test_num_images(self, cfg):
        assert cfg.extract_count({"num_images": 4}) == 4

    def test_n_preferred_over_num_images(self, cfg):
        assert cfg.extract_count({"n": 2, "num_images": 4}) == 2

    def test_neither(self, cfg):
        assert cfg.extract_count({}) is None


class TestMapOpenaiParams:
    def test_size_and_image(self, cfg):
        mapped = cfg.map_openai_params(
            {"size": "1024x1024", "image_url": "https://cdn/r.jpg"}, model="ai6700/m"
        )
        assert mapped["size"] == "1K"
        assert mapped["images"] == "https://cdn/r.jpg"

    def test_parameters_wins(self, cfg):
        mapped = cfg.map_openai_params(
            {"size": "1024x1024", "parameters": {"size": "4K"}}, model="ai6700/m"
        )
        assert mapped["size"] == "4K"

    def test_unknown_keys_pass_through(self, cfg):
        mapped = cfg.map_openai_params({"aspect_ratio": "16:9"}, model="ai6700/m")
        assert mapped == {"aspect_ratio": "16:9"}

    def test_openai_standard_keys_not_passed_through(self, cfg):
        mapped = cfg.map_openai_params(
            {"n": 2, "quality": "hd", "style": "vivid", "response_format": "url"},
            model="ai6700/m",
        )
        assert mapped == {}  # all are OpenAI-standard, dropped

    def test_image_alias_chain(self, cfg):
        """input_reference / image / image_url all map to 'images'."""
        for key in ("image_url", "image", "input_reference"):
            mapped = cfg.map_openai_params({key: "https://cdn/a.jpg"}, model="m")
            assert mapped["images"] == "https://cdn/a.jpg"

    def test_local_path_rejected(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.map_openai_params({"image_url": "/local"}, model="m")
        assert exc.value.status_code == 400


class TestUploadValidation:
    def test_accepts_url(self):
        AI6700ImageConfig._validate_upload_value("images", "https://x.com/a.jpg")

    def test_accepts_list(self):
        AI6700ImageConfig._validate_upload_value("images", ["https://a", "https://b"])

    @pytest.mark.parametrize(
        "bad", ["file:///x", "/local/path", "data:image/png;base64,xx", ""]
    )
    def test_rejects_bad(self, bad):
        with pytest.raises(AI6700Error) as exc:
            AI6700ImageConfig._validate_upload_value("images", bad)
        assert exc.value.status_code == 400


class TestBuildRequestBody:
    def test_minimal(self, cfg):
        b = cfg.build_request_body(model="ai6700/m", prompt="cat")
        assert b == {"model": "m", "prompt": "cat"}

    def test_with_params_and_count(self, cfg):
        b = cfg.build_request_body(
            model="m", prompt="cat", params={"size": "2K"}, count=3
        )
        assert b == {
            "model": "m",
            "prompt": "cat",
            "params": {"size": "2K"},
            "count": 3,
        }

    def test_empty_prompt_raises(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.build_request_body(model="m", prompt="")
        assert exc.value.status_code == 400


class TestTransformTaskToImageResponse:
    def test_single_url(self, cfg, terminal_status):
        resp = cfg.transform_task_to_image_response(
            status=terminal_status, model="ai6700/m", price_markup=1.1
        )
        assert isinstance(resp, ImageResponse)
        assert len(resp.data) == 1
        assert resp.data[0].url == "https://cdn/img.png"

    def test_provider_specific_fields(self, cfg, terminal_status):
        resp = cfg.transform_task_to_image_response(
            status=terminal_status, model="ai6700/m", price_markup=1.0
        )
        psf = resp.data[0].provider_specific_fields
        assert psf["ai6700_task_id"] == 7777
        assert psf["ai6700_channel_group"] == "默认"

    def test_hidden_params_cost(self, cfg, terminal_status):
        resp = cfg.transform_task_to_image_response(
            status=terminal_status, model="ai6700/m", price_markup=1.1
        )
        hp = resp._hidden_params
        assert hp["response_cost"] == pytest.approx(0.22)
        assert hp["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ] == pytest.approx(0.22)
        assert hp["ai6700_raw_cost"] == 0.2

    def test_multiple_urls_via_extras(self, cfg, terminal_status):
        status = {
            **terminal_status,
            "result_urls": ["https://cdn/a.png", "https://cdn/b.png"],
        }
        resp = cfg.transform_task_to_image_response(
            status=status, model="ai6700/m", price_markup=1.0
        )
        # primary + 2 extras = 3 (extras dedupe against primary if same)
        assert len(resp.data) == 3
        assert resp.data[0].url == "https://cdn/img.png"
        assert resp.data[1].url == "https://cdn/a.png"
        assert resp.data[2].url == "https://cdn/b.png"

    def test_revised_prompt(self, cfg, terminal_status):
        resp = cfg.transform_task_to_image_response(
            status=terminal_status,
            model="ai6700/m",
            revised_prompt="a cute cat",
        )
        assert resp.data[0].revised_prompt == "a cute cat"

    def test_missing_result_url_raises(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.transform_task_to_image_response(
                status={"task_id": 1}, model="ai6700/m"
            )
        assert exc.value.status_code == 502
