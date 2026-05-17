"""Tests for AI6700 video transformation (shape conversion only)."""

import pytest

from litellm.llms.ai6700 import AI6700Error
from litellm.llms.ai6700.videos import AI6700VideoConfig
from litellm.types.videos.main import VideoObject


@pytest.fixture
def cfg() -> AI6700VideoConfig:
    return AI6700VideoConfig()


@pytest.fixture
def terminal_status() -> dict:
    return {
        "task_id": 12345,
        "status": "生成完成",
        "status_group": "已完成",
        "is_final": True,
        "progress": "100%",
        "result_url": "https://cdn.example.com/video/abc.mp4",
        "result_type": "video",
        "cost": 1.5,
        "channel_group": "标准渠道",
        "created_at": "2026-03-17T10:00:00Z",
        "completed_at": "2026-03-17T10:00:42Z",
        "duration_seconds": 42,
    }


class TestEnvironment:
    def test_validates_api_key(self, cfg):
        h = cfg.validate_environment({}, "ai6700/x", api_key="sk-test")
        assert h["Authorization"] == "Bearer sk-test"
        assert h["Content-Type"] == "application/json"

    def test_missing_api_key_raises(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.validate_environment({}, "x", api_key=None)
        assert exc.value.status_code == 401

    def test_default_auth_overrides_when_user_omits(self, cfg):
        h = cfg.validate_environment({"X-Custom": "v"}, "ai6700/x", api_key="sk")
        assert h["X-Custom"] == "v"
        assert h["Authorization"] == "Bearer sk"

    def test_user_authorization_preserved(self, cfg):
        h = cfg.validate_environment(
            {"Authorization": "Custom xyz"}, "ai6700/x", api_key="sk"
        )
        assert h["Authorization"] == "Custom xyz"


class TestPrefixStrip:
    def test_with_prefix(self):
        assert AI6700VideoConfig.strip_provider_prefix("ai6700/foo") == "foo"

    def test_without_prefix(self):
        assert AI6700VideoConfig.strip_provider_prefix("foo") == "foo"


class TestSizeCoercion:
    @pytest.mark.parametrize(
        "size,expected",
        [
            ("1280x720", "720p"),
            ("1920x1080", "1080p"),
            ("3840x2160", "4K"),
            ("720p", "720p"),  # already shaped
            ("4K", "4K"),
            ("anything-else", "anything-else"),  # pass-through
        ],
    )
    def test_coerce(self, size, expected):
        assert AI6700VideoConfig._coerce_size_to_resolution(size) == expected

    def test_none(self):
        assert AI6700VideoConfig._coerce_size_to_resolution(None) is None

    def test_empty(self):
        assert AI6700VideoConfig._coerce_size_to_resolution("") is None

    def test_non_string(self):
        assert AI6700VideoConfig._coerce_size_to_resolution(1280) is None


class TestDurationCoercion:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (5, "5"),
            (5.0, "5"),
            ("5", "5"),
            ("5s", "5"),
            ("5S", "5"),
            (5.5, "5.5"),
        ],
    )
    def test_valid(self, value, expected):
        assert AI6700VideoConfig._coerce_duration(value) == expected

    @pytest.mark.parametrize("value", [None, "bad", "not-a-number"])
    def test_invalid_returns_none(self, value):
        assert AI6700VideoConfig._coerce_duration(value) is None


class TestUploadValidation:
    def test_accepts_https_url(self):
        AI6700VideoConfig._validate_upload_value("images", "https://x.com/a.jpg")

    def test_accepts_http_url(self):
        AI6700VideoConfig._validate_upload_value("images", "http://x.com/a.jpg")

    def test_accepts_url_list(self):
        AI6700VideoConfig._validate_upload_value("images", ["https://a", "https://b"])

    @pytest.mark.parametrize(
        "bad",
        ["file:///x", "/local/path.jpg", "data:image/png;base64,xx", "", "ftp://x"],
    )
    def test_rejects_non_url(self, bad):
        with pytest.raises(AI6700Error) as exc:
            AI6700VideoConfig._validate_upload_value("images", bad)
        assert exc.value.status_code == 400

    def test_rejects_int(self):
        with pytest.raises(AI6700Error):
            AI6700VideoConfig._validate_upload_value("images", 12345)


class TestMapOpenaiParams:
    def test_full_mapping(self, cfg):
        mapped = cfg.map_openai_params(
            {
                "size": "1280x720",
                "seconds": 8,
                "input_reference": "https://cdn/ref.jpg",
                "parameters": {"ratio": "16:9", "generate_audio": "false"},
            },
            model="ai6700/m",
        )
        assert mapped == {
            "ratio": "16:9",
            "generate_audio": "false",
            "resolution": "720p",
            "audio_duration": "8",
            "images": "https://cdn/ref.jpg",
        }

    def test_explicit_parameters_win(self, cfg):
        mapped = cfg.map_openai_params(
            {"size": "1280x720", "parameters": {"resolution": "1080p"}},
            model="ai6700/m",
        )
        assert mapped["resolution"] == "1080p"

    def test_unknown_keys_pass_through(self, cfg):
        mapped = cfg.map_openai_params(
            {"ratio": "9:16", "seconds": "5"}, model="ai6700/m"
        )
        assert mapped["ratio"] == "9:16"
        assert mapped["audio_duration"] == "5"

    def test_image_alias_for_input_reference(self, cfg):
        mapped = cfg.map_openai_params({"image": "https://cdn/a.jpg"}, model="ai6700/m")
        assert mapped["images"] == "https://cdn/a.jpg"

    def test_input_reference_wins_over_image(self, cfg):
        mapped = cfg.map_openai_params(
            {
                "input_reference": "https://cdn/ref.jpg",
                "image": "https://cdn/other.jpg",
            },
            model="ai6700/m",
        )
        # input_reference is the first one we look at
        assert mapped["images"] == "https://cdn/ref.jpg"

    def test_local_path_rejected_during_mapping(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.map_openai_params({"input_reference": "/local/x.jpg"}, model="ai6700/m")
        assert exc.value.status_code == 400

    def test_none_values_skipped(self, cfg):
        mapped = cfg.map_openai_params(
            {"size": None, "seconds": None, "input_reference": None},
            model="ai6700/m",
        )
        assert mapped == {}


class TestBuildRequestBody:
    def test_minimal(self, cfg):
        body = cfg.build_request_body(model="ai6700/m", prompt="hi")
        assert body == {"model": "m", "prompt": "hi"}

    def test_with_params(self, cfg):
        body = cfg.build_request_body(
            model="ai6700/m", prompt="hi", params={"resolution": "720p"}
        )
        assert body["params"] == {"resolution": "720p"}

    def test_with_count(self, cfg):
        body = cfg.build_request_body(model="m", prompt="hi", count=3)
        assert body["count"] == 3

    def test_count_one_omitted(self, cfg):
        body = cfg.build_request_body(model="m", prompt="hi", count=1)
        assert "count" not in body

    def test_empty_prompt_raises(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.build_request_body(model="m", prompt="", params=None)
        assert exc.value.status_code == 400


class TestProgressToPercent:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("100%", 100),
            ("50%", 50),
            (75, 75),
            (75.5, 75),
            (None, None),
            ("bad", None),
            ("", None),
        ],
    )
    def test_progress(self, value, expected):
        assert AI6700VideoConfig._progress_to_percent(value) == expected


class TestTransformTaskToVideoObject:
    def test_basic(self, cfg, terminal_status):
        vid = cfg.transform_task_to_video_object(
            status=terminal_status, model="ai6700/grok-video-3", price_markup=1.1
        )
        assert isinstance(vid, VideoObject)
        assert vid.id == "12345"
        assert vid.status == "completed"
        assert vid.progress == 100
        assert vid.model == "grok-video-3"

    def test_timestamps(self, cfg, terminal_status):
        vid = cfg.transform_task_to_video_object(
            status=terminal_status, model="ai6700/x", price_markup=1.0
        )
        assert vid.created_at is not None
        assert vid.completed_at is not None
        assert vid.completed_at > vid.created_at

    def test_hidden_params_response_cost(self, cfg, terminal_status):
        vid = cfg.transform_task_to_video_object(
            status=terminal_status, model="ai6700/x", price_markup=1.1
        )
        hp = vid.__dict__["_hidden_params"]
        assert hp["response_cost"] == pytest.approx(1.65)
        assert hp["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ] == pytest.approx(1.65)
        assert hp["ai6700_raw_cost"] == 1.5
        assert hp["ai6700_channel_group"] == "标准渠道"

    def test_missing_result_url_raises(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.transform_task_to_video_object(
                status={"task_id": 1, "cost": 0}, model="ai6700/x"
            )
        assert exc.value.status_code == 502

    def test_extra_hidden_merged(self, cfg, terminal_status):
        vid = cfg.transform_task_to_video_object(
            status=terminal_status,
            model="ai6700/x",
            price_markup=1.0,
            extra_hidden={"custom_field": "test"},
        )
        assert vid.__dict__["_hidden_params"]["custom_field"] == "test"
