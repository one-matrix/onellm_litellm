"""Tests for AI6700 audio/TTS/music transformation."""

import pytest

from litellm.llms.ai6700 import AI6700Error
from litellm.llms.ai6700.audio import AI6700AudioConfig, AI6700AudioResponse


@pytest.fixture
def cfg() -> AI6700AudioConfig:
    return AI6700AudioConfig()


@pytest.fixture
def terminal_status() -> dict:
    return {
        "task_id": 9876,
        "status": "生成完成",
        "status_group": "已完成",
        "is_final": True,
        "progress": "100%",
        "result_url": "https://cdn/audio/x.mp3",
        "result_type": "audio",
        "cost": 0.05,
        "channel_group": "火山引擎官方直连",
        "created_at": "2026-03-17T10:00:00Z",
        "completed_at": "2026-03-17T10:00:20Z",
        "duration_seconds": 20,
    }


class TestEnvironment:
    def test_validates_api_key(self, cfg):
        h = cfg.validate_environment({}, "ai6700/x", api_key="sk")
        assert h["Authorization"] == "Bearer sk"

    def test_missing_key_raises(self, cfg):
        with pytest.raises(AI6700Error):
            cfg.validate_environment({}, "x", api_key=None)


class TestSpeedToSpeechRate:
    @pytest.mark.parametrize(
        "speed,expected",
        [
            (1.0, "0"),
            (0.5, "-50"),
            (0.75, "-25"),
            (1.25, "25"),
            (1.5, "50"),
            (2.0, "100"),
            ("1.5", "50"),
            (1.1, "0"),  # snap to 1.0 (closer than 1.25)
            (1.3, "25"),  # snap to 1.25
            (3.0, "100"),  # cap
            (0.1, "-50"),  # floor
        ],
    )
    def test_snap(self, speed, expected):
        assert AI6700AudioConfig._coerce_speed_to_speech_rate(speed) == expected

    @pytest.mark.parametrize("v", [None, "bad", "abc"])
    def test_invalid(self, v):
        assert AI6700AudioConfig._coerce_speed_to_speech_rate(v) is None


class TestMapOpenaiParams:
    def test_speed_and_voice(self, cfg):
        m = cfg.map_openai_params({"speed": 1.5, "voice": "BV001"}, model="ai6700/m")
        assert m == {"speech_rate": "50", "voice": "BV001"}

    def test_response_format_not_passed(self, cfg):
        m = cfg.map_openai_params(
            {"response_format": "mp3", "voice": "BV001"}, model="m"
        )
        assert "response_format" not in m
        assert m["voice"] == "BV001"

    def test_emotion_passes_through(self, cfg):
        m = cfg.map_openai_params({"emotion": "happy", "emotion_scale": "4"}, model="m")
        assert m == {"emotion": "happy", "emotion_scale": "4"}

    def test_parameters_wins_for_speech_rate(self, cfg):
        m = cfg.map_openai_params(
            {"speed": 1.5, "parameters": {"speech_rate": "0"}}, model="m"
        )
        assert m["speech_rate"] == "0"

    def test_parameters_wins_for_voice(self, cfg):
        m = cfg.map_openai_params(
            {"voice": "foo", "parameters": {"voice": "bar"}}, model="m"
        )
        assert m["voice"] == "bar"

    def test_combined(self, cfg):
        m = cfg.map_openai_params(
            {
                "speed": 1.5,
                "voice": "BV001",
                "emotion": "happy",
                "parameters": {"model_version": "flash"},
                "response_format": "mp3",  # should drop
            },
            model="m",
        )
        assert m == {
            "model_version": "flash",
            "speech_rate": "50",
            "voice": "BV001",
            "emotion": "happy",
        }


class TestBuildRequestBody:
    def test_minimal(self, cfg):
        b = cfg.build_request_body(model="ai6700/doubao-tts-2.0", prompt="你好")
        assert b == {"model": "doubao-tts-2.0", "prompt": "你好"}

    def test_with_params(self, cfg):
        b = cfg.build_request_body(model="m", prompt="hi", params={"voice": "BV001"})
        assert b["params"] == {"voice": "BV001"}

    def test_empty_prompt_raises(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.build_request_body(model="m", prompt="")
        assert exc.value.status_code == 400
        # informative error message references litellm.aspeech compat
        assert "input" in exc.value.message.lower()


class TestTransformTaskToAudioResponse:
    def test_basic(self, cfg, terminal_status):
        resp = cfg.transform_task_to_audio_response(
            status=terminal_status, model="ai6700/doubao-tts-2.0", price_markup=1.1
        )
        assert isinstance(resp, AI6700AudioResponse)
        assert resp.task_id == 9876
        assert resp.url == "https://cdn/audio/x.mp3"
        assert resp.model == "doubao-tts-2.0"
        assert resp.result_type == "audio"
        assert resp.object == "audio.task"
        assert resp.status == "completed"

    def test_cost_with_markup(self, cfg, terminal_status):
        resp = cfg.transform_task_to_audio_response(
            status=terminal_status, model="ai6700/m", price_markup=1.1
        )
        assert resp.raw_cost == 0.05
        assert resp.cost == pytest.approx(0.055)

    def test_hidden_params_wiring(self, cfg, terminal_status):
        resp = cfg.transform_task_to_audio_response(
            status=terminal_status, model="ai6700/m", price_markup=1.1
        )
        hp = resp.__dict__["_hidden_params"]
        assert hp["response_cost"] == pytest.approx(0.055)
        assert hp["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ] == pytest.approx(0.055)

    def test_music_result_type(self, cfg, terminal_status):
        """Same config handles music — result_type comes from upstream."""
        status_music = {
            **terminal_status,
            "result_type": "music",
            "result_url": "https://cdn/music/y.mp3",
        }
        resp = cfg.transform_task_to_audio_response(
            status=status_music, model="ai6700/some-music", price_markup=1.0
        )
        assert resp.result_type == "music"
        assert resp.cost == 0.05

    def test_dict_style_access(self, cfg, terminal_status):
        resp = cfg.transform_task_to_audio_response(
            status=terminal_status, model="ai6700/m", price_markup=1.0
        )
        assert resp["task_id"] == 9876
        assert resp.get("url") == "https://cdn/audio/x.mp3"
        assert "url" in resp
        assert "nonexistent" not in resp

    def test_duration_seconds(self, cfg, terminal_status):
        resp = cfg.transform_task_to_audio_response(status=terminal_status, model="m")
        assert resp.duration_seconds == 20.0

    def test_missing_url_raises(self, cfg):
        with pytest.raises(AI6700Error) as exc:
            cfg.transform_task_to_audio_response(
                status={"task_id": 1}, model="ai6700/m"
            )
        assert exc.value.status_code == 502
