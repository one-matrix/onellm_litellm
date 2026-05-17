"""End-to-end tests for litellm.amedia_generation."""

from unittest.mock import MagicMock

import httpx
import orjson
import pytest

import litellm
from litellm.llms.ai6700 import AI6700Error
from litellm.types.media import MediaAsset, MediaResponse


def _fake_response(status_code: int, payload: dict) -> MagicMock:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status_code
    r.headers = {}
    r.text = str(payload)
    r.json.return_value = payload
    return r


class _FakeClient:
    def __init__(self, submit_payload, status_payloads):
        self.submit_payload = submit_payload
        self.status_payloads = list(status_payloads)
        self.posts = []
        self.gets = []

    async def post(self, url=None, headers=None, json=None):
        self.posts.append({"url": url, "headers": headers, "json": json})
        return _fake_response(200, self.submit_payload)

    async def get(self, url=None, params=None, headers=None):
        self.gets.append({"url": url, "params": params, "headers": headers})
        return _fake_response(200, self.status_payloads.pop(0))

    async def close(self):
        pass


@pytest.fixture
def video_fake_client() -> _FakeClient:
    return _FakeClient(
        submit_payload={"msg": "ok", "code": 200, "data": {"任务id": 1001}},
        status_payloads=[
            {"task_id": 1001, "is_final": False, "progress": "50%"},
            {
                "task_id": 1001,
                "is_final": True,
                "status_group": "已完成",
                "result_url": "https://cdn/v.mp4",
                "result_type": "video",
                "cost": 1.5,
                "channel_group": "标准",
                "created_at": "2026-03-17T10:00:00Z",
                "completed_at": "2026-03-17T10:00:42Z",
                "duration_seconds": 42,
            },
        ],
    )


class TestProviderResolution:
    async def test_wrong_provider_rejected(self):
        with pytest.raises(AI6700Error) as exc:
            await litellm.amedia_generation(
                model="openai/dall-e-3",
                prompt="x",
                type="image",
                api_key="sk",
            )
        assert exc.value.status_code == 400
        assert "ai6700" in exc.value.message


class TestApiKeyResolution:
    async def test_missing_key_rejected(self, monkeypatch):
        # Ensure no env fallback
        monkeypatch.delenv("LINGKE_API_KEY", raising=False)
        monkeypatch.delenv("AI6700_API_KEY", raising=False)
        with pytest.raises(AI6700Error) as exc:
            await litellm.amedia_generation(
                model="ai6700/grok-video-3", prompt="hi", type="video"
            )
        assert exc.value.status_code == 401

    async def test_env_lingke_key_used(self, monkeypatch, video_fake_client):
        monkeypatch.setenv("LINGKE_API_KEY", "from-env")
        resp = await litellm.amedia_generation(
            model="ai6700/grok-video-3",
            prompt="hi",
            type="video",
            client=video_fake_client,
            poll_interval=0.001,
        )
        assert resp.task_id == "1001"
        assert (
            video_fake_client.posts[0]["headers"]["Authorization"] == "Bearer from-env"
        )

    async def test_ai6700_api_key_fallback(self, monkeypatch, video_fake_client):
        monkeypatch.delenv("LINGKE_API_KEY", raising=False)
        monkeypatch.setenv("AI6700_API_KEY", "alt-key")
        resp = await litellm.amedia_generation(
            model="ai6700/grok-video-3",
            prompt="hi",
            type="video",
            client=video_fake_client,
            poll_interval=0.001,
        )
        assert resp.task_id == "1001"


class TestMediaTypeResolution:
    async def test_explicit_type(self, video_fake_client):
        resp = await litellm.amedia_generation(
            model="ai6700/grok-video-3",
            prompt="hi",
            type="video",
            api_key="sk",
            client=video_fake_client,
            poll_interval=0.001,
        )
        assert resp.media_type == "video"

    async def test_unknown_type_rejected(self):
        with pytest.raises(AI6700Error) as exc:
            await litellm.amedia_generation(
                model="ai6700/grok-video-3",
                prompt="hi",
                type="something-weird",
                api_key="sk",
            )
        assert exc.value.status_code == 400

    async def test_missing_type_without_model_info_rejected(self):
        with pytest.raises(AI6700Error) as exc:
            await litellm.amedia_generation(
                model="ai6700/totally-unknown-model",
                prompt="hi",
                api_key="sk",
            )
        assert exc.value.status_code == 400
        assert (
            "media_type" in exc.value.message.lower()
            or "type" in exc.value.message.lower()
        )


class TestVideoEndToEnd:
    async def test_basic_video_flow(self, video_fake_client):
        resp = await litellm.amedia_generation(
            model="ai6700/grok-video-3",
            prompt="一只猫在草地奔跑",
            type="video",
            api_key="sk",
            client=video_fake_client,
            poll_interval=0.001,
            price_markup=1.1,
        )
        assert isinstance(resp, MediaResponse)
        assert resp.task_id == "1001"
        assert resp.media_type == "video"
        assert resp.model == "grok-video-3"
        assert resp.url == "https://cdn/v.mp4"
        assert resp.raw_cost == 1.5
        assert resp.cost == pytest.approx(1.65)
        assert resp.channel_group == "标准"
        assert resp.duration_seconds == 42.0
        assert resp.provider == "ai6700"

    async def test_openai_params_auto_mapped(self, video_fake_client):
        await litellm.amedia_generation(
            model="ai6700/grok-video-3",
            prompt="hi",
            type="video",
            size="1280x720",
            seconds=5,
            api_key="sk",
            client=video_fake_client,
            poll_interval=0.001,
        )
        body = video_fake_client.posts[0]["json"]
        assert body["params"]["resolution"] == "720p"
        assert body["params"]["audio_duration"] == "5"

    async def test_caller_params_merge_with_auto_map(self, video_fake_client):
        await litellm.amedia_generation(
            model="ai6700/grok-video-3",
            prompt="hi",
            type="video",
            size="1280x720",
            params={"generate_audio": "false"},
            api_key="sk",
            client=video_fake_client,
            poll_interval=0.001,
        )
        params = video_fake_client.posts[0]["json"]["params"]
        # caller params win, auto-mapped fields added
        assert params["generate_audio"] == "false"
        assert params["resolution"] == "720p"

    async def test_polling_executed_until_terminal(self, video_fake_client):
        await litellm.amedia_generation(
            model="ai6700/grok-video-3",
            prompt="hi",
            type="video",
            api_key="sk",
            client=video_fake_client,
            poll_interval=0.001,
        )
        # 1 submit + 2 status (mid + terminal)
        assert len(video_fake_client.posts) == 1
        assert len(video_fake_client.gets) == 2


class TestImageEndToEnd:
    async def test_n_extracted_as_count(self):
        client = _FakeClient(
            submit_payload={"data": {"任务id": 2002}},
            status_payloads=[
                {
                    "task_id": 2002,
                    "is_final": True,
                    "status_group": "已完成",
                    "result_url": "https://cdn/a.png",
                    "result_type": "image",
                    "cost": 0.6,
                    "channel_group": "默认",
                    "created_at": "2026-03-17T10:00:00Z",
                }
            ],
        )
        await litellm.amedia_generation(
            model="ai6700/doubao-seedream-4-5",
            prompt="cat",
            type="image",
            n=2,
            size="2048x2048",
            api_key="sk",
            client=client,
            poll_interval=0.001,
        )
        body = client.posts[0]["json"]
        assert body["count"] == 2
        assert body["params"]["size"] == "2K"

    async def test_multiple_result_urls(self):
        client = _FakeClient(
            submit_payload={"data": {"任务id": 2003}},
            status_payloads=[
                {
                    "task_id": 2003,
                    "is_final": True,
                    "status_group": "已完成",
                    "result_url": "https://cdn/a.png",
                    "result_urls": ["https://cdn/b.png", "https://cdn/c.png"],
                    "result_type": "image",
                    "cost": 0.6,
                    "created_at": "2026-03-17T10:00:00Z",
                }
            ],
        )
        resp = await litellm.amedia_generation(
            model="ai6700/m",
            prompt="cat",
            type="image",
            api_key="sk",
            client=client,
            poll_interval=0.001,
        )
        assert len(resp.data) == 3
        assert resp.url == "https://cdn/a.png"
        assert resp.urls == [
            "https://cdn/a.png",
            "https://cdn/b.png",
            "https://cdn/c.png",
        ]


class TestAudioEndToEnd:
    async def test_speed_snap_and_voice_pass_through(self):
        client = _FakeClient(
            submit_payload={"data": {"任务id": 3003}},
            status_payloads=[
                {
                    "task_id": 3003,
                    "is_final": True,
                    "status_group": "已完成",
                    "result_url": "https://cdn/x.mp3",
                    "result_type": "audio",
                    "cost": 0.05,
                    "channel_group": "火山",
                    "created_at": "2026-03-17T10:00:00Z",
                }
            ],
        )
        resp = await litellm.amedia_generation(
            model="ai6700/doubao-tts-2.0",
            prompt="你好",
            type="audio",
            speed=1.5,
            voice="BV001",
            parameters={"emotion": "happy"},
            api_key="sk",
            client=client,
            poll_interval=0.001,
        )
        assert resp.media_type == "audio"
        assert resp.url == "https://cdn/x.mp3"
        body = client.posts[0]["json"]
        # speed → speech_rate, voice + emotion all preserved
        assert body["params"]["speech_rate"] == "50"
        assert body["params"]["voice"] == "BV001"
        assert body["params"]["emotion"] == "happy"


class TestUploadValidation:
    async def test_bad_upload_url_rejected_before_http(self):
        client = _FakeClient(submit_payload={"data": {"任务id": 0}}, status_payloads=[])
        with pytest.raises(AI6700Error) as exc:
            await litellm.amedia_generation(
                model="ai6700/m",
                prompt="hi",
                type="image",
                image_url="/local/x.jpg",
                api_key="sk",
                client=client,
            )
        assert exc.value.status_code == 400
        # No HTTP request issued
        assert len(client.posts) == 0


class TestMediaResponseSerialization:
    """MediaResponse must serialise cleanly for proxy ORJSONResponse."""

    async def test_orjson_round_trip(self, video_fake_client):
        resp = await litellm.amedia_generation(
            model="ai6700/grok-video-3",
            prompt="hi",
            type="video",
            api_key="sk",
            client=video_fake_client,
            poll_interval=0.001,
            price_markup=1.1,
        )
        encoded = orjson.dumps(resp.model_dump())
        decoded = orjson.loads(encoded)
        assert decoded["task_id"] == "1001"
        assert decoded["media_type"] == "video"
        assert decoded["data"][0]["url"] == "https://cdn/v.mp4"
        assert decoded["cost"] == pytest.approx(1.65)

    def test_url_property(self):
        resp = MediaResponse(
            task_id="1",
            model="m",
            media_type="video",
            data=[MediaAsset(url="https://x.mp4")],
        )
        assert resp.url == "https://x.mp4"
        assert resp.urls == ["https://x.mp4"]

    def test_url_property_empty(self):
        resp = MediaResponse(task_id="1", model="m", media_type="video", data=[])
        assert resp.url is None
        assert resp.urls == []
