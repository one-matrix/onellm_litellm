"""Tests for AI6700 common utilities (submit + poll + hidden_params)."""

from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.ai6700 import (
    AI6700_DEFAULT_API_BASE,
    AI6700Error,
    AI6700Helper,
)


def _fake_response(status_code: int, payload: dict) -> MagicMock:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status_code
    r.headers = {}
    r.text = str(payload)
    r.json.return_value = payload
    return r


class _FakeClient:
    """Async HTTP client double for AI6700 helper tests."""

    def __init__(self, submit_payload=None, status_payloads=None):
        self.submit_payload = submit_payload
        self.status_payloads = list(status_payloads or [])
        self.posts = []
        self.gets = []

    async def post(self, url=None, headers=None, json=None):
        self.posts.append({"url": url, "headers": headers, "json": json})
        return _fake_response(200, self.submit_payload or {})

    async def get(self, url=None, params=None, headers=None):
        self.gets.append({"url": url, "params": params, "headers": headers})
        return _fake_response(200, self.status_payloads.pop(0))

    async def close(self):
        pass


class TestNormalizeAndAuth:
    def test_normalize_base_strips_trailing_slash(self):
        assert AI6700Helper.normalize_base("https://x/api/") == "https://x/api"

    def test_normalize_base_default(self):
        assert AI6700Helper.normalize_base(None) == AI6700_DEFAULT_API_BASE

    def test_auth_headers(self):
        h = AI6700Helper.auth_headers("sk-test")
        assert h["Authorization"] == "Bearer sk-test"
        assert h["Content-Type"] == "application/json"

    def test_auth_headers_missing_key_raises(self):
        with pytest.raises(AI6700Error) as exc:
            AI6700Helper.auth_headers("")
        assert exc.value.status_code == 401

    def test_auth_headers_with_extra(self):
        h = AI6700Helper.auth_headers("sk", extra={"X-Trace": "abc"})
        assert h["X-Trace"] == "abc"


class TestBuildPayload:
    def test_minimal(self):
        body = AI6700Helper.build_payload(model="grok-video-3", prompt="hi")
        assert body == {"model": "grok-video-3", "prompt": "hi"}

    def test_with_params(self):
        body = AI6700Helper.build_payload(
            model="m", prompt="p", params={"resolution": "720p"}
        )
        assert body["params"] == {"resolution": "720p"}

    def test_count_omits_default(self):
        body = AI6700Helper.build_payload(model="m", prompt="p", count=1)
        assert "count" not in body

    def test_count_included_when_not_one(self):
        body = AI6700Helper.build_payload(model="m", prompt="p", count=3)
        assert body["count"] == 3


class TestExtractTaskId:
    def test_chinese_key(self):
        assert AI6700Helper._extract_task_id({"data": {"任务id": "1234"}}) == 1234

    def test_english_key(self):
        assert AI6700Helper._extract_task_id({"data": {"task_id": 9}}) == 9

    def test_camelcase_key(self):
        assert AI6700Helper._extract_task_id({"data": {"taskId": 5}}) == 5

    def test_missing(self):
        with pytest.raises(AI6700Error) as exc:
            AI6700Helper._extract_task_id({"data": {}})
        assert exc.value.status_code == 502

    def test_no_data_field(self):
        with pytest.raises(AI6700Error):
            AI6700Helper._extract_task_id({"msg": "ok"})

    def test_non_integer_value(self):
        with pytest.raises(AI6700Error):
            AI6700Helper._extract_task_id({"data": {"任务id": "abc"}})


class TestTerminalDetection:
    def test_is_final_true(self):
        assert AI6700Helper._is_terminal({"is_final": True})

    def test_status_group_completed(self):
        assert AI6700Helper._is_terminal({"is_final": False, "status_group": "已完成"})

    def test_status_group_failed(self):
        assert AI6700Helper._is_terminal({"is_final": False, "status_group": "失败"})

    def test_in_progress(self):
        assert not AI6700Helper._is_terminal(
            {"is_final": False, "status_group": "处理中"}
        )


class TestCheckTaskOutcome:
    def test_failure_raises(self):
        with pytest.raises(AI6700Error) as exc:
            AI6700Helper._check_task_outcome(
                {"task_id": 1, "status_group": "失败", "error": "boom"}
            )
        assert exc.value.status_code == 502

    def test_missing_result_url_raises(self):
        with pytest.raises(AI6700Error):
            AI6700Helper._check_task_outcome({"task_id": 1, "status_group": "已完成"})

    def test_success_passes_through(self):
        status = {
            "task_id": 1,
            "status_group": "已完成",
            "result_url": "https://x",
        }
        assert AI6700Helper._check_task_outcome(status) is status

    def test_error_without_result_raises(self):
        with pytest.raises(AI6700Error):
            AI6700Helper._check_task_outcome({"task_id": 1, "error": "bad"})


class TestCollectResultUrls:
    def test_single(self):
        urls = AI6700Helper.collect_result_urls({"result_url": "https://a"})
        assert urls == ["https://a"]

    def test_multi_with_extras(self):
        urls = AI6700Helper.collect_result_urls(
            {"result_url": "https://a", "result_urls": ["https://b", "https://c"]}
        )
        assert urls == ["https://a", "https://b", "https://c"]

    def test_dedupe(self):
        urls = AI6700Helper.collect_result_urls(
            {"result_url": "https://a", "result_urls": ["https://a", "https://b"]}
        )
        assert urls == ["https://a", "https://b"]

    def test_empty(self):
        assert AI6700Helper.collect_result_urls({}) == []


class TestIsoToUnix:
    def test_z_suffix(self):
        assert AI6700Helper.iso_to_unix("2026-03-17T10:00:00Z") == 1773741600

    def test_naive_treated_as_utc(self):
        assert AI6700Helper.iso_to_unix("2026-03-17T10:00:00") == 1773741600

    def test_none(self):
        assert AI6700Helper.iso_to_unix(None) is None

    def test_invalid(self):
        assert AI6700Helper.iso_to_unix("not-a-date") is None


class TestBuildHiddenParams:
    """The key wiring: cost into both response_cost AND additional_headers."""

    STATUS = {
        "task_id": 999,
        "status": "生成完成",
        "status_group": "已完成",
        "is_final": True,
        "result_url": "https://cdn/x.mp4",
        "result_type": "video",
        "cost": 1.5,
        "channel_group": "标准",
        "duration_seconds": 42,
    }

    def test_response_cost_with_markup(self):
        hp = AI6700Helper.build_hidden_params(
            status=self.STATUS, bare_model="m", price_markup=1.1
        )
        assert hp["response_cost"] == pytest.approx(1.65)

    def test_additional_headers_match_response_cost(self):
        hp = AI6700Helper.build_hidden_params(
            status=self.STATUS, bare_model="m", price_markup=1.1
        )
        assert hp["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ] == pytest.approx(1.65)

    def test_raw_cost_preserved(self):
        hp = AI6700Helper.build_hidden_params(
            status=self.STATUS, bare_model="m", price_markup=1.1
        )
        assert hp["ai6700_raw_cost"] == 1.5
        assert hp["ai6700_price_markup"] == 1.1

    def test_provenance_fields(self):
        hp = AI6700Helper.build_hidden_params(
            status=self.STATUS, bare_model="m", price_markup=1.0
        )
        assert hp["ai6700_task_id"] == 999
        assert hp["ai6700_result_url"] == "https://cdn/x.mp4"
        assert hp["ai6700_result_urls"] == ["https://cdn/x.mp4"]
        assert hp["ai6700_channel_group"] == "标准"
        assert hp["ai6700_status"] == "生成完成"
        assert hp["model"] == "m"
        assert hp["custom_llm_provider"] == "ai6700"

    def test_extra_merge(self):
        hp = AI6700Helper.build_hidden_params(
            status=self.STATUS,
            bare_model="m",
            price_markup=1.0,
            extra={"media_type": "video", "ai6700_status": "overridden"},
        )
        assert hp["media_type"] == "video"
        # extra overrides the default fields
        assert hp["ai6700_status"] == "overridden"

    def test_zero_cost(self):
        hp = AI6700Helper.build_hidden_params(
            status={"task_id": 1, "result_url": "https://x"},
            bare_model="m",
            price_markup=1.5,
        )
        assert hp["response_cost"] == 0.0
        assert hp["ai6700_raw_cost"] == 0.0


class TestAsubmit:
    async def test_extracts_task_id(self):
        client = _FakeClient(
            submit_payload={"msg": "ok", "code": 200, "data": {"任务id": 7777}}
        )
        tid = await AI6700Helper.asubmit(
            client,
            api_base="https://x",
            api_key="sk",
            body={"model": "m", "prompt": "p"},
        )
        assert tid == 7777
        assert client.posts[0]["url"].endswith("/v1/media/generate")
        assert client.posts[0]["headers"]["Authorization"] == "Bearer sk"
        assert client.posts[0]["json"] == {"model": "m", "prompt": "p"}


class TestApoll:
    async def test_loops_until_terminal(self):
        client = _FakeClient(
            status_payloads=[
                {"task_id": 5, "is_final": False, "progress": "10%"},
                {"task_id": 5, "is_final": False, "progress": "60%"},
                {
                    "task_id": 5,
                    "is_final": True,
                    "status_group": "已完成",
                    "result_url": "https://x",
                    "cost": 1.0,
                },
            ]
        )
        status = await AI6700Helper.apoll(
            client,
            api_base="https://x",
            api_key="sk",
            task_id=5,
            poll_interval=0.001,
            timeout=30,
        )
        assert status["result_url"] == "https://x"
        assert len(client.gets) == 3

    async def test_timeout_raises(self):
        client = _FakeClient(
            status_payloads=[{"task_id": 5, "is_final": False, "progress": "10%"}] * 5
        )
        with pytest.raises(AI6700Error) as exc:
            await AI6700Helper.apoll(
                client,
                api_base="https://x",
                api_key="sk",
                task_id=5,
                poll_interval=0.05,
                timeout=0.01,
            )
        assert exc.value.status_code == 504

    async def test_failure_status_raises(self):
        client = _FakeClient(
            status_payloads=[
                {
                    "task_id": 5,
                    "is_final": True,
                    "status_group": "失败",
                    "error": "model crashed",
                }
            ]
        )
        with pytest.raises(AI6700Error) as exc:
            await AI6700Helper.apoll(
                client,
                api_base="https://x",
                api_key="sk",
                task_id=5,
                poll_interval=0.001,
            )
        assert "model crashed" in exc.value.message


class TestAgetStatus:
    """Single-shot status query — returns even if not terminal."""

    async def test_returns_in_progress(self):
        client = _FakeClient(
            status_payloads=[{"task_id": 1, "is_final": False, "progress": "30%"}]
        )
        status = await AI6700Helper.aget_status(
            client, api_base="https://x", api_key="sk", task_id=1
        )
        assert status["progress"] == "30%"
        assert status["is_final"] is False
        # Did not loop
        assert len(client.gets) == 1

    async def test_returns_terminal(self):
        client = _FakeClient(
            status_payloads=[
                {"task_id": 1, "is_final": True, "result_url": "https://x"}
            ]
        )
        status = await AI6700Helper.aget_status(
            client, api_base="https://x", api_key="sk", task_id=1
        )
        assert status["is_final"] is True


class TestAsubmitAndPoll:
    async def test_end_to_end(self):
        client = _FakeClient(
            submit_payload={"data": {"任务id": 42}},
            status_payloads=[
                {"task_id": 42, "is_final": False, "progress": "50%"},
                {
                    "task_id": 42,
                    "is_final": True,
                    "status_group": "已完成",
                    "result_url": "https://cdn/v.mp4",
                    "cost": 2.0,
                },
            ],
        )
        status = await AI6700Helper.asubmit_and_poll(
            client,
            api_base="https://x",
            api_key="sk",
            body={"model": "m", "prompt": "p"},
            poll_interval=0.001,
            timeout=10,
        )
        assert status["cost"] == 2.0
        assert status["result_url"] == "https://cdn/v.mp4"
        assert len(client.posts) == 1
        assert len(client.gets) == 2


class TestErrorResponses:
    async def test_http_error_propagates(self):
        client = _FakeClient(submit_payload={"error": "bad"})
        resp = _fake_response(500, {"error": "bad"})

        async def bad_post(**_k):
            return resp

        client.post = bad_post  # type: ignore[assignment]
        with pytest.raises(AI6700Error) as exc:
            await AI6700Helper.asubmit(
                client,
                api_base="https://x",
                api_key="sk",
                body={"model": "m", "prompt": "p"},
            )
        assert exc.value.status_code == 500
