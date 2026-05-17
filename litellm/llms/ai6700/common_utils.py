"""Shared helpers for the AI6700 (灵壳AI / lk888) media provider.

AI6700 exposes a single async API for image/video/audio/TTS/music generation:

    1. POST /v1/media/generate          -> {"data": {"任务id": <int>}}
    2. GET  /v1/skills/task-status?task_id=<int>
       -> poll until is_final=true, then read result_url / cost / result_type.

This module owns the submit + poll loop and the error class. Per-modality
transformation modules call ``AI6700Helper.submit_and_poll`` and shape the
final task payload into the appropriate litellm response object.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

AI6700_DEFAULT_API_BASE = "https://api.lk888.ai/api"
AI6700_DEFAULT_POLL_INTERVAL = 5.0
AI6700_DEFAULT_TIMEOUT = 600.0

_SUBMIT_PATH = "/v1/media/generate"
_STATUS_PATH = "/v1/skills/task-status"

_TASK_ID_KEYS = ("任务id", "task_id", "taskId")


class AI6700Error(BaseLLMException):
    """Raised for any AI6700-specific failure (HTTP, polling timeout, task failure)."""


class AI6700TaskStatus(TypedDict, total=False):
    task_id: int
    model: str
    status: str
    status_group: str
    progress: str
    is_final: bool
    result_url: str
    result_type: str
    input_files: List[str]
    cost: float
    channel_group: str
    error: str
    created_at: str
    completed_at: str
    duration_seconds: float


class AI6700Helper:
    """Stateless helper for AI6700 submit + poll flow.

    All public methods accept an explicit httpx client so callers can reuse
    pooled connections (litellm's ``get_async_httpx_client``) and so tests
    can inject a mock transport.
    """

    @staticmethod
    def normalize_base(api_base: Optional[str]) -> str:
        base = api_base or AI6700_DEFAULT_API_BASE
        return base.rstrip("/")

    @staticmethod
    def auth_headers(
        api_key: str, extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        if not api_key:
            raise AI6700Error(
                status_code=401,
                message="ai6700: missing api_key (set LINGKE_API_KEY or pass api_key=...)",
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def build_payload(
        *,
        model: str,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        count: Optional[int] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "prompt": prompt}
        if params:
            body["params"] = params
        if count is not None and count != 1:
            body["count"] = int(count)
        return body

    @staticmethod
    def _extract_task_id(submit_json: Dict[str, Any]) -> int:
        data = submit_json.get("data") if isinstance(submit_json, dict) else None
        if not isinstance(data, dict):
            raise AI6700Error(
                status_code=502,
                message=f"ai6700: malformed submit response (no data field): {submit_json!r}",
            )
        for key in _TASK_ID_KEYS:
            if key in data and data[key] is not None:
                try:
                    return int(data[key])
                except (TypeError, ValueError) as exc:
                    raise AI6700Error(
                        status_code=502,
                        message=f"ai6700: non-integer task id under {key!r}: {data[key]!r}",
                    ) from exc
        raise AI6700Error(
            status_code=502,
            message=f"ai6700: submit response missing task id (looked for {_TASK_ID_KEYS}): {data!r}",
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response, action: str) -> None:
        if response.status_code >= 400:
            raise AI6700Error(
                status_code=response.status_code,
                message=f"ai6700 {action} failed: HTTP {response.status_code} {response.text}",
                headers=dict(response.headers),
                response=response,
            )

    @staticmethod
    def _coerce_status(payload: Dict[str, Any]) -> AI6700TaskStatus:
        # /v1/skills/task-status returns the status object at top level,
        # /v1/media/status wraps it under {"code":200,"data":{...}}.
        if (
            isinstance(payload, dict)
            and "data" in payload
            and isinstance(payload["data"], dict)
        ):
            data = payload["data"]
        else:
            data = payload
        return data  # type: ignore[return-value]

    @staticmethod
    def _is_terminal(status: AI6700TaskStatus) -> bool:
        if status.get("is_final"):
            return True
        # Belt-and-suspenders: some early responses may omit is_final.
        return status.get("status_group") in {"已完成", "失败"}

    @staticmethod
    def _check_task_outcome(status: AI6700TaskStatus) -> AI6700TaskStatus:
        if status.get("status_group") == "失败" or (
            status.get("error") and not status.get("result_url")
        ):
            raise AI6700Error(
                status_code=502,
                message=(
                    f"ai6700 task {status.get('task_id')} failed: "
                    f"{status.get('error') or status.get('status')}"
                ),
            )
        if not status.get("result_url"):
            raise AI6700Error(
                status_code=502,
                message=f"ai6700 task {status.get('task_id')} finished without result_url: {status!r}",
            )
        return status

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    @staticmethod
    async def asubmit(
        client: AsyncHTTPHandler,
        *,
        api_base: Optional[str],
        api_key: str,
        body: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> int:
        url = AI6700Helper.normalize_base(api_base) + _SUBMIT_PATH
        headers = AI6700Helper.auth_headers(api_key, extra_headers)
        response = await client.post(url=url, headers=headers, json=body)
        AI6700Helper._raise_for_status(response, action="submit")
        return AI6700Helper._extract_task_id(response.json())

    @staticmethod
    async def aget_status(
        client: AsyncHTTPHandler,
        *,
        api_base: Optional[str],
        api_key: str,
        task_id: int,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> AI6700TaskStatus:
        """Single GET against /v1/skills/task-status — no polling loop.

        Use this for "tell me the current state once" — e.g. a manual
        status-check endpoint. For the wait-until-done flow use ``apoll``.
        """
        url = AI6700Helper.normalize_base(api_base) + _STATUS_PATH
        headers = AI6700Helper.auth_headers(api_key, extra_headers)
        response = await client.get(
            url=url, params={"task_id": task_id}, headers=headers
        )
        AI6700Helper._raise_for_status(response, action="status")
        return AI6700Helper._coerce_status(response.json())

    @staticmethod
    async def apoll(
        client: AsyncHTTPHandler,
        *,
        api_base: Optional[str],
        api_key: str,
        task_id: int,
        timeout: float = AI6700_DEFAULT_TIMEOUT,
        poll_interval: float = AI6700_DEFAULT_POLL_INTERVAL,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> AI6700TaskStatus:
        url = AI6700Helper.normalize_base(api_base) + _STATUS_PATH
        headers = AI6700Helper.auth_headers(api_key, extra_headers)
        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            attempt += 1
            response = await client.get(
                url=url, params={"task_id": task_id}, headers=headers
            )
            AI6700Helper._raise_for_status(response, action="status")
            status = AI6700Helper._coerce_status(response.json())
            verbose_logger.debug(
                "ai6700 poll attempt=%s task_id=%s status=%s progress=%s",
                attempt,
                task_id,
                status.get("status"),
                status.get("progress"),
            )
            if AI6700Helper._is_terminal(status):
                return AI6700Helper._check_task_outcome(status)
            if time.monotonic() >= deadline:
                raise AI6700Error(
                    status_code=504,
                    message=(
                        f"ai6700 polling timed out after {timeout:.0f}s "
                        f"(task_id={task_id}, last_status={status.get('status')})"
                    ),
                )
            await asyncio.sleep(poll_interval)

    @staticmethod
    async def asubmit_and_poll(
        client: AsyncHTTPHandler,
        *,
        api_base: Optional[str],
        api_key: str,
        body: Dict[str, Any],
        timeout: float = AI6700_DEFAULT_TIMEOUT,
        poll_interval: float = AI6700_DEFAULT_POLL_INTERVAL,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> AI6700TaskStatus:
        task_id = await AI6700Helper.asubmit(
            client,
            api_base=api_base,
            api_key=api_key,
            body=body,
            extra_headers=extra_headers,
        )
        return await AI6700Helper.apoll(
            client,
            api_base=api_base,
            api_key=api_key,
            task_id=task_id,
            timeout=timeout,
            poll_interval=poll_interval,
            extra_headers=extra_headers,
        )

    # ------------------------------------------------------------------
    # Sync API (used by litellm's sync image/video entry points)
    # ------------------------------------------------------------------

    @staticmethod
    def submit(
        client: HTTPHandler,
        *,
        api_base: Optional[str],
        api_key: str,
        body: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> int:
        url = AI6700Helper.normalize_base(api_base) + _SUBMIT_PATH
        headers = AI6700Helper.auth_headers(api_key, extra_headers)
        response = client.post(url=url, headers=headers, json=body)
        AI6700Helper._raise_for_status(response, action="submit")
        return AI6700Helper._extract_task_id(response.json())

    @staticmethod
    def poll(
        client: HTTPHandler,
        *,
        api_base: Optional[str],
        api_key: str,
        task_id: int,
        timeout: float = AI6700_DEFAULT_TIMEOUT,
        poll_interval: float = AI6700_DEFAULT_POLL_INTERVAL,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> AI6700TaskStatus:
        url = AI6700Helper.normalize_base(api_base) + _STATUS_PATH
        headers = AI6700Helper.auth_headers(api_key, extra_headers)
        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            attempt += 1
            response = client.get(url=url, params={"task_id": task_id}, headers=headers)
            AI6700Helper._raise_for_status(response, action="status")
            status = AI6700Helper._coerce_status(response.json())
            verbose_logger.debug(
                "ai6700 sync poll attempt=%s task_id=%s status=%s progress=%s",
                attempt,
                task_id,
                status.get("status"),
                status.get("progress"),
            )
            if AI6700Helper._is_terminal(status):
                return AI6700Helper._check_task_outcome(status)
            if time.monotonic() >= deadline:
                raise AI6700Error(
                    status_code=504,
                    message=(
                        f"ai6700 polling timed out after {timeout:.0f}s "
                        f"(task_id={task_id}, last_status={status.get('status')})"
                    ),
                )
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Cost / hidden_params shared shape
    # ------------------------------------------------------------------

    @staticmethod
    def iso_to_unix(ts: Optional[str]) -> Optional[int]:
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def collect_result_urls(status: AI6700TaskStatus) -> List[str]:
        """Extract one or more result URLs from a terminal task.

        AI6700's documented response uses singular ``result_url``. Some
        channels (or future shapes) may also emit ``result_urls`` as an
        array — accept both and de-duplicate.
        """
        urls: List[str] = []
        primary = status.get("result_url")
        if isinstance(primary, str) and primary:
            urls.append(primary)
        extras = status.get("result_urls")  # type: ignore[index]
        if isinstance(extras, (list, tuple)):
            for u in extras:
                if isinstance(u, str) and u and u not in urls:
                    urls.append(u)
        return urls

    @staticmethod
    def build_hidden_params(
        *,
        status: AI6700TaskStatus,
        bare_model: str,
        price_markup: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the canonical ``_hidden_params`` dict for an AI6700 response.

        Critically populates BOTH:
          - ``additional_headers["llm_provider-x-litellm-response-cost"]``
            so ``litellm.cost_calculator.response_cost_calculator`` short-circuits
            and uses our pre-computed cost instead of recomputing.
          - ``response_cost`` so proxy code that reads
            ``hidden_params.get("response_cost")`` directly (spend logs,
            usage endpoints, OTEL hooks) finds the same number.

        Plus all the ``ai6700_*`` provenance fields for debugging /
        reconciliation against the upstream account.
        """
        urls = AI6700Helper.collect_result_urls(status)
        raw_cost = float(status.get("cost") or 0)
        billed_cost = raw_cost * (price_markup or 1.0)

        hidden: Dict[str, Any] = {
            "ai6700_task_id": status.get("task_id"),
            "ai6700_result_url": urls[0] if urls else None,
            "ai6700_result_urls": urls,
            "ai6700_result_type": status.get("result_type"),
            "ai6700_channel_group": status.get("channel_group"),
            "ai6700_status": status.get("status"),
            "ai6700_status_group": status.get("status_group"),
            "ai6700_duration_seconds": status.get("duration_seconds"),
            "ai6700_raw_cost": raw_cost,
            "ai6700_price_markup": price_markup,
            "response_cost": billed_cost,
            "additional_headers": {
                "llm_provider-x-litellm-response-cost": billed_cost,
            },
            "model": bare_model,
            "custom_llm_provider": "ai6700",
        }
        if extra:
            hidden.update(extra)
        return hidden

    @staticmethod
    def submit_and_poll(
        client: HTTPHandler,
        *,
        api_base: Optional[str],
        api_key: str,
        body: Dict[str, Any],
        timeout: float = AI6700_DEFAULT_TIMEOUT,
        poll_interval: float = AI6700_DEFAULT_POLL_INTERVAL,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> AI6700TaskStatus:
        task_id = AI6700Helper.submit(
            client,
            api_base=api_base,
            api_key=api_key,
            body=body,
            extra_headers=extra_headers,
        )
        return AI6700Helper.poll(
            client,
            api_base=api_base,
            api_key=api_key,
            task_id=task_id,
            timeout=timeout,
            poll_interval=poll_interval,
            extra_headers=extra_headers,
        )
