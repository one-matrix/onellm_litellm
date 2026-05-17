"""Uniform response type for poll-then-URL media generation providers.

Used by ``litellm.amedia_generation`` to wrap the result of any async-poll
media provider (currently AI6700, extensible to other gateways) without
forcing callers to know whether the result is video / image / audio / TTS /
music. For modality-specific shapes (``VideoObject`` / ``ImageResponse`` /
``AI6700AudioResponse``), use the corresponding provider transformation
directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class MediaAsset(BaseModel):
    """One generated artifact (URL + minimal metadata).

    AI6700 typically returns a single asset per task; provider-specific
    fields (channel_group, original task id, etc.) live on the parent
    ``MediaResponse``. Multiple assets are rare but supported (e.g. an
    image batch where the provider returns ``result_urls=[...]``).
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    url: str
    result_type: Optional[str] = None
    revised_prompt: Optional[str] = None


class MediaResponse(BaseModel):
    """Uniform response for ``litellm.amedia_generation``.

    Fields:
      - ``task_id``: provider-side task identifier (int → string normalized)
      - ``model``: bare model name (provider prefix stripped)
      - ``media_type``: ``image`` / ``video`` / ``audio`` / ``tts`` / ``music``
      - ``data``: list of ``MediaAsset`` (typically one)
      - ``cost``: billed cost in algorithm-points (raw × price_markup)
      - ``raw_cost``: cost as reported by the provider (no markup)
      - ``channel_group``: which provider channel actually served the task
      - ``duration_seconds``: end-to-end submit→complete latency
      - ``created`` / ``completed_at``: unix timestamps
      - ``_hidden_params``: includes ``response_cost`` for the standard
        litellm cost-logging chain to pick up.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    object: Literal["media.task"] = "media.task"
    task_id: str
    model: str
    media_type: str
    status: Literal["completed"] = "completed"
    data: List[MediaAsset] = []
    cost: float = 0.0
    raw_cost: float = 0.0
    price_markup: float = 1.0
    channel_group: Optional[str] = None
    duration_seconds: Optional[float] = None
    created: Optional[int] = None
    completed_at: Optional[int] = None
    provider: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    _hidden_params: Dict[str, Any] = {}

    # ----- convenience -----

    @property
    def url(self) -> Optional[str]:
        """First asset's URL — convenience for single-asset tasks."""
        return self.data[0].url if self.data else None

    @property
    def urls(self) -> List[str]:
        return [a.url for a in self.data]

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def json(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()
