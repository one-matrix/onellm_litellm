"""AI6700 image generation transformation.

Mirrors ``AI6700VideoConfig`` in shape: only owns conversions between
OpenAI-style image params and AI6700's ``POST /v1/media/generate`` body,
plus task-status → ``ImageResponse``. Submit+poll lives in
``AI6700Helper`` (common_utils.py).

We deliberately don't inherit from ``BaseImageGenerationConfig`` here for
the same reasons as the video config — that base assumes a single-shot
request/response and is over-prescriptive for a poll-until-done provider.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.llms.ai6700.common_utils import AI6700Error, AI6700Helper, AI6700TaskStatus
from litellm.types.utils import ImageObject, ImageResponse

# OpenAI "WIDTHxHEIGHT" → AI6700 size shorthand. Defaults are conservative;
# anything we cannot match passes through for the platform to validate.
_SIZE_TO_AI6700: Dict[str, str] = {
    "1024x1024": "1K",
    "1024x1792": "1K",
    "1792x1024": "1K",
    "2048x2048": "2K",
    "2048x1152": "2K",
    "1152x2048": "2K",
    "4096x4096": "4K",
}

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

# OpenAI-standard keys we explicitly understand (everything else under
# ``optional_params`` we pass through as AI6700 params keys).
_OPENAI_STANDARD_KEYS = {
    "n",
    "num_images",
    "size",
    "quality",
    "style",
    "response_format",
    "background",
    "moderation",
    "output_compression",
    "output_format",
    "seed",
    "user",
    "extra_headers",
    "extra_body",
    "parameters",
    "image_url",
    "image",
    "input_reference",
    "safety_tolerance",
    "prompt_upsampling",
    "raw",
    "image_prompt_strength",
    "model",
}

# AI6700 params known to take URL(s). Validated to reject base64/local paths.
_UPLOAD_PARAM_KEYS = {"images", "image_url", "first_frame", "last_frame", "reference"}


class AI6700ImageConfig:
    """Shape converter between litellm's image API and AI6700 /v1/media/generate."""

    # ------------------------------------------------------------------
    # Surface
    # ------------------------------------------------------------------

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "n",
            "num_images",
            "size",
            "image_url",
            "image",
            "input_reference",
            "parameters",
            "seed",
            "user",
            "extra_headers",
            "extra_body",
        ]

    def validate_environment(
        self,
        headers: Dict[str, str],
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        if not api_key:
            raise AI6700Error(
                status_code=401,
                message="ai6700 image: api_key is required (set LINGKE_API_KEY)",
            )
        merged = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if headers:
            user_has_auth = "Authorization" in headers
            merged.update(headers)
            if not user_has_auth:
                merged["Authorization"] = f"Bearer {api_key}"
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def strip_provider_prefix(model: str) -> str:
        return model.split("/", 1)[1] if "/" in model else model

    @staticmethod
    def _coerce_size(size: Any) -> Optional[str]:
        if not isinstance(size, str):
            return None
        size = size.strip()
        if not size:
            return None
        # Already AI6700-shaped ("1K"/"2K"/"4K"/"8K") → pass through.
        if re.match(r"^\d+[Kk]$", size):
            return size.upper()
        normalized = size.lower().replace(" ", "")
        return _SIZE_TO_AI6700.get(normalized, size)

    @staticmethod
    def _coerce_count(*candidates: Any) -> Optional[int]:
        for c in candidates:
            if c is None:
                continue
            try:
                n = int(c)
            except (TypeError, ValueError):
                continue
            if n > 0:
                return n
        return None

    @staticmethod
    def _validate_upload_value(param_name: str, value: Any) -> None:
        if isinstance(value, str):
            if not _URL_RE.match(value):
                raise AI6700Error(
                    status_code=400,
                    message=(
                        f"ai6700 image: param {param_name!r} must be an http(s):// URL "
                        f"(got {value[:80]!r}). AI6700 does not host files — upload "
                        f"to your own object storage and pass the public URL."
                    ),
                )
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                AI6700ImageConfig._validate_upload_value(f"{param_name}[{idx}]", item)
        else:
            raise AI6700Error(
                status_code=400,
                message=(
                    f"ai6700 image: param {param_name!r} must be a URL string or "
                    f"list of URL strings (got {type(value).__name__})."
                ),
            )

    # ------------------------------------------------------------------
    # Request shaping
    # ------------------------------------------------------------------

    def map_openai_params(
        self,
        optional_params: Dict[str, Any],
        model: str,
        drop_params: bool = False,
    ) -> Dict[str, Any]:
        """Map OpenAI-style image-gen optional params into AI6700 ``params``.

        Returns the *params* sub-dict only. The count (``n``/``num_images``)
        is **not** included here — pull it via ``extract_count()`` for the
        top-level body field.
        """
        params: Dict[str, Any] = {}

        # 1. caller-supplied parameters wins as base
        parameters = optional_params.get("parameters")
        if isinstance(parameters, dict):
            params.update(parameters)

        # 2. size → size (coerced)
        size = optional_params.get("size")
        if size:
            coerced = self._coerce_size(size)
            if coerced and "size" not in params:
                params["size"] = coerced

        # 3. image_url / image / input_reference → images
        for src in ("image_url", "image", "input_reference"):
            ref = optional_params.get(src)
            if ref is None:
                continue
            if "images" not in params:
                params["images"] = ref

        # 4. validate all upload-shaped params
        for key in _UPLOAD_PARAM_KEYS:
            if key in params and params[key] is not None:
                self._validate_upload_value(key, params[key])

        # 5. pass through unknown keys (e.g. aspect_ratio, seed-named differently)
        for key, value in optional_params.items():
            if key in _OPENAI_STANDARD_KEYS or value is None:
                continue
            if key not in params:
                params[key] = value

        return params

    def extract_count(self, optional_params: Dict[str, Any]) -> Optional[int]:
        return self._coerce_count(
            optional_params.get("n"),
            optional_params.get("num_images"),
        )

    def build_request_body(
        self,
        *,
        model: str,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        count: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not prompt:
            raise AI6700Error(
                status_code=400,
                message="ai6700 image: prompt is required",
            )
        body: Dict[str, Any] = {
            "model": self.strip_provider_prefix(model),
            "prompt": prompt,
        }
        if params:
            body["params"] = params
        if count is not None and count != 1:
            body["count"] = int(count)
        verbose_logger.debug("ai6700 image submit body: %s", body)
        return body

    # ------------------------------------------------------------------
    # Response shaping
    # ------------------------------------------------------------------

    def transform_task_to_image_response(
        self,
        *,
        status: AI6700TaskStatus,
        model: str,
        price_markup: float = 1.0,
        revised_prompt: Optional[str] = None,
        extra_hidden: Optional[Dict[str, Any]] = None,
    ) -> ImageResponse:
        """AI6700 terminal task-status → litellm ImageResponse.

        Caller is expected to have ensured ``status`` is the *finished*
        payload (``AI6700Helper._check_task_outcome`` does this).
        """
        urls = AI6700Helper.collect_result_urls(status)
        if not urls:
            raise AI6700Error(
                status_code=502,
                message=f"ai6700 image: task {status.get('task_id')} missing result_url",
            )

        provider_specific = {
            "ai6700_task_id": status.get("task_id"),
            "ai6700_result_type": status.get("result_type"),
            "ai6700_channel_group": status.get("channel_group"),
        }
        data = [
            ImageObject(
                url=url,
                revised_prompt=revised_prompt,
                provider_specific_fields=provider_specific,
            )
            for url in urls
        ]

        created_at = AI6700Helper.iso_to_unix(status.get("created_at")) or int(
            time.time()
        )
        bare_model = self.strip_provider_prefix(model)

        response = ImageResponse(
            created=created_at,
            data=data,
        )
        response._hidden_params = AI6700Helper.build_hidden_params(
            status=status,
            bare_model=bare_model,
            price_markup=price_markup,
            extra=extra_hidden,
        )
        return response
