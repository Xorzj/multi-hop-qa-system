from __future__ import annotations

import importlib
from typing import Any

from src.common.exceptions import LLMError
from src.common.logger import get_logger
from src.llm.base_client import BaseLLMClient, GenerationParams, post_with_retry

logger = get_logger(__name__)


class ZhipuClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "glm-4-flash",
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    ) -> None:
        """Set credentials and defaults for the Zhipu client."""
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def generate(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> str:
        """Generate a completion from a single prompt."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, params)

    async def chat(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams | None = None,
    ) -> str:
        """Generate a completion from structured chat messages."""
        logger.debug("Calling Zhipu chat completion", extra={"provider": self.provider})
        return await self._call_api(messages, params)

    @property
    def provider(self) -> str:
        """Return the provider name for this client."""
        return "zhipu"

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams | None,
    ) -> str:
        """Call the Zhipu chat completions API and return the content."""
        httpx_module = self._load_httpx()
        resolved_params = params or GenerationParams()
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": resolved_params.max_new_tokens,
            "temperature": resolved_params.temperature,
            "top_p": resolved_params.top_p,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"

        try:
            response = await post_with_retry(
                httpx_module,
                url,
                json=payload,
                headers=headers,
                timeout=60.0,
                provider_name="Zhipu",
            )
        except httpx_module.HTTPStatusError as exc:
            detail = exc.response.text
            logger.error(
                "Zhipu API returned an error",
                extra={"status_code": exc.response.status_code, "detail": detail},
            )
            raise LLMError(
                f"Zhipu API error: {exc.response.status_code} {detail}"
            ) from exc
        except httpx_module.HTTPError as exc:
            logger.error("Zhipu API request failed", extra={"error": str(exc)})
            raise LLMError(f"Zhipu API request failed: {exc}") from exc

        try:
            data = response.json()
            message = data["choices"][0]["message"]
            raw_content = message.get("content")
            content = raw_content if isinstance(raw_content, str) else ""
            # Thinking models (e.g. glm-4.7-flash) put output in reasoning_content
            if not content.strip():
                fallback = message.get("reasoning_content")
                content = fallback if isinstance(fallback, str) else ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error(
                "Unexpected Zhipu API response format",
                extra={"response": response.text},
            )
            raise LLMError(f"Unexpected Zhipu API response: {response.text}") from exc

        if not isinstance(content, str):
            raise LLMError("Zhipu API response content is not a string")

        return content

    def _load_httpx(self) -> Any:
        try:
            return importlib.import_module("httpx")
        except ModuleNotFoundError as exc:
            raise LLMError("httpx is required to call Zhipu API") from exc
