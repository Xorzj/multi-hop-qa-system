"""OpenAI-compatible API client."""

from __future__ import annotations

import importlib
import os
from typing import Any

from src.common.config import LLMConfig
from src.common.exceptions import ConfigError, LLMError
from src.common.logger import get_logger
from src.llm.base_client import BaseLLMClient, GenerationParams, post_with_retry

logger = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 180.0


class OpenAIClient(BaseLLMClient):
    """LLM client that talks to any OpenAI-compatible chat completions API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._model = config.model_path or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._base_url = config.base_url.rstrip("/") if config.base_url else ""
        self._api_key = config.api_key
        self._timeout_seconds = self._parse_timeout()

    # ── public interface ───────────────────────────────────

    async def generate(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, params)

    async def chat(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams | None = None,
    ) -> str:
        logger.debug(
            "Calling OpenAI chat completion",
            extra={"provider": self.provider, "model": self._model},
        )
        return await self._call_api(messages, params)

    @property
    def provider(self) -> str:
        return "openai"

    # ── internal ───────────────────────────────────────────

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams | None,
    ) -> str:
        httpx_module = self._load_httpx()
        endpoint, api_key = self._resolve_endpoint_and_key()
        resolved_params = params or GenerationParams()
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": resolved_params.max_new_tokens,
            "temperature": resolved_params.temperature,
            "top_p": resolved_params.top_p,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await post_with_retry(
                httpx_module,
                endpoint,
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
                provider_name="OpenAI",
            )
        except httpx_module.HTTPStatusError as exc:
            detail = exc.response.text
            logger.error(
                "OpenAI API returned an error",
                extra={"status_code": exc.response.status_code, "detail": detail},
            )
            raise LLMError(
                f"OpenAI API error: {exc.response.status_code} {detail}"
            ) from exc
        except httpx_module.HTTPError as exc:
            logger.error("OpenAI API request failed", extra={"error": str(exc)})
            raise LLMError(f"OpenAI API request failed: {exc}") from exc

        try:
            data = response.json()
            message = data["choices"][0]["message"]
            raw_content = message.get("content")
            content = raw_content if isinstance(raw_content, str) else ""
            # Thinking models put output in reasoning_content
            if not content.strip():
                fallback = message.get("reasoning_content")
                content = fallback if isinstance(fallback, str) else ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error(
                "Unexpected OpenAI API response format",
                extra={"response": response.text},
            )
            raise LLMError(f"Unexpected OpenAI API response: {response.text}") from exc

        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        if not isinstance(content, str):
            raise LLMError("OpenAI API response content is not a string")
        return content

    def _resolve_endpoint_and_key(self) -> tuple[str, str]:
        endpoint = self._base_url
        api_key = self._api_key
        if not endpoint:
            raise ConfigError(
                "llm.base_url is required in config.toml "
                "(or set OPENAI_BASE_URL environment variable)"
            )
        if not api_key:
            raise ConfigError(
                "llm.api_key is required in config.toml "
                "(or set OPENAI_API_KEY environment variable)"
            )
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"
        return endpoint, api_key

    @staticmethod
    def _parse_timeout() -> float:
        raw = os.getenv("OPENAI_TIMEOUT_SECONDS", "")
        if not raw:
            return _DEFAULT_TIMEOUT_SECONDS
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "Invalid OPENAI_TIMEOUT_SECONDS=%r, using default %.0f",
                raw,
                _DEFAULT_TIMEOUT_SECONDS,
            )
            return _DEFAULT_TIMEOUT_SECONDS
        if value <= 0:
            logger.warning(
                "OPENAI_TIMEOUT_SECONDS must be positive, using default %.0f",
                _DEFAULT_TIMEOUT_SECONDS,
            )
            return _DEFAULT_TIMEOUT_SECONDS
        return value

    def _load_httpx(self) -> Any:
        try:
            return importlib.import_module("httpx")
        except ModuleNotFoundError as exc:
            raise LLMError("httpx is required to call OpenAI API") from exc
