"""Base LLM client abstractions, generation parameters, and HTTP retry helpers."""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.common.logger import get_logger

logger = get_logger(__name__)

# ──────────────────── 重试配置 ────────────────────

MAX_RETRIES = 5
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 529})


@dataclass
class GenerationParams:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    system_message: str | None = None
    enable_thinking: bool = True


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> str:
        raise NotImplementedError("TODO: implement")

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams | None = None,
    ) -> str:
        raise NotImplementedError("TODO: implement")

    @property
    @abstractmethod
    def provider(self) -> str:
        raise NotImplementedError("TODO: implement")

    async def start(self) -> None:
        """Initialize the client (e.g. load model weights)."""

    async def stop(self) -> None:
        """Release resources held by the client."""


async def post_with_retry(
    httpx_module: Any,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_S,
    provider_name: str = "API",
) -> Any:
    """HTTP POST with exponential backoff on retryable errors (429, 5xx).

    Returns the httpx.Response on success.
    Re-raises the original exception after all retries are exhausted
    or on non-retryable errors.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            async with httpx_module.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=json, headers=headers)
            response.raise_for_status()
            return response
        except httpx_module.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                wait = _compute_backoff(
                    attempt,
                    initial_backoff,
                    exc.response.headers.get("retry-after"),
                )
                logger.warning(
                    "%s %d, retry %d/%d in %.1fs",
                    provider_name,
                    status,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                last_exc = exc
                await asyncio.sleep(wait)
                continue
            raise
        except (httpx_module.ConnectError, httpx_module.ReadTimeout) as exc:
            if attempt < max_retries:
                wait = _compute_backoff(attempt, initial_backoff)
                logger.warning(
                    "%s connection error (%s), retry %d/%d in %.1fs",
                    provider_name,
                    type(exc).__name__,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                last_exc = exc
                await asyncio.sleep(wait)
                continue
            raise

    # Should not reach here, but satisfy type checker
    assert last_exc is not None  # noqa: S101
    raise last_exc


def _compute_backoff(
    attempt: int,
    initial: float,
    retry_after_header: str | None = None,
) -> float:
    """Compute wait time with exponential backoff + jitter."""
    if retry_after_header is not None:
        try:
            return max(float(retry_after_header), 0.5)
        except ValueError:
            pass
    backoff = min(initial * (2**attempt), MAX_BACKOFF_S)
    jitter = random.uniform(0, min(backoff * 0.5, 2.0))  # noqa: S311
    return backoff + jitter
