from __future__ import annotations

from typing import Any

from src.common.logger import get_logger
from src.inference.inference_engine import InferenceEngine
from src.llm.base_client import BaseLLMClient, GenerationParams

logger = get_logger(__name__)


class LocalLLMClient(BaseLLMClient):
    def __init__(self, engine: InferenceEngine) -> None:
        """Initialize the client with the provided inference engine."""
        self._engine = engine

    async def generate(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> str:
        """Generate a completion from a single prompt."""
        logger.debug("Generating completion", extra={"provider": self.provider})
        kwargs = self._build_generation_kwargs(params)
        return await self._engine.generate(prompt, **kwargs)

    async def chat(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams | None = None,
    ) -> str:
        """Generate a completion from structured chat messages."""
        logger.debug("Generating chat completion", extra={"provider": self.provider})
        kwargs = self._build_generation_kwargs(params)
        return await self._engine.chat(messages, **kwargs)

    @property
    def provider(self) -> str:
        """Return the provider name for this client."""
        return "local"

    async def start(self) -> None:
        """Start the inference engine by loading the model."""
        logger.info("Starting local inference engine")
        await self._engine.start()

    async def stop(self) -> None:
        """Stop the inference engine and release resources."""
        logger.info("Stopping local inference engine")
        await self._engine.stop()

    @property
    def is_ready(self) -> bool:
        """Return True when the engine is loaded and ready."""
        return self._engine.is_ready

    def _build_generation_kwargs(
        self, params: GenerationParams | None
    ) -> dict[str, Any]:
        resolved_params = params or GenerationParams()
        return {
            "max_tokens": resolved_params.max_new_tokens,
            "temperature": resolved_params.temperature,
            "top_p": resolved_params.top_p,
        }
