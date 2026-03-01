from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.common.logger import get_logger
from src.inference.model_loader import ModelConfig, ModelLoader

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    model_config: ModelConfig
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    batch_size: int = 4


class InferenceEngine:
    def __init__(self, config: InferenceConfig) -> None:
        """Initialize inference engine without loading the model."""
        self._config = config
        self._model_loader = ModelLoader(config.model_config)

    @property
    def is_ready(self) -> bool:
        """Return True when the model is loaded and ready."""
        return self._model_loader.is_loaded

    async def start(self) -> None:
        """Load the model and tokenizer asynchronously."""
        await asyncio.to_thread(self._model_loader.load)

    async def stop(self) -> None:
        """Unload the model and release resources asynchronously."""
        await asyncio.to_thread(self._model_loader.unload)

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """Generate text for a single prompt asynchronously."""
        max_new_tokens = (
            max_tokens if max_tokens is not None else self._config.default_max_tokens
        )
        resolved_temperature = (
            temperature if temperature is not None else self._config.default_temperature
        )
        resolved_top_p = top_p if top_p is not None else self._config.default_top_p
        return await asyncio.to_thread(
            self._model_loader.generate,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=resolved_temperature,
            top_p=resolved_top_p,
        )

    async def generate_batch(self, prompts: list[str], **kwargs: object) -> list[str]:
        """Generate text for a list of prompts asynchronously in batches."""
        max_tokens = kwargs.get("max_tokens")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")

        max_new_tokens = (
            int(max_tokens)
            if isinstance(max_tokens, int)
            else self._config.default_max_tokens
        )
        resolved_temperature = (
            float(temperature)
            if isinstance(temperature, (float, int))
            else self._config.default_temperature
        )
        resolved_top_p = (
            float(top_p)
            if isinstance(top_p, (float, int))
            else self._config.default_top_p
        )

        return await asyncio.to_thread(
            self._generate_batch_sync,
            prompts,
            max_new_tokens,
            resolved_temperature,
            resolved_top_p,
        )

    async def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Generate a chat completion from role-based messages."""
        prompt = self._format_chat_messages(messages)
        max_tokens = kwargs.get("max_tokens")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")
        return await self.generate(
            prompt,
            max_tokens=int(max_tokens) if isinstance(max_tokens, int) else None,
            temperature=float(temperature)
            if isinstance(temperature, (float, int))
            else None,
            top_p=float(top_p) if isinstance(top_p, (float, int)) else None,
        )

    def switch_adapter(self, adapter_name: str) -> None:
        """Switch the active LoRA adapter on the underlying model."""
        self._model_loader.switch_adapter(adapter_name)

    def load_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        """Load a LoRA adapter into the underlying model."""
        self._model_loader.load_adapter(adapter_path, adapter_name=adapter_name)

    def _format_chat_messages(self, messages: list[dict[str, str]]) -> str:
        """Format role/content messages into Qwen ChatML prompt text."""
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _generate_batch_sync(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        responses: list[str] = []
        batch_size = max(self._config.batch_size, 1)
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            for prompt in batch:
                responses.append(
                    self._model_loader.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
        return responses
