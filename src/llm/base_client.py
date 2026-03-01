"""Base LLM client abstractions and generation parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationParams:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


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
