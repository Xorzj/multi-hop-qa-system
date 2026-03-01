"""LLM client factory and stub implementations."""

from __future__ import annotations

import os

from src.common.config import LLMConfig
from src.common.exceptions import ConfigError
from src.inference.inference_engine import InferenceConfig, InferenceEngine
from src.inference.model_loader import ModelConfig
from src.llm.base_client import BaseLLMClient, GenerationParams
from src.llm.local_client import LocalLLMClient
from src.llm.zhipu_client import ZhipuClient


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def generate(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> str:
        raise NotImplementedError("TODO: implement")

    async def chat(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams | None = None,
    ) -> str:
        raise NotImplementedError("TODO: implement")

    @property
    def provider(self) -> str:
        raise NotImplementedError("TODO: implement")


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    if config.provider == "local":
        model_config = ModelConfig(
            model_name=config.model_path or ModelConfig().model_name,
        )
        inference_config = InferenceConfig(
            model_config=model_config,
            default_max_tokens=config.generation.max_new_tokens,
            default_temperature=config.generation.temperature,
            default_top_p=config.generation.top_p,
        )
        engine = InferenceEngine(inference_config)
        return LocalLLMClient(engine)
    if config.provider == "openai":
        return OpenAIClient(config)
    if config.provider == "zhipu":
        api_key = os.getenv("ZHIPU_API_KEY", "")
        if not api_key:
            raise ConfigError("ZHIPU_API_KEY is required for Zhipu provider")
        model_name = config.model_path or "glm-4-flash"
        return ZhipuClient(api_key=api_key, model=model_name)
    raise ConfigError(f"Unknown LLM provider: {config.provider}")
