"""LLM client factory and stub implementations."""

from __future__ import annotations

import os

from src.common.config import LLMConfig
from src.common.exceptions import ConfigError
from src.inference.inference_engine import InferenceConfig, InferenceEngine
from src.inference.model_loader import ModelConfig
from src.llm.base_client import BaseLLMClient
from src.llm.local_client import LocalLLMClient
from src.llm.openai_client import OpenAIClient
from src.llm.zhipu_client import ZhipuClient

__all__ = ["OpenAIClient", "create_llm_client"]


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
        api_key = config.api_key or os.getenv("ZHIPU_API_KEY", "")
        if not api_key:
            raise ConfigError(
                "llm.api_key is required in config.toml for Zhipu provider "
                "(or set ZHIPU_API_KEY environment variable)"
            )
        model_name = config.model_path or "glm-4-flash"
        return ZhipuClient(api_key=api_key, model=model_name)
    raise ConfigError(f"Unknown LLM provider: {config.provider}")
