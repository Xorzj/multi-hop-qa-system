from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONFIG_PATH = "config/config.yaml"


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass
class LLMConfig:
    provider: str
    model_path: str
    adapter_path: str
    generation: GenerationConfig


@dataclass
class GraphConfig:
    uri: str
    user: str
    password: str
    database: str


@dataclass
class APIConfig:
    host: str
    port: int


@dataclass
class LoggingConfig:
    level: str


@dataclass
class DataProcessingConfig:
    strip_base64_images: bool


@dataclass
class ExtractionConfig:
    entity_chunk_size: int
    triple_chunk_size: int
    max_context_entities: int
    max_new_tokens: int
    temperature: float
    top_p: float
    max_retries: int
    relation_types: list[str] = field(default_factory=list)


@dataclass
class Config:
    llm: LLMConfig
    graph: GraphConfig
    api: APIConfig
    logging: LoggingConfig
    data_processing: DataProcessingConfig
    extraction: ExtractionConfig


def _interpolate_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _interpolate_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_values(item) for item in value]
    return value


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        message = (
            "PyYAML is required to load configuration. "
            "Install it with `uv add pyyaml` or `pip install pyyaml`."
        )
        raise ModuleNotFoundError(message) from exc
    return yaml


def _build_config(data: dict[str, Any]) -> Config:
    llm_data = data.get("llm", {})
    generation_data = llm_data.get("generation", {})
    llm_config = LLMConfig(
        provider=llm_data.get("provider", ""),
        model_path=llm_data.get("model_path", ""),
        adapter_path=llm_data.get("adapter_path", ""),
        generation=GenerationConfig(
            max_new_tokens=generation_data.get("max_new_tokens", 0),
            temperature=generation_data.get("temperature", 0.0),
            top_p=generation_data.get("top_p", 0.0),
        ),
    )
    graph_data = data.get("graph", {})
    graph_config = GraphConfig(
        uri=graph_data.get("uri", ""),
        user=graph_data.get("user", ""),
        password=graph_data.get("password", ""),
        database=graph_data.get("database", ""),
    )
    api_data = data.get("api", {})
    api_config = APIConfig(
        host=api_data.get("host", ""),
        port=api_data.get("port", 0),
    )
    logging_data = data.get("logging", {})
    logging_config = LoggingConfig(level=logging_data.get("level", ""))
    data_processing_data = data.get("data_processing", {})
    data_processing_config = DataProcessingConfig(
        strip_base64_images=data_processing_data.get("strip_base64_images", False)
    )
    extraction_data = data.get("extraction", {})
    extraction_config = ExtractionConfig(
        entity_chunk_size=extraction_data.get("entity_chunk_size", 1500),
        triple_chunk_size=extraction_data.get("triple_chunk_size", 800),
        max_context_entities=extraction_data.get("max_context_entities", 15),
        max_new_tokens=extraction_data.get("max_new_tokens", 2048),
        temperature=extraction_data.get("temperature", 0.05),
        top_p=extraction_data.get("top_p", 0.1),
        max_retries=extraction_data.get("max_retries", 3),
        relation_types=extraction_data.get("relation_types", []),
    )
    return Config(
        llm=llm_config,
        graph=graph_config,
        api=api_config,
        logging=logging_config,
        data_processing=data_processing_config,
        extraction=extraction_config,
    )


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Config:
    yaml = _require_yaml()
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    interpolated = _interpolate_env_values(data)
    return _build_config(interpolated)
