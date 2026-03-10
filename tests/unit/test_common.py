import logging
from collections.abc import Generator
from pathlib import Path

import pytest

from src.common import config as config_module
from src.common import exceptions as exceptions_module
from src.common import logger as logger_module


@pytest.fixture
def sample_config_data() -> dict:
    return {
        "llm": {
            "provider": "local",
            "model_path": "/models/base",
            "adapter_path": "/models/adapter",
            "generation": {
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        },
        "graph": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",
            "database": "neo4j",
        },
        "api": {"host": "0.0.0.0", "port": 8000},
        "logging": {"level": "DEBUG"},
        "data_processing": {"strip_base64_images": True},
    }


@pytest.fixture
def config_path(tmp_path: Path, sample_config_data: dict) -> Path:
    import tomli_w

    path = tmp_path / "config.toml"
    path.write_bytes(tomli_w.dumps(sample_config_data).encode())
    return path


@pytest.fixture
def empty_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "empty.toml"
    path.write_bytes(b"")
    return path


@pytest.fixture
def clean_root_logger() -> Generator[logging.Logger, None, None]:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    root_logger.handlers.clear()
    yield root_logger
    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)


def test_load_config_builds_nested_config(config_path: Path) -> None:
    config = config_module.load_config(str(config_path))

    assert isinstance(config, config_module.Config)
    assert isinstance(config.llm, config_module.LLMConfig)
    assert isinstance(config.llm.generation, config_module.GenerationConfig)
    assert isinstance(config.graph, config_module.GraphConfig)
    assert isinstance(config.api, config_module.APIConfig)
    assert isinstance(config.logging, config_module.LoggingConfig)
    assert isinstance(config.data_processing, config_module.DataProcessingConfig)

    assert config.llm.provider == "local"
    assert config.llm.model_path == "/models/base"
    assert config.llm.adapter_path == "/models/adapter"
    assert config.llm.base_url == ""
    assert config.llm.api_key == ""
    assert config.llm.generation.max_new_tokens == 128
    assert config.llm.generation.temperature == 0.7
    assert config.llm.generation.top_p == 0.9
    assert config.graph.uri == "bolt://localhost:7687"
    assert config.graph.user == "neo4j"
    assert config.graph.password == "password"
    assert config.graph.database == "neo4j"
    assert config.api.host == "0.0.0.0"
    assert config.api.port == 8000
    assert config.logging.level == "DEBUG"
    assert config.data_processing.strip_base64_images is True


def test_load_config_env_interpolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tomli_w

    monkeypatch.setenv("TEST_MODEL_DIR", "/env/models")
    config_data = {
        "llm": {
            "provider": "local",
            "model_path": "${TEST_MODEL_DIR}/base",
            "adapter_path": "${TEST_MODEL_DIR}/adapter",
            "base_url": "",
            "api_key": "",
            "generation": {"max_new_tokens": 5, "temperature": 0.1, "top_p": 0.2},
        }
    }
    path = tmp_path / "env.toml"
    path.write_bytes(tomli_w.dumps(config_data).encode())

    config = config_module.load_config(str(path))

    assert config.llm.model_path == "/env/models/base"
    assert config.llm.adapter_path == "/env/models/adapter"


def test_load_config_with_missing_sections_uses_defaults(
    empty_config_path: Path,
) -> None:
    config = config_module.load_config(str(empty_config_path))

    assert config.llm.provider == ""
    assert config.llm.model_path == ""
    assert config.llm.base_url == ""
    assert config.llm.api_key == ""
    assert config.llm.generation.max_new_tokens == 0
    assert config.llm.generation.temperature == 0.0
    assert config.llm.generation.top_p == 0.0
    assert config.graph.uri == ""
    assert config.graph.user == ""
    assert config.graph.password == ""
    assert config.graph.database == ""
    assert config.api.host == ""
    assert config.api.port == 0
    assert config.logging.level == ""
    assert config.data_processing.strip_base64_images is False


def test_load_config_invalid_path_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        config_module.load_config(str(missing_path))


def test_setup_logging_configures_root_logger(
    clean_root_logger: logging.Logger,
) -> None:
    logger_module.setup_logging("DEBUG")

    assert clean_root_logger.handlers
    assert clean_root_logger.level == logging.DEBUG
    handler = clean_root_logger.handlers[0]
    assert handler.formatter is not None
    assert isinstance(handler.formatter, logging.Formatter)

    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    formatted = handler.formatter.format(record)
    assert "test.logger" in formatted
    assert "INFO" in formatted
    assert "hello" in formatted


def test_setup_logging_idempotent(clean_root_logger: logging.Logger) -> None:
    logger_module.setup_logging("WARNING")
    first_handlers = list(clean_root_logger.handlers)
    logger_module.setup_logging("ERROR")

    assert clean_root_logger.handlers == first_handlers
    assert clean_root_logger.level == logging.ERROR


def test_get_logger_returns_named_logger() -> None:
    logger = logger_module.get_logger("custom.logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "custom.logger"


def test_exceptions_inheritance_and_catching() -> None:
    assert issubclass(exceptions_module.ConfigError, exceptions_module.AppError)
    assert issubclass(exceptions_module.ValidationError, exceptions_module.AppError)
    assert issubclass(exceptions_module.NotFoundError, exceptions_module.AppError)
    assert issubclass(
        exceptions_module.ExternalServiceError, exceptions_module.AppError
    )
    assert issubclass(
        exceptions_module.GraphError, exceptions_module.ExternalServiceError
    )
    assert issubclass(
        exceptions_module.LLMError, exceptions_module.ExternalServiceError
    )

    for exc_type in (
        exceptions_module.ConfigError,
        exceptions_module.ValidationError,
        exceptions_module.NotFoundError,
        exceptions_module.ExternalServiceError,
        exceptions_module.GraphError,
        exceptions_module.LLMError,
    ):
        with pytest.raises(exceptions_module.AppError):
            raise exc_type("boom")
