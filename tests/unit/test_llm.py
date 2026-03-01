from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.common.config import GenerationConfig, LLMConfig
from src.common.exceptions import ConfigError, LLMError
from src.llm.base_client import BaseLLMClient, GenerationParams
from src.llm.client_factory import OpenAIClient, create_llm_client
from src.llm.zhipu_client import ZhipuClient


@pytest.fixture()
def generation_config() -> GenerationConfig:
    return GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8)


@pytest.fixture()
def llm_config_factory(generation_config: GenerationConfig):
    def _factory(provider: str, model_path: str = "model") -> LLMConfig:
        return LLMConfig(
            provider=provider,
            model_path=model_path,
            adapter_path="",
            generation=generation_config,
        )

    return _factory


def test_base_client_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseLLMClient()


def test_create_llm_client_local(
    monkeypatch: pytest.MonkeyPatch, llm_config_factory
) -> None:
    @dataclass
    class DummyModelConfig:
        model_name: str = "default-model"

    @dataclass
    class DummyInferenceConfig:
        model_config: DummyModelConfig
        default_max_tokens: int
        default_temperature: float
        default_top_p: float

    class DummyEngine:
        def __init__(self, config: DummyInferenceConfig) -> None:
            self.config = config

    class DummyLocalClient:
        def __init__(self, engine: DummyEngine) -> None:
            self.engine = engine

    import src.llm.client_factory as client_factory

    monkeypatch.setattr(client_factory, "ModelConfig", DummyModelConfig)
    monkeypatch.setattr(client_factory, "InferenceConfig", DummyInferenceConfig)
    monkeypatch.setattr(client_factory, "InferenceEngine", DummyEngine)
    monkeypatch.setattr(client_factory, "LocalLLMClient", DummyLocalClient)

    config = llm_config_factory("local", model_path="test-model")
    client = create_llm_client(config)

    assert isinstance(client, DummyLocalClient)
    assert client.engine.config.model_config.model_name == "test-model"
    assert client.engine.config.default_max_tokens == config.generation.max_new_tokens
    assert client.engine.config.default_temperature == config.generation.temperature
    assert client.engine.config.default_top_p == config.generation.top_p


def test_create_llm_client_openai(llm_config_factory) -> None:
    client = create_llm_client(llm_config_factory("openai"))
    assert isinstance(client, OpenAIClient)


def test_create_llm_client_zhipu_requires_api_key(
    monkeypatch: pytest.MonkeyPatch, llm_config_factory
) -> None:
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ZHIPU_API_KEY is required"):
        create_llm_client(llm_config_factory("zhipu"))


def test_create_llm_client_zhipu_uses_default_model(
    monkeypatch: pytest.MonkeyPatch, llm_config_factory
) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    config = llm_config_factory("zhipu", model_path="")
    client = create_llm_client(config)

    assert isinstance(client, ZhipuClient)
    assert client._api_key == "test-key"
    assert client._model == "glm-4-flash"


def test_create_llm_client_unknown_provider(llm_config_factory) -> None:
    with pytest.raises(ConfigError, match="Unknown LLM provider"):
        create_llm_client(llm_config_factory("unknown"))


class FakeHttpxModule:
    class HTTPError(Exception):
        pass

    class HTTPStatusError(HTTPError):
        def __init__(self, message: str, response: Any) -> None:
            super().__init__(message)
            self.response = response

    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.last_request: dict[str, Any] | None = None

        module = self

        class AsyncClient:
            def __init__(self, timeout: float) -> None:
                self.timeout = timeout

            async def __aenter__(self) -> AsyncClient:
                return self

            async def __aexit__(self, exc_type, exc, tb) -> bool:
                return False

            async def post(
                self, url: str, json: dict[str, Any], headers: dict[str, str]
            ):
                module.last_request = {"url": url, "json": json, "headers": headers}
                return module._response

        self.AsyncClient = AsyncClient


class FakeResponse:
    def __init__(
        self,
        json_data: Any,
        text: str = "ok",
        status_code: int = 200,
        raise_error: Exception | None = None,
    ) -> None:
        self._json_data = json_data
        self.text = text
        self.status_code = status_code
        self._raise_error = raise_error

    def raise_for_status(self) -> None:
        if self._raise_error:
            raise self._raise_error

    def json(self) -> Any:
        return self._json_data


@pytest.fixture()
def zhipu_client() -> ZhipuClient:
    return ZhipuClient(
        api_key="test-key", model="glm-4-flash", base_url="https://example.com/"
    )


@pytest.mark.asyncio
async def test_zhipu_generate_calls_chat(
    monkeypatch: pytest.MonkeyPatch, zhipu_client
) -> None:
    async def fake_chat(messages, params=None):
        assert messages == [{"role": "user", "content": "hello"}]
        return "ok"

    monkeypatch.setattr(zhipu_client, "chat", fake_chat)

    result = await zhipu_client.generate("hello")

    assert result == "ok"


@pytest.mark.asyncio
async def test_zhipu_chat_success(
    monkeypatch: pytest.MonkeyPatch, zhipu_client
) -> None:
    response = FakeResponse({"choices": [{"message": {"content": "pong"}}]})
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(zhipu_client, "_load_httpx", lambda: fake_httpx)

    result = await zhipu_client.chat([{"role": "user", "content": "ping"}])

    assert result == "pong"
    assert fake_httpx.last_request is not None
    assert fake_httpx.last_request["url"] == "https://example.com/chat/completions"
    assert fake_httpx.last_request["json"]["model"] == "glm-4-flash"
    assert (
        fake_httpx.last_request["json"]["max_tokens"]
        == GenerationParams().max_new_tokens
    )
    assert fake_httpx.last_request["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_zhipu_chat_http_status_error(
    monkeypatch: pytest.MonkeyPatch, zhipu_client
) -> None:
    response = FakeResponse({"choices": []}, text="bad", status_code=400)
    error = FakeHttpxModule.HTTPStatusError("bad request", response)
    response._raise_error = error
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(zhipu_client, "_load_httpx", lambda: fake_httpx)

    with pytest.raises(LLMError, match="Zhipu API error: 400 bad"):
        await zhipu_client.chat([{"role": "user", "content": "ping"}])


@pytest.mark.asyncio
async def test_zhipu_chat_http_error(
    monkeypatch: pytest.MonkeyPatch, zhipu_client
) -> None:
    response = FakeResponse(
        {"choices": []}, raise_error=FakeHttpxModule.HTTPError("boom")
    )
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(zhipu_client, "_load_httpx", lambda: fake_httpx)

    with pytest.raises(LLMError, match="Zhipu API request failed"):
        await zhipu_client.chat([{"role": "user", "content": "ping"}])


@pytest.mark.asyncio
async def test_zhipu_chat_unexpected_response(
    monkeypatch: pytest.MonkeyPatch, zhipu_client
) -> None:
    response = FakeResponse({"unexpected": "payload"})
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(zhipu_client, "_load_httpx", lambda: fake_httpx)

    with pytest.raises(LLMError, match="Unexpected Zhipu API response"):
        await zhipu_client.chat([{"role": "user", "content": "ping"}])


@pytest.mark.asyncio
async def test_zhipu_chat_non_string_content(
    monkeypatch: pytest.MonkeyPatch, zhipu_client
) -> None:
    response = FakeResponse({"choices": [{"message": {"content": 123}}]})
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(zhipu_client, "_load_httpx", lambda: fake_httpx)

    with pytest.raises(LLMError, match="response content is not a string"):
        await zhipu_client.chat([{"role": "user", "content": "ping"}])
