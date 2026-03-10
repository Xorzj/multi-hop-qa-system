from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.common.config import GenerationConfig, LLMConfig
from src.common.exceptions import ConfigError, LLMError
from src.llm.base_client import BaseLLMClient, GenerationParams
from src.llm.client_factory import create_llm_client
from src.llm.openai_client import OpenAIClient
from src.llm.zhipu_client import ZhipuClient


@pytest.fixture()
def generation_config() -> GenerationConfig:
    return GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8)


@pytest.fixture()
def llm_config_factory(generation_config: GenerationConfig):
    def _factory(
        provider: str,
        model_path: str = "model",
        base_url: str = "",
        api_key: str = "",
    ) -> LLMConfig:
        return LLMConfig(
            provider=provider,
            model_path=model_path,
            adapter_path="",
            base_url=base_url,
            api_key=api_key,
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
    with pytest.raises(ConfigError, match="api_key is required"):
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

    class ConnectError(Exception):
        pass

    class ReadTimeout(Exception):
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

    # Non-string content (e.g. int) is treated as empty, returns ""
    result = await zhipu_client.chat([{"role": "user", "content": "ping"}])
    assert result == ""


@pytest.mark.asyncio
async def test_openai_chat_success(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LLMConfig(
        provider="openai",
        model_path="gpt-4o-mini",
        adapter_path="",
        base_url="https://example.com/v1",
        api_key="test-key",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    response = FakeResponse({"choices": [{"message": {"content": "pong"}}]})
    fake_httpx = FakeHttpxModule(response)

    monkeypatch.setattr(client, "_load_httpx", lambda: fake_httpx)
    result = await client.chat([{"role": "user", "content": "ping"}])

    assert result == "pong"
    assert fake_httpx.last_request is not None
    assert fake_httpx.last_request["json"]["model"] == "gpt-4o-mini"
    assert fake_httpx.last_request["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_openai_generate_calls_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LLMConfig(
        provider="openai",
        model_path="gpt-4o-mini",
        adapter_path="",
        base_url="https://example.com/v1",
        api_key="test-key",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)

    async def fake_chat(messages, params=None):
        assert messages == [{"role": "user", "content": "hello"}]
        return "ok"

    monkeypatch.setattr(client, "chat", fake_chat)
    result = await client.generate("hello")
    assert result == "ok"


@pytest.mark.asyncio
async def test_openai_chat_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LLMConfig(
        provider="openai",
        model_path="gpt-4o-mini",
        adapter_path="",
        base_url="https://example.com/v1",
        api_key="test-key",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    response = FakeResponse({"choices": []}, text="bad", status_code=400)
    error = FakeHttpxModule.HTTPStatusError("bad request", response)
    response._raise_error = error
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(client, "_load_httpx", lambda: fake_httpx)

    with pytest.raises(LLMError, match="OpenAI API error: 400 bad"):
        await client.chat([{"role": "user", "content": "ping"}])


@pytest.mark.asyncio
async def test_openai_chat_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LLMConfig(
        provider="openai",
        model_path="gpt-4o-mini",
        adapter_path="",
        base_url="https://example.com/v1",
        api_key="test-key",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    response = FakeResponse(
        {"choices": []}, raise_error=FakeHttpxModule.HTTPError("boom")
    )
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(client, "_load_httpx", lambda: fake_httpx)

    with pytest.raises(LLMError, match="OpenAI API request failed"):
        await client.chat([{"role": "user", "content": "ping"}])


@pytest.mark.asyncio
async def test_openai_chat_unexpected_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LLMConfig(
        provider="openai",
        model_path="gpt-4o-mini",
        adapter_path="",
        base_url="https://example.com/v1",
        api_key="test-key",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    response = FakeResponse({"unexpected": "payload"})
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(client, "_load_httpx", lambda: fake_httpx)

    with pytest.raises(LLMError, match="Unexpected OpenAI API response"):
        await client.chat([{"role": "user", "content": "ping"}])


@pytest.mark.asyncio
async def test_openai_chat_non_string_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LLMConfig(
        provider="openai",
        model_path="gpt-4o-mini",
        adapter_path="",
        base_url="https://example.com/v1",
        api_key="test-key",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    response = FakeResponse({"choices": [{"message": {"content": 123}}]})
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(client, "_load_httpx", lambda: fake_httpx)

    # Non-string content (e.g. int) is treated as empty, returns ""
    result = await client.chat([{"role": "user", "content": "ping"}])
    assert result == ""


@pytest.mark.asyncio
async def test_openai_chat_list_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LLMConfig(
        provider="openai",
        model_path="gpt-4o-mini",
        adapter_path="",
        base_url="https://example.com/v1",
        api_key="test-key",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    list_content = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]
    response = FakeResponse({"choices": [{"message": {"content": list_content}}]})
    fake_httpx = FakeHttpxModule(response)
    monkeypatch.setattr(client, "_load_httpx", lambda: fake_httpx)

    result = await client.chat([{"role": "user", "content": "ping"}])
    # List content is not a string, treated as empty
    assert result == ""


def test_openai_resolve_endpoint_missing_base_url() -> None:
    config = LLMConfig(
        provider="openai",
        model_path="",
        adapter_path="",
        base_url="",
        api_key="sk-test",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    with pytest.raises(ConfigError, match="base_url is required"):
        client._resolve_endpoint_and_key()


def test_openai_resolve_endpoint_missing_api_key() -> None:
    config = LLMConfig(
        provider="openai",
        model_path="",
        adapter_path="",
        base_url="https://example.com/v1",
        api_key="",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    with pytest.raises(ConfigError, match="api_key is required"):
        client._resolve_endpoint_and_key()


def test_openai_resolve_appends_chat_completions() -> None:
    config = LLMConfig(
        provider="openai",
        model_path="",
        adapter_path="",
        base_url="https://example.com/v1",
        api_key="sk-test",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    endpoint, key = client._resolve_endpoint_and_key()
    assert endpoint == "https://example.com/v1/chat/completions"
    assert key == "sk-test"


def test_openai_timeout_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "not-a-number")
    config = LLMConfig(
        provider="openai",
        model_path="",
        adapter_path="",
        base_url="",
        api_key="",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    assert client._timeout_seconds == 180.0


def test_openai_timeout_negative_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "-5")
    config = LLMConfig(
        provider="openai",
        model_path="",
        adapter_path="",
        base_url="",
        api_key="",
        generation=GenerationConfig(max_new_tokens=128, temperature=0.3, top_p=0.8),
    )
    client = OpenAIClient(config)
    assert client._timeout_seconds == 180.0


# ──────────────── post_with_retry tests ────────────────


class _FakeResponse:
    """Minimal fake httpx.Response for testing retry logic."""

    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        headers: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _FakeHTTPStatusError(self)


class _FakeHTTPStatusError(Exception):
    """Minimal stand-in for httpx.HTTPStatusError."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


class _FakeConnectError(Exception):
    pass


class _FakeReadTimeout(Exception):
    pass


class _FakeHTTPError(Exception):
    pass


class _FakeAsyncClient:
    """Fake httpx.AsyncClient that returns pre-configured responses."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        resp = self._responses[idx]
        self._call_count += 1
        return resp


def _make_fake_httpx(responses: list[_FakeResponse]) -> Any:
    """Build a fake httpx module that returns a sequence of responses."""
    client_instance = _FakeAsyncClient(responses)

    class FakeModule:
        HTTPStatusError = _FakeHTTPStatusError
        HTTPError = _FakeHTTPError
        ConnectError = _FakeConnectError
        ReadTimeout = _FakeReadTimeout

        class AsyncClient:
            def __init__(self, **kwargs: Any):
                pass

            async def __aenter__(self):
                return client_instance

            async def __aexit__(self, *args: Any):
                pass

            async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
                return await client_instance.post(url, **kwargs)

    return FakeModule()


class TestPostWithRetry:
    """Tests for post_with_retry exponential backoff logic."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        from src.llm.base_client import post_with_retry

        fake = _make_fake_httpx([_FakeResponse(200)])
        resp = await post_with_retry(
            fake,
            "http://api/v1/chat",
            json={},
            headers={},
            timeout=30.0,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_429_retries_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Make backoff instant for testing
        import asyncio as _asyncio

        from src.llm.base_client import post_with_retry

        sleep_calls: list[float] = []
        original_sleep = _asyncio.sleep

        async def fast_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            await original_sleep(0)  # instant

        monkeypatch.setattr(_asyncio, "sleep", fast_sleep)

        fake = _make_fake_httpx(
            [
                _FakeResponse(429, text="rate limited"),
                _FakeResponse(429, text="rate limited"),
                _FakeResponse(200, text="ok"),
            ]
        )
        resp = await post_with_retry(
            fake,
            "http://api/v1/chat",
            json={},
            headers={},
            timeout=30.0,
            initial_backoff=1.0,
        )
        assert resp.status_code == 200
        assert len(sleep_calls) == 2  # slept twice before success
        # Backoff should increase (exponential)
        assert sleep_calls[1] > sleep_calls[0]

    @pytest.mark.asyncio
    async def test_429_exhausts_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio as _asyncio

        from src.llm.base_client import post_with_retry

        async def fast_sleep(seconds: float) -> None:
            pass

        monkeypatch.setattr(_asyncio, "sleep", fast_sleep)

        fake = _make_fake_httpx([_FakeResponse(429, text="rate limited")])
        with pytest.raises(_FakeHTTPStatusError):
            await post_with_retry(
                fake,
                "http://api/v1/chat",
                json={},
                headers={},
                timeout=30.0,
                max_retries=3,
            )

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self) -> None:
        from src.llm.base_client import post_with_retry

        fake = _make_fake_httpx([_FakeResponse(401, text="unauthorized")])
        with pytest.raises(_FakeHTTPStatusError) as exc_info:
            await post_with_retry(
                fake,
                "http://api/v1/chat",
                json={},
                headers={},
                timeout=30.0,
            )
        assert exc_info.value.response.status_code == 401

    @pytest.mark.asyncio
    async def test_500_is_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio as _asyncio

        from src.llm.base_client import post_with_retry

        async def fast_sleep(seconds: float) -> None:
            pass  # instant, no real sleep

        monkeypatch.setattr(_asyncio, "sleep", fast_sleep)

        fake = _make_fake_httpx(
            [
                _FakeResponse(500, text="server error"),
                _FakeResponse(200, text="ok"),
            ]
        )
        resp = await post_with_retry(
            fake,
            "http://api/v1/chat",
            json={},
            headers={},
            timeout=30.0,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_retry_after_header_respected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import asyncio as _asyncio

        from src.llm.base_client import post_with_retry

        sleep_calls: list[float] = []

        async def fast_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        monkeypatch.setattr(_asyncio, "sleep", fast_sleep)

        fake = _make_fake_httpx(
            [
                _FakeResponse(429, headers={"retry-after": "3.5"}),
                _FakeResponse(200),
            ]
        )
        await post_with_retry(
            fake,
            "http://api/v1/chat",
            json={},
            headers={},
            timeout=30.0,
        )
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == 3.5  # exact Retry-After value


class TestComputeBackoff:
    """Tests for _compute_backoff helper."""

    def test_exponential_growth(self) -> None:
        from src.llm.base_client import _compute_backoff

        b0 = _compute_backoff(0, 1.0)
        b1 = _compute_backoff(1, 1.0)
        b2 = _compute_backoff(2, 1.0)
        # Base doubles: 1, 2, 4 (plus jitter)
        assert b0 < b1 < b2

    def test_max_capped(self) -> None:
        from src.llm.base_client import MAX_BACKOFF_S, _compute_backoff

        result = _compute_backoff(100, 1.0)  # huge attempt number
        assert result <= MAX_BACKOFF_S * 1.5 + 2  # backoff + max jitter

    def test_retry_after_header(self) -> None:
        from src.llm.base_client import _compute_backoff

        assert _compute_backoff(0, 1.0, "5.0") == 5.0
        assert _compute_backoff(0, 1.0, "0.3") == 0.5  # min 0.5

    def test_retry_after_header_invalid(self) -> None:
        from src.llm.base_client import _compute_backoff

        result = _compute_backoff(0, 1.0, "not-a-number")
        # Falls back to exponential
        assert 0.5 <= result <= 3.0  # 1.0 base + up to 2.0 jitter
