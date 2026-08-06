"""Tests for the provider abstraction — BaseProvider, registry, OpenRouterProvider."""

from unittest.mock import patch, MagicMock
import pytest

from model_router.providers import (
    BaseProvider,
    register_provider,
    get_provider,
    list_providers,
)
from model_router.providers.openrouter import OpenRouterProvider
from model_router.models import GenerationResult


# =============================================================================
# Registry
# =============================================================================


def test_register_and_list():
    """Registered providers appear in list_providers()."""
    before = list_providers()
    assert "openrouter" in before  # auto-registered


def test_get_provider_openrouter():
    """get_provider returns an OpenRouterProvider instance."""
    provider = get_provider("openrouter", api_key="test-key")
    assert isinstance(provider, OpenRouterProvider)
    assert provider.name == "openrouter"
    assert provider.api_key == "test-key"


def test_get_provider_unknown():
    """Unknown provider name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent")


def test_get_provider_with_config():
    """Extra config kwargs are passed to the provider."""
    provider = get_provider(
        "openrouter",
        api_key="test-key",
        config={"timeout": 99, "base_url": "http://custom:8080"},
    )
    assert provider.timeout == 99
    assert provider.base_url == "http://custom:8080"


# =============================================================================
# BaseProvider
# =============================================================================


def test_base_provider_abstract():
    """BaseProvider cannot be instantiated directly (abstract methods)."""
    with pytest.raises(TypeError):
        BaseProvider()  # noqa


class _ConcreteProvider(BaseProvider):
    """Minimal concrete implementation for testing."""

    name = "test"
    model_prefix = "test/"

    def generate(
        self,
        query: str,
        model_id: str,
        tier: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str = None,
        fallback_models: list[str] = None,
    ):
        return GenerationResult(
            query=query, response="ok", model_id=model_id, tier=tier,
            tokens_in=0, tokens_out=0, latency_ms=0,
        )

    def list_available_models(self):
        return [{"id": "test/model"}]


def test_concrete_provider():
    p = _ConcreteProvider(api_key="k")
    assert p.name == "test"
    assert p.api_key == "k"
    result = p.generate("hi", "test/model", "fast")
    assert result.response == "ok"


def test_custom_register_and_get():
    """Custom provider can be registered and resolved."""
    register_provider("test-provider", _ConcreteProvider)
    p = get_provider("test-provider", api_key="test")
    assert isinstance(p, _ConcreteProvider)


# =============================================================================
# OpenRouterProvider — unit tests (mock HTTP, no real API)
# =============================================================================


@pytest.fixture
def orp():
    """OpenRouterProvider with a mock session."""
    provider = OpenRouterProvider(api_key="test-key", timeout=10)
    provider.session = MagicMock()
    return provider


class TestOpenRouterGenerate:
    def test_success_returns_response(self, orp):
        """200 status returns a GenerationResult with response text."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        orp.session.post.return_value = mock_resp

        result = orp.generate("hi", "test-model", "fast")
        assert result.error is None
        assert result.response == "Hello!"
        assert result.tokens_in == 10
        assert result.tokens_out == 5

    def test_rate_limit_then_retry_then_success(self, orp):
        """429 then 200 — jittered backoff fires, then succeeds."""
        # First call: 429, second call: 200
        error_resp = MagicMock()
        error_resp.status_code = 429
        error_resp.headers = {}

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "choices": [{"message": {"content": "Retried!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

        orp.session.post.side_effect = [error_resp, success_resp]
        orp.retry_count = 1  # only 1 retry

        result = orp.generate("hi", "test-model", "fast")
        assert result.error is None
        assert result.response == "Retried!"
        assert orp.session.post.call_count == 2

    def test_all_retries_exhausted(self, orp):
        """Persistent 429 after all retries returns error."""
        error_resp = MagicMock()
        error_resp.status_code = 429
        error_resp.headers = {}

        orp.session.post.return_value = error_resp
        orp.retry_count = 2

        result = orp.generate("hi", "test-model", "fast")
        assert result.error is not None

    def test_circuit_breaker_skips_open_model(self, orp):
        """Circuit breaker skips model on cooldown."""
        orp.circuit_breaker.record_failure("test-model")
        orp.circuit_breaker.record_failure("test-model")
        orp.circuit_breaker.record_failure("test-model")  # trips

        result = orp.generate("hi", "test-model", "fast")
        assert "Circuit open" in (result.error or "")
        orp.session.post.assert_not_called()

    def test_fallback_models_tried_on_failure(self, orp):
        """When primary fails, fallback models are tried in order."""
        fail_resp = MagicMock()
        fail_resp.status_code = 429
        fail_resp.headers = {}

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "choices": [{"message": {"content": "Fallback worked!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

        orp.session.post.side_effect = [fail_resp, success_resp]
        orp.retry_count = 0  # no retries on primary

        result = orp.generate(
            "hi", "primary-model", "fast",
            fallback_models=["fallback-model"],
        )
        assert result.response == "Fallback worked!"
        # Called twice: primary + fallback
        assert orp.session.post.call_count == 2

    def test_fallback_skips_same_model(self, orp):
        """Fallback list containing the same model as primary is skipped."""
        fail_resp = MagicMock()
        fail_resp.status_code = 429
        fail_resp.headers = {}

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "choices": [{"message": {"content": "Other worked!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

        orp.session.post.side_effect = [fail_resp, success_resp]
        orp.retry_count = 0

        result = orp.generate(
            "hi", "test-model", "fast",
            fallback_models=["test-model", "other-model"],
        )
        # Should skip test-model (same as primary) and try other-model
        assert result.response == "Other worked!"

    def test_timeout_retries(self, orp):
        """ConnectionError triggers retry, then success on second attempt."""
        import requests

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "choices": [{"message": {"content": "After timeout!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

        orp.session.post.side_effect = [
            requests.ConnectionError("timeout"),
            success_resp,
        ]
        orp.retry_count = 1

        result = orp.generate("hi", "test-model", "fast")
        assert result.error is None
        assert result.response == "After timeout!"

    def test_system_prompt_included(self, orp):
        """System prompt is included in the messages array."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        orp.session.post.return_value = mock_resp

        orp.generate("hi", "test-model", "fast", system_prompt="You are a bot")
        call_kwargs = orp.session.post.call_args[1]
        messages = call_kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a bot"
        assert messages[1]["role"] == "user"


class TestOpenRouterListModels:
    def test_list_available_models(self, orp):
        """list_available_models returns parsed model data."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"id": "model-a"}, {"id": "model-b"}],
        }
        orp.session.get.return_value = mock_resp

        models = orp.list_available_models()
        assert len(models) == 2
        assert models[0]["id"] == "model-a"

    def test_list_models_failure_returns_empty(self, orp):
        """Non-200 or exception returns empty list."""
        orp.session.get.side_effect = Exception("network error")
        models = orp.list_available_models()
        assert models == []
