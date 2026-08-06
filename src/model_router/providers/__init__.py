"""Provider abstraction — make Model Router service-agnostic.

BaseProvider defines the interface. Register implementations with
``register_provider()`` and resolve with ``get_provider()``.

Built-in providers:
  - openrouter: OpenRouter API (free-model pool)
  - (future) anthropic, openai, ollama, google
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..models import GenerationResult

logger = logging.getLogger(__name__)

# ─── Registry ──────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type["BaseProvider"]] = {}


def register_provider(name: str, cls: type["BaseProvider"]):
    """Register a provider class under a short name (e.g. 'openrouter')."""
    _PROVIDERS[name] = cls
    logger.debug("Registered provider: %s (%s)", name, cls.__name__)


def get_provider(
    name: str,
    api_key: str = "",
    config: Optional[dict] = None,
) -> "BaseProvider":
    """Resolve a provider by name with the given config.

    Args:
        name: Provider name ('openrouter', etc.)
        api_key: API key for the provider.
        config: Optional dict of provider-specific settings.

    Returns:
        An initialised provider instance.

    Raises:
        ValueError: If the provider name is not registered.
    """
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Registered: {list(_PROVIDERS.keys())}"
        )
    kwargs = (config or {}).copy()
    # Use the explicit api_key arg if given; otherwise fall back to config dict
    if api_key:
        kwargs.pop("api_key", None)
    else:
        api_key = kwargs.pop("api_key", "")
    return cls(api_key=api_key, **kwargs)


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    return list(_PROVIDERS.keys())


# ─── Abstract Base ────────────────────────────────────────────────────────


class BaseProvider(ABC):
    """Abstract interface every provider must implement.

    Each provider handles:
      - API calls with retry / backoff (jittered exponential)
      - Per-model circuit breaker (skip dead models temporarily)
      - Intra-tier fallback (try alternatives when primary fails)

    Provider metadata:
      - ``self.name`` — short identifier ('openrouter', 'anthropic', …)
      - ``self.model_prefix`` — string prefix for model IDs this provider
        handles (e.g. 'openai/', 'google/', 'ollama/'). Used by the router
        to dispatch model selection to the right provider.
    """

    name: str = "base"
    model_prefix: str = ""

    def __init__(self, api_key: str = "", **kwargs):
        self.api_key = api_key
        self._extra = kwargs

    # ── Required ──────────────────────────────────────────────────────────

    @abstractmethod
    def generate(
        self,
        query: str,
        model_id: str,
        tier: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate a response from the given model.

        Must implement:
          - Jittered exponential backoff on transient errors (429, 5xx)
          - Circuit-breaker awareness (skip models on cooldown)
          - Fallback chain to ``fallback_models`` when primary fails

        Returns a ``GenerationResult`` with response text or error set.
        """
        ...

    @abstractmethod
    def list_available_models(self) -> list[dict]:
        """Fetch available models from the provider.

        Returns a list of raw model dicts (provider-specific format)
        or an empty list on failure.
        """
        ...


# --- Auto-register built-in providers ------------------------------------------
from . import openrouter  # noqa: E402 — triggers OpenRouterProvider registration
from . import groq  # noqa: E402 — triggers GroqProvider registration
