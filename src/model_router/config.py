"""Configuration for Model Router — loaded from environment.

Supports multiple LLM providers via the ``providers`` dict.
Backward-compat: OPENROUTER_API_KEY env var still wires up the
openrouter provider automatically.
"""

import os
import json
from dataclasses import dataclass, field


def _default_providers():
    """Build default provider config from environment variables.

    OPENROUTER_API_KEY → openrouter provider added automatically.
    GROQ_API_KEY → groq provider added automatically.
    Additional providers can be configured via PROVIDERS_JSON env var.
    """
    providers = {}

    or_key = os.getenv("OPENROUTER_API_KEY", "")
    if or_key:
        providers["openrouter"] = {
            "api_key": or_key,
            "base_url": os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
        }

    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        providers["groq"] = {
            "api_key": groq_key,
            "base_url": os.getenv(
                "GROQ_BASE_URL",
                "https://api.groq.com/openai/v1",
            ),
        }

    # Allow full override via JSON env var
    extra = os.getenv("PROVIDERS_JSON")
    if extra:
        try:
            parsed = json.loads(extra)
            providers.update(parsed)
        except json.JSONDecodeError:
            pass

    return providers


@dataclass
class RouterConfig:
    """Configuration loaded from environment with sensible defaults.

    New provider system:
      - ``providers`` maps provider name → {api_key, base_url, ...}
      - ``default_provider`` used when a model has no explicit provider

    Legacy fields (still work):
      - ``openrouter_api_key`` — auto-populated from OPENROUTER_API_KEY
      - ``openrouter_base_url`` — auto-populated
    """

    # ── Provider system (new) ──────────────────────────────────────────

    providers: dict = field(default_factory=_default_providers)
    default_provider: str = "openrouter"

    # ── OpenRouter (legacy, auto-populated from providers dict) ─────────

    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Request settings ───────────────────────────────────────────────

    request_timeout_seconds: int = 20

    # Classification
    complexity_method: str = "hybrid"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Embedder backend: "minilm" (semantic) or "dice" (zero-dep word overlap)
    embedder_backend: str = "minilm"

    # Routing
    default_tier: str = "fast"
    rate_limit_retry_count: int = 3
    rate_limit_base_delay: float = 1.0
    rate_limit_max_delay: float = 30.0

    # Cascade
    cascade_enabled: bool = True
    cascade_max_hops: int = 2
    cascade_max_budget_tokens: int = 10000

    # Circuit breaker
    circuit_breaker_cooldown: int = 60
    circuit_breaker_max_failures: int = 3

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080
    dashboard_history_size: int = 1000

    # Logging
    log_level: str = "INFO"


def get_config() -> RouterConfig:
    """Load config from environment, with env var overrides."""
    return RouterConfig(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT", "20")),
        complexity_method=os.getenv("COMPLEXITY_METHOD", "hybrid"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedder_backend=os.getenv("MODEL_ROUTER_EMBEDDER", "minilm"),
        cascade_enabled=os.getenv("CASCADE_ENABLED", "true").lower() == "true",
        cascade_max_hops=int(os.getenv("CASCADE_MAX_HOPS", "2")),
        cascade_max_budget_tokens=int(os.getenv("CASCADE_MAX_BUDGET", "10000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "8080")),
        rate_limit_retry_count=int(os.getenv("RATE_LIMIT_RETRIES", "3")),
        circuit_breaker_cooldown=int(os.getenv("CB_COOLDOWN", "60")),
        circuit_breaker_max_failures=int(os.getenv("CB_MAX_FAILURES", "3")),
        # Provider system fields are populated by _default_providers()
        providers=_default_providers(),
        default_provider=os.getenv("MODEL_ROUTER_DEFAULT_PROVIDER", "openrouter"),
    )
