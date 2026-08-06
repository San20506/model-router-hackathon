"""OpenRouter provider — talks to openrouter.ai API.

Extracted from ``client.py`` into the provider interface. The old
``OpenRouterClient`` now wraps this provider for backward compatibility.
"""

import logging
import random
import time
from typing import Optional, Callable

import requests

from ..models import GenerationResult
from ..client import CircuitBreaker
from . import BaseProvider, register_provider

logger = logging.getLogger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter's free-model pool.

    Uses the same jittered backoff, circuit breaker, and intra-tier fallback
    that ``OpenRouterClient`` had — just wrapped in the ``BaseProvider``
    interface so the pipeline can dispatch to any provider uniformly.
    """

    name = "openrouter"
    model_prefix = ""

    def __init__(
        self,
        api_key: str = "",
        timeout: int = 60,
        retry_count: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        circuit_breaker: Optional[CircuitBreaker] = None,
        base_url: str = _BASE_URL,
        **kwargs,
    ):
        super().__init__(api_key=api_key, **kwargs)
        self.timeout = timeout
        self.retry_count = retry_count
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/San20506/model-router",
            "X-Title": "Model Router",
        })

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
        """Generate, with jittered backoff and circuit-breaker awareness.

        If the primary ``model_id`` fails with a retryable error after all
        retries, tries each ``fallback_models`` in order (same tier).
        """
        result = self._generate_single(
            query, model_id, tier, max_tokens, temperature, system_prompt,
        )
        if result.error and fallback_models:
            for fb_id in fallback_models:
                if fb_id == model_id:
                    continue
                if self.circuit_breaker.is_open(fb_id):
                    continue
                logger.warning(
                    "Primary model %s failed, trying fallback: %s",
                    model_id, fb_id,
                )
                result = self._generate_single(
                    query, fb_id, tier, max_tokens, temperature, system_prompt,
                )
                if not result.error:
                    break
        return result

    def _generate_single(
        self,
        query: str,
        model_id: str,
        tier: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        on_attempt: Optional[Callable] = None,
    ) -> GenerationResult:
        """Generate, with jittered backoff and circuit-breaker awareness.

        ``on_attempt`` — optional callback ``fn(attempt, err)`` for upstream
        cascade / fallback logic to observe each failure.
        """
        # Circuit breaker check
        if self.circuit_breaker.is_open(model_id):
            cooldown = self.circuit_breaker.remaining_cooldown(model_id)
            msg = f"Circuit open for {model_id}, {cooldown:.0f}s remaining"
            logger.warning(msg)
            return GenerationResult(
                query=query, response="", model_id=model_id, tier=tier,
                tokens_in=0, tokens_out=0, latency_ms=0,
                error=msg,
            )

        start = time.perf_counter()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        for attempt in range(self.retry_count + 1):
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=self.timeout,
                )

                latency_ms = (time.perf_counter() - start) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    usage = data.get("usage", {})
                    self.circuit_breaker.record_success(model_id)
                    return GenerationResult(
                        query=query,
                        response=choice["message"]["content"],
                        model_id=model_id,
                        tier=tier,
                        tokens_in=usage.get("prompt_tokens", 0),
                        tokens_out=usage.get("completion_tokens", 0),
                        latency_ms=round(latency_ms, 1),
                    )

                # Transient errors (retryable)
                if resp.status_code == 429 and attempt < self.retry_count:
                    wait = self._backoff(attempt, resp)
                    logger.warning(
                        "Rate limited (429), retry %d/%d in %.1fs: %s",
                        attempt + 1, self.retry_count, wait, model_id,
                    )
                    if on_attempt:
                        on_attempt(attempt, "429")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500 and attempt < self.retry_count:
                    wait = self._backoff(attempt, resp)
                    logger.warning(
                        "Server error %d, retry %d/%d in %.1fs: %s",
                        resp.status_code, attempt + 1, self.retry_count, wait,
                        model_id,
                    )
                    if on_attempt:
                        on_attempt(attempt, f"{resp.status_code}")
                    time.sleep(wait)
                    continue

                # Non-retryable
                error_msg = f"API error {resp.status_code}: {resp.text[:200]}"
                logger.error("%s — %s", error_msg, model_id)
                self.circuit_breaker.record_failure(model_id)
                return GenerationResult(
                    query=query, response="", model_id=model_id, tier=tier,
                    tokens_in=0, tokens_out=0,
                    latency_ms=round(latency_ms, 1),
                    error=error_msg,
                )

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < self.retry_count:
                    wait = self._backoff(attempt)
                    logger.warning(
                        "%s, retry %d/%d in %.1fs: %s",
                        type(e).__name__, attempt + 1, self.retry_count, wait,
                        model_id,
                    )
                    if on_attempt:
                        on_attempt(attempt, type(e).__name__)
                    time.sleep(wait)
                    continue
                self.circuit_breaker.record_failure(model_id)
                return GenerationResult(
                    query=query, response="", model_id=model_id, tier=tier,
                    tokens_in=0, tokens_out=0,
                    latency_ms=round(
                        (time.perf_counter() - start) * 1000, 1
                    ),
                    error=f"{type(e).__name__}: max retries exceeded",
                )

            except Exception as e:
                logger.error("Request failed (%s): %s", model_id, e)
                self.circuit_breaker.record_failure(model_id)
                return GenerationResult(
                    query=query, response="", model_id=model_id, tier=tier,
                    tokens_in=0, tokens_out=0,
                    latency_ms=round(
                        (time.perf_counter() - start) * 1000, 1
                    ),
                    error=str(e),
                )

        # All retries exhausted
        self.circuit_breaker.record_failure(model_id)
        return GenerationResult(
            query=query, response="", model_id=model_id, tier=tier,
            tokens_in=0, tokens_out=0,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            error="Max retries exceeded",
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _backoff(
        self, attempt: int, resp: Optional[requests.Response] = None
    ) -> float:
        """Jittered exponential backoff with Retry-After header support."""
        if resp and resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), self.max_delay)
                except ValueError:
                    pass
        delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.5 * delay)
        return min(delay + jitter, self.max_delay)

    def list_available_models(self) -> list[dict]:
        """Fetch available models from OpenRouter."""
        try:
            resp = self.session.get(
                f"{self.base_url}/models", timeout=10
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.warning("Failed to list models: %s", e)
        return []


# ─── Auto-register ────────────────────────────────────────────────────────

register_provider("openrouter", OpenRouterProvider)
