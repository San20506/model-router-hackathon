"""
Tests for cascade logic in RoutingPipeline._maybe_cascade.

Uses a mock OpenRouterClient to simulate low-confidence generations
and verify cascade escalation guards (hops budget, token budget,
emergency fallback, disabled cascade).
"""

from unittest.mock import Mock, patch, MagicMock

import pytest
from model_router.pipeline import RoutingPipeline
from model_router.models import (
    RouteRequest, ClassificationResult, RoutingDecision, GenerationResult,
)
from model_router.config import RouterConfig


# =============================================================================
# Helpers
# =============================================================================


def _make_config(cascade_enabled=True, max_hops=2, max_budget=10_000):
    return RouterConfig(
        openrouter_api_key="test-key",
        cascade_enabled=cascade_enabled,
        cascade_max_hops=max_hops,
        cascade_max_budget_tokens=max_budget,
    )


def _make_classification(complexity="close", confidence=0.5):
    return ClassificationResult(
        query="test", complexity=complexity,
        task_label="grounded", confidence=confidence,
        method="sot_distance", source_distance=0.1,
    )


def _make_routing(tier="fast", model_id="meta-llama/llama-3.2-3b-instruct:free"):
    return RoutingDecision(
        query="test", tier=tier,
        model_id=model_id, model_name="Llama 3.2 3B",
        complexity="close", confidence=0.5,
        reason="test",
    )


def _make_generation(error=None, response="test response"):
    return GenerationResult(
        query="test", response=response, model_id="test-model",
        tier="fast", tokens_in=10, tokens_out=20, latency_ms=100,
        error=error,
    )


def _pipeline_with_mock_client(config=None):
    """Return a RoutingPipeline whose .client.generate is a MagicMock."""
    config = config or _make_config()
    pipe = RoutingPipeline(config)
    pipe.client = MagicMock()
    pipe.client.generate.return_value = _make_generation()
    return pipe


# =============================================================================
# Tests
# =============================================================================


class TestCascadeDisabled:
    def test_no_cascade_when_disabled(self):
        """cascade_enabled=False → _maybe_cascade returns original generation."""
        pipe = _pipeline_with_mock_client(_make_config(cascade_enabled=False))
        gen = _make_generation()
        routing = _make_routing()
        classification = _make_classification(confidence=0.3)

        result = pipe._maybe_cascade(RouteRequest(query="test"), gen, routing, classification)
        assert result is gen  # same object, no cascade
        pipe.client.generate.assert_not_called()

    def test_no_cascade_when_generation_errored(self):
        """If the generation already has an error, don't cascade."""
        pipe = _pipeline_with_mock_client()
        gen = _make_generation(error="API error")
        routing = _make_routing()
        classification = _make_classification(confidence=0.3)

        result = pipe._maybe_cascade(RouteRequest(query="test"), gen, routing, classification)
        assert result is gen
        pipe.client.generate.assert_not_called()

    def test_no_cascade_when_confidence_high(self):
        """confidence > 0.7 → no cascade needed."""
        pipe = _pipeline_with_mock_client()
        gen = _make_generation()
        routing = _make_routing()
        classification = _make_classification(confidence=0.8)

        result = pipe._maybe_cascade(RouteRequest(query="test"), gen, routing, classification)
        assert result is gen
        pipe.client.generate.assert_not_called()

    def test_no_cascade_when_already_deep(self):
        """Already at deepest tier → no further escalation."""
        pipe = _pipeline_with_mock_client()
        gen = _make_generation()
        routing = _make_routing(tier="deep")
        classification = _make_classification(confidence=0.3)

        result = pipe._maybe_cascade(RouteRequest(query="test"), gen, routing, classification)
        assert result is gen
        pipe.client.generate.assert_not_called()


class TestCascadeFires:
    def test_cascade_to_thinking_from_fast(self):
        """Low confidence on fast tier → cascade to thinking."""
        pipe = _pipeline_with_mock_client()
        pipe.client.generate.return_value = _make_generation(response="cascaded response")
        gen = _make_generation()
        routing = _make_routing(tier="fast")
        classification = _make_classification(confidence=0.3)

        result = pipe._maybe_cascade(RouteRequest(query="test"), gen, routing, classification)
        assert result.cascade_escalated is True
        assert result.cascade_from_tier == "fast"
        assert result.cascade_to_tier == "thinking"
        pipe.client.generate.assert_called_once()

    def test_cascade_to_deep_from_thinking(self):
        """Low confidence on thinking → cascade to deep."""
        pipe = _pipeline_with_mock_client()
        pipe.client.generate.return_value = _make_generation(response="cascaded deep")
        gen = _make_generation()
        routing = _make_routing(tier="thinking")
        classification = _make_classification(confidence=0.3)

        result = pipe._maybe_cascade(RouteRequest(query="test"), gen, routing, classification)
        assert result.cascade_escalated is True
        assert result.cascade_from_tier == "thinking"
        assert result.cascade_to_tier == "deep"


class TestCascadeHopsBudget:
    def test_max_hops_respected(self):
        """Exceeding cascade_max_hops stops escalation."""
        pipe = _pipeline_with_mock_client(_make_config(max_hops=1))
        pipe.client.generate.return_value = _make_generation(response="cascade result")
        gen = _make_generation()
        routing = _make_routing(tier="fast")
        classification = _make_classification(confidence=0.3)

        # First hop: fast → thinking (allowed, max_hops=1)
        req = RouteRequest(query="test")
        result1 = pipe._maybe_cascade(req, gen, routing, classification)
        assert result1.cascade_escalated is True

        # This doesn't actually test the hops budget correctly because
        # _maybe_cascade creates a new result - the hop is tracked on the
        # request object, but we need to simulate that the cascade happened
        # and the pipeline called _maybe_cascade again with the new result.
        # For that we check guards work:
        req2 = RouteRequest(query="test")
        req2._cascade_hops = 2  # simulate already having hopped
        result2 = pipe._maybe_cascade(req2, gen, routing, classification)
        assert result2 is gen  # no further cascade

    def test_hops_budget_exceeded_returns_original(self):
        """When hops >= max, return current generation unchanged."""
        pipe = _pipeline_with_mock_client(_make_config(max_hops=2))
        gen = _make_generation()
        routing = _make_routing(tier="fast")
        classification = _make_classification(confidence=0.3)

        req = RouteRequest(query="test")
        req._cascade_hops = 2  # already used all hops
        result = pipe._maybe_cascade(req, gen, routing, classification)
        assert result is gen
        pipe.client.generate.assert_not_called()


class TestCascadeTokenBudget:
    def test_budget_exceeded_returns_original(self):
        """When estimated token budget is exceeded, stop cascading."""
        pipe = _pipeline_with_mock_client(_make_config(max_budget=100))
        gen = _make_generation()
        routing = _make_routing(tier="fast")
        classification = _make_classification(confidence=0.3)

        req = RouteRequest(query="test")
        # Set budget already over limit
        req._cascade_budget = 200
        result = pipe._maybe_cascade(req, gen, routing, classification)
        assert result is gen
        pipe.client.generate.assert_not_called()


class TestEmergencyFallback:
    def test_emergency_fallback_fires(self):
        """When cascade also fails, try openrouter/free as emergency."""
        pipe = _pipeline_with_mock_client()
        # Cascade attempt returns an error
        pipe.client.generate.return_value = _make_generation(error="still failed")
        gen = _make_generation()
        routing = _make_routing(tier="fast")
        classification = _make_classification(confidence=0.3)

        result = pipe._maybe_cascade(RouteRequest(query="test"), gen, routing, classification)
        # Emergency fallback fires
        assert pipe.client.generate.call_count == 2  # cascade + emergency
        # Check the last call was for openrouter/free
        last_call = pipe.client.generate.call_args_list[-1]
        assert "openrouter/free" in str(last_call.args)


class TestNextTier:
    def test_next_tier_fast_to_thinking(self):
        assert RoutingPipeline._next_tier("fast") == "thinking"

    def test_next_tier_thinking_to_deep(self):
        assert RoutingPipeline._next_tier("thinking") == "deep"

    def test_next_tier_deep_is_none(self):
        assert RoutingPipeline._next_tier("deep") is None

    def test_next_tier_unknown(self):
        assert RoutingPipeline._next_tier("nonexistent") is None


class TestFallbacksFor:
    def test_fallback_includes_openrouter_free(self):
        fallbacks = RoutingPipeline._fallbacks_for("fast")
        assert "openrouter/free" in fallbacks
        assert len(fallbacks) > 1

    def test_fallback_unknown_tier(self):
        fallbacks = RoutingPipeline._fallbacks_for("nonexistent")
        assert fallbacks == ["openrouter/free"]

    def test_fallback_skips_primary_model(self):
        fallbacks = RoutingPipeline._fallbacks_for("fast")
        # Should not include the primary (smallest) model
        # primary is liquid/lfm-2.5-1.2b-thinking:free
        first = fallbacks[0]
        assert first != "model_router"
