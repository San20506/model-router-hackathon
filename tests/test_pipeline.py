"""Tests for RoutingPipeline (no API key — tests only local logic)."""

from model_router.pipeline import RoutingPipeline
from model_router.models import RouteRequest
from model_router.config import RouterConfig


def _make_config():
    return RouterConfig(
        openrouter_api_key="test-key",
        cascade_enabled=False,
    )


def test_pipeline_classifies_without_api():
    """Pipeline runs classification and intent/decomposition without API calls."""
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("Python is a programming language", source="test")
    pipe.sot.add_document("Paris is the capital of France", source="test")

    req = RouteRequest(query="What is the capital of France?")
    result = pipe.route(req)

    # Classification should work
    assert result.classification.complexity in ("close", "moderate", "distant")
    assert result.classification.task_label in ("grounded", "web_search", "deep_reasoning")
    assert result.classification.confidence > 0

    # Routing decision should be set
    assert result.routing.tier in ("fast", "thinking", "deep")
    assert result.routing.model_name
    assert result.routing.model_id

    # Intent should be detected
    assert result.intent is not None
    assert result.intent.intent == "question"
    assert result.intent.confidence > 0

    # Decomposition should be present (simple query, no sub-tasks)
    assert result.decomposition is not None
    assert result.decomposition.has_sub_tasks is False
    assert result.decomposition.needs_reasoning is False


def test_pipeline_decomposition_flags():
    """Pipeline marks reasoning/vision flags from decomposition."""
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("test content", source="test")

    # Multi-part query that requires reasoning
    req = RouteRequest(
        query="Write a Python function and then explain how it works"
    )
    result = pipe.route(req)
    assert result.decomposition is not None
    assert result.decomposition.needs_reasoning is True
    assert result.routing.tier in ("thinking", "deep")


def test_pipeline_force_tier():
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("test content", source="test")

    req = RouteRequest(query="hello", force_tier="deep")
    result = pipe.route(req)
    assert result.routing.tier == "deep"


def test_pipeline_history():
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("test content", source="test")

    pipe.route(RouteRequest(query="hello"))
    pipe.route(RouteRequest(query="world"))

    stats = pipe.get_stats()
    assert stats["total_routes"] == 2

    history = pipe.get_history(limit=10)
    assert len(history) == 2


def test_pipeline_stats_with_zero_history():
    config = _make_config()
    pipe = RoutingPipeline(config)
    stats = pipe.get_stats()
    assert stats["total"] == 0


def test_pipeline_listener_fires():
    """Registered listener callback receives route response."""
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("test content", source="test")

    received = []

    def listener(response):
        received.append(response)

    pipe.on_route(listener)
    pipe.route(RouteRequest(query="hello"))
    assert len(received) == 1
    assert received[0].query == "hello"


def test_pipeline_listener_exception_does_not_crash():
    """Listener that raises should not break the pipeline."""
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("test content", source="test")

    def broken_listener(response):
        raise RuntimeError("boom")

    pipe.on_route(broken_listener)
    # Should not raise
    result = pipe.route(RouteRequest(query="hello"))
    assert result is not None


def test_pipeline_history_capped():
    """History list is capped at 1000 entries."""
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("test content", source="test")

    for i in range(1005):
        pipe.route(RouteRequest(query=f"query-{i}"))

    stats = pipe.get_stats()
    assert stats["total_routes"] == 1000
    assert len(pipe.history) == 1000


def test_pipeline_empty_sot():
    config = _make_config()
    pipe = RoutingPipeline(config)
    # No documents seeded

    # Trivial intent with empty SOT → close (not distant)
    req = RouteRequest(query="hi")
    result = pipe.route(req)
    assert result.classification.complexity == "close"
    assert result.source_query.min_distance >= 0.9

    # Non-trivial intent with empty SOT → moderate (not deep)
    req2 = RouteRequest(query="explain quantum computing")
    result2 = pipe.route(req2)
    assert result2.classification.complexity == "moderate"
