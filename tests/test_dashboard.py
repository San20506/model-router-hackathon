"""Tests for the dashboard integration payloads."""

import asyncio

from dashboard import app as dashboard_app
from model_router.config import RouterConfig
from model_router.constants import UTILITY_MODELS
from model_router.models import RouteRequest
from model_router.pipeline import RoutingPipeline


def _make_config():
    return RouterConfig(
        openrouter_api_key="test-key",
        cascade_enabled=False,
    )


def test_pipeline_stats_expose_dashboard_aliases():
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("Paris is the capital of France", source="test")

    pipe.route(RouteRequest(query="What is the capital of France?"))

    stats = pipe.get_stats()
    assert stats["total"] == 1
    assert stats["total_routes"] == 1
    assert stats["models_used"]


def test_dashboard_models_include_live_availability(monkeypatch):
    monkeypatch.setattr(
        dashboard_app,
        "_get_live_available_model_ids",
        lambda ttl_seconds=60: {UTILITY_MODELS[0].openrouter_id},
    )

    payload = asyncio.run(dashboard_app.get_models())

    assert payload["count"] >= 1
    assert payload["available_count"] == 1
    assert payload["utility"][0]["available"] is True