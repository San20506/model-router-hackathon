"""Model Router Dashboard — FastAPI + WebSocket real-time UI.

Standalone web app for testing and monitoring the routing pipeline.
"""

import csv
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse

# Load .env before anything else reads config
load_dotenv()

from model_router.config import get_config
from model_router.models import RouteRequest
from model_router.pipeline import RoutingPipeline
from model_router.constants import ALL_MODELS, TIER_MODELS, UTILITY_MODELS
from model_router.store import SourceOfTruth

logger = logging.getLogger(__name__)

config = get_config()
pipeline = RoutingPipeline(config)

# WebSocket connections for live dashboard
_websockets: list[WebSocket] = []
_models_cache_lock = Lock()
_models_cache: dict[str, object] = {"timestamp": 0.0, "available_ids": set()}


def _get_live_available_model_ids(ttl_seconds: int = 60) -> set[str]:
    now = time.time()
    with _models_cache_lock:
        cached_timestamp = float(_models_cache["timestamp"])
        cached_ids = set(_models_cache["available_ids"])
        if cached_ids and now - cached_timestamp < ttl_seconds:
            return cached_ids

    available_ids: set[str] = set()
    try:
        available_ids = {
            model.get("id")
            for model in pipeline.client.list_available()
            if model.get("id")
        }
    except Exception:
        logger.exception("Failed to refresh OpenRouter model list")

    with _models_cache_lock:
        if available_ids:
            _models_cache["available_ids"] = available_ids
        _models_cache["timestamp"] = now
        return set(_models_cache["available_ids"])


def seed_sot_from_csv(csv_path: str, max_rows: int = 500):
    """Seed the Source of Truth from a CSV with question/answer columns."""
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Seed CSV not found: %s", csv_path)
        return 0
    sot = pipeline.sot  # direct access for seeding
    if sot.count() > 0:
        logger.info("SOT already has %s docs, skipping seed", sot.count())
        return sot.count()
    count = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            q = row.get("question", row.get("Question", ""))
            a = row.get("answer", row.get("Answer", ""))
            if q and a:
                sot.add_document(f"Q: {q}\nA: {a}", source="alexa-qa")
                count += 1
    logger.info("Seeded %s docs into SOT from %s", count, csv_path)
    return count


def broadcast(route_response):
    """Push route decision to all connected dashboard clients."""
    meta = route_response.classification.metadata or {}
    data = {
        "type": "route",
        "query": route_response.query,
        "response_preview": route_response.response[:200],
        "complexity": route_response.classification.complexity,
        "task": route_response.classification.task_label,
        "confidence": route_response.classification.confidence,
        "method": route_response.classification.method,
        "tier": route_response.routing.tier,
        "model_id": route_response.routing.model_id,
        "model_name": route_response.routing.model_name,
        "reason": route_response.routing.reason,
        "tokens_in": route_response.generation.tokens_in,
        "tokens_out": route_response.generation.tokens_out,
        "latency_ms": route_response.generation.latency_ms,
        "escalated": route_response.generation.cascade_escalated,
        "error": route_response.generation.error,
        # Heatmap metadata
        "match_density": meta.get("match_density"),
        "coverage": meta.get("coverage"),
        "concentration": meta.get("concentration"),
        "matched_words": meta.get("matched_words"),
        "query_words": meta.get("query_words"),
        "docs_hit": meta.get("docs_hit"),
        "stats": pipeline.get_stats(),
    }
    for ws in _websockets[:]:
        try:
            import anyio
            anyio.from_thread.run(ws.send_json, data)
        except Exception:
            _websockets.remove(ws)


pipeline.on_route(broadcast)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Model Router dashboard starting...")
    seed_path = os.getenv("SOT_SEED_CSV", "data/alexa_qa/train.csv")
    seed_sot_from_csv(
        seed_path,
        max_rows=int(os.getenv("SOT_SEED_MAX", "500")),
    )
    yield
    logger.info("Model Router dashboard stopping.")


app = FastAPI(title="Model Router Dashboard", lifespan=lifespan)


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    from pathlib import Path
    return (Path(__file__).resolve().parent / "template.html").read_text()


@app.post("/route")
async def route_query(req: RouteRequest) -> dict:
    result = pipeline.route(req)
    return _response_to_dict(result)


@app.get("/history")
async def get_history(limit: int = Query(50, ge=1, le=200)):
    return [_response_to_dict(r) for r in pipeline.get_history(limit=limit)]


@app.get("/stats")
async def get_stats():
    return pipeline.get_stats()


@app.get("/models")
async def get_models():
    live_available_ids = _get_live_available_model_ids()

    def _annotate(models):
        annotated = []
        for model in models:
            annotated.append({
                "name": model.name,
                "id": model.openrouter_id,
                "params": model.total_params_b,
                "active": model.active_params_b,
                "available": model.openrouter_id in live_available_ids,
            })
        return annotated

    return {
        "utility": _annotate(UTILITY_MODELS),
        "fast": _annotate(TIER_MODELS.get("fast", [])),
        "thinking": _annotate(TIER_MODELS.get("thinking", [])),
        "deep": _annotate(TIER_MODELS.get("deep", [])),
        "count": len(ALL_MODELS),
        "available_count": len(live_available_ids),
    }


# =============================================================================
# WebSocket — Real-time route updates
# =============================================================================


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _websockets.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _websockets.remove(ws)
    except Exception:
        if ws in _websockets:
            _websockets.remove(ws)


def _response_to_dict(r) -> dict:
    meta = r.classification.metadata or {}
    return {
        "query": r.query,
        "response": r.response[:500],
        "complexity": r.classification.complexity,
        "task": r.classification.task_label,
        "confidence": r.classification.confidence,
        "method": r.classification.method,
        "tier": r.routing.tier,
        "model_id": r.routing.model_id,
        "model_name": r.routing.model_name,
        "reason": r.routing.reason,
        "tokens_in": r.generation.tokens_in,
        "tokens_out": r.generation.tokens_out,
        "latency_ms": r.generation.latency_ms,
        "escalated": r.generation.cascade_escalated,
        "error": r.generation.error,
        "timestamp": r.generation.timestamp.isoformat(),
        # Heatmap
        "match_density": meta.get("match_density"),
        "coverage": meta.get("coverage"),
        "concentration": meta.get("concentration"),
        "matched_words": meta.get("matched_words"),
        "query_words": meta.get("query_words"),
        "docs_hit": meta.get("docs_hit"),
    }


def run_dashboard():
    import uvicorn
    uvicorn.run(
        app,
        host=config.dashboard_host,
        port=config.dashboard_port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    run_dashboard()


# =============================================================================
# Dashboard HTML
# =============================================================================

HERE = Path(__file__).resolve().parent
DASHBOARD_HTML = (HERE / "template.html").read_text()

