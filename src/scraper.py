"""Scrape OpenRouter model catalog."""

import logging
import requests
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global cache of model metrics
_MODEL_CACHE: Dict[str, Any] = {}
CACHE_FILE = "data/openrouter_models.json"

def fetch_openrouter_models() -> Dict[str, Any]:
    """Fetch all models from OpenRouter API, store in memory, and persist to disk."""
    logger.info("Fetching OpenRouter models catalog...")
    import os
    os.makedirs("data", exist_ok=True)
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        response.raise_for_status()
        data = response.json().get("data", [])
        
        cache = {}
        for model in data:
            model_id = model.get("id")
            if not model_id:
                continue
                
            pricing = model.get("pricing", {})
            try:
                input_cost = float(pricing.get("prompt", 0)) * 1_000_000
                output_cost = float(pricing.get("completion", 0)) * 1_000_000
            except (ValueError, TypeError):
                input_cost = 0.0
                output_cost = 0.0
                
            cache[model_id] = {
                "id": model_id,
                "name": model.get("name", model_id),
                "context_length": model.get("context_length", 0),
                "input_cost": input_cost,
                "output_cost": output_cost,
                "architecture": model.get("architecture", {})
            }
            
        global _MODEL_CACHE
        _MODEL_CACHE = cache
        
        # Persist to disk
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
            
        logger.info(f"Successfully cached {len(cache)} OpenRouter models to {CACHE_FILE}.")
        return cache
        
    except Exception as e:
        logger.error(f"Failed to fetch OpenRouter models: {e}")
        return _MODEL_CACHE

def get_model_metrics() -> Dict[str, Any]:
    """Return the cached model metrics, loading from disk if necessary."""
    global _MODEL_CACHE
    if not _MODEL_CACHE:
        import os
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    _MODEL_CACHE = json.load(f)
                logger.info(f"Loaded {_MODEL_CACHE} models from {CACHE_FILE}.")
            except Exception as e:
                logger.error(f"Failed to load cache from {CACHE_FILE}: {e}")
    return _MODEL_CACHE

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    models = fetch_openrouter_models()
    print(f"\n[Scraper] Successfully scraped {len(models)} models from OpenRouter.")
    if models:
        sample_model = list(models.values())[0]
        print(f"[Scraper] Sample Model Data ({sample_model['id']}):")
        print(json.dumps(sample_model, indent=2))
