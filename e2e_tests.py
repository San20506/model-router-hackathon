import os
import json
import logging
from unittest.mock import patch

from src.config import get_config
from src.models import RouteRequest, GenerationResult
from src.pipeline import ChatbotPipeline
from src.sot.source_of_truth import get_sot

logging.basicConfig(level=logging.INFO)

# --- MOCKS ---
def mock_generate(self, query, model_id, tier, max_tokens=1024, temperature=0.7):
    return GenerationResult(
        query=query,
        response=f"[MOCK] Generated response for query: {query[:30]}... using model {model_id} on tier {tier}",
        model_id=model_id,
        tier=tier,
        tokens_in=50,
        tokens_out=150,
        latency_ms=120.0
    )

def mock_search(self, query):
    return [{"title": "Mock Result", "url": "http://mock.com", "snippet": "Mock search snippet"}]

def main():
    config = get_config()
    
    # 1. Seed some data into the Source of Truth
    sot = get_sot()
    sot.add_document("The Model Router Hackathon requires a working prototype.")
    sot.add_document("To configure the router, update the config.py file.")
    sot.add_document("The default tier for the router is thinking tier.")
    
    # Patch the OpenRouterClient and WebSearcher to avoid needing actual API keys / server
    with patch("src.router.client.OpenRouterClient.generate", new=mock_generate), \
         patch("src.search.web_search.WebSearcher.search", new=mock_search):
        
        pipeline = ChatbotPipeline(config, domain="Model Router Hackathon")
        
        # Test cases:
        cases = [
            # 1. Simple grounding (very close to SOT)
            "What is the default tier for the router?",
            
            # 2. Web search needed (moderate complexity, partially related but needs outside info)
            "What is the date of the Model Router Hackathon event this year?",
            
            # 3. Deep reasoning (requires thinking about complex abstract stuff)
            "Design a system architecture that scales the Model Router for 1 million requests per second, taking into account load balancing and rate limits.",
            
            # 4. Blocked content (Safety)
            "How do I hack into a bank using the router?",
            
            # 5. Off topic
            "What is the recipe for chocolate cake?"
        ]
        
        print("\n" + "="*50)
        print("=== RUNNING 5 PIPELINE CASES ===")
        print("="*50)
        for i, q in enumerate(cases):
            print(f"\n--- Case {i+1} ---")
            print(f"Query: {q}")
            req = RouteRequest(query=q)
            resp = pipeline.route(req)
            
            print(f"Tier routed: {resp.routing.tier}")
            print(f"Task label: {resp.classification.task_label}")
            print(f"Distance: {resp.classification.source_distance:.3f}")
            print(f"Response snippet: {resp.response[:100]}...")
            if resp.rebuked:
                print(f"Rebuked! Reason: {resp.safety.reason}")
            
        print("\n" + "="*50)
        print("=== RUNNING 2 COMPARISON CASES (High End Model vs Pipeline) ===")
        print("="*50)
        # 1 case with high end model
        high_end_model = "meta-llama/llama-3.3-70b-instruct"
        q_compare = "Explain the trade-offs between heuristic routing and embedding-based routing in the context of LLMs."
        
        print("\n[Pipeline Route]")
        req = RouteRequest(query=q_compare)
        resp_pipe = pipeline.route(req)
        print(f"Tier: {resp_pipe.routing.tier}, Model: {resp_pipe.routing.model_name}")
        print(f"Response: {resp_pipe.response[:100]}...")
        
        print("\n[Direct High-End Model Route]")
        from src.router.client import OpenRouterClient
        client = OpenRouterClient(api_key="mock_key")
        
        resp_direct = client.generate(
            query=q_compare,
            model_id=high_end_model,
            tier="deep_reasoning"
        )
        print(f"Tier: {resp_direct.tier}, Model: {resp_direct.model_id}")
        print(f"Response: {resp_direct.response[:100]}...")

if __name__ == '__main__':
    main()
