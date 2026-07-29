import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# Add project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

# Add MOIP path
moip_path = os.path.join(base_dir, "..", "Multi-Objective-Itinerary-Planning", "Multi-Objective-Itinerary-Planning--master", "src")
sys.path.append(os.path.normpath(moip_path))

from src.agents.v2.graph import app as rag_graph
from src.core.logger import get_logger

logger = get_logger("api")

app = FastAPI(title="Sri Lanka Tourism RAG API")

class ChatRequest(BaseModel):
    question: str

class PlanItineraryRequest(BaseModel):
    start_place: str
    days: int
    hours_per_day: float
    max_places: int
    must_visit: str
    attraction_priority: int
    budget_priority: int
    time_priority: int
    popular_priority: int
    question: str

# Lazy-loaded MOIP graph
G_MOIP = None
def get_moip_graph():
    global G_MOIP
    if G_MOIP is None:
        from build_graph import build_graph
        from trips import initialize_pheromones_from_past_trips
        moip_data_dir = os.path.normpath(os.path.join(moip_path, "..", "data"))
        G = build_graph(
            os.path.join(moip_data_dir, "pois.csv"),
            os.path.join(moip_data_dir, "route_distance_matrix.csv"),
            os.path.join(moip_data_dir, "route_duration_matrix.csv")
        )
        G, _, _ = initialize_pheromones_from_past_trips(G, os.path.join(moip_data_dir, "past_trips_dataset.csv"), boost=5.0)
        G_MOIP = G
    return G_MOIP

@app.post("/api/chat")
async def chat(request: ChatRequest):
    logger.info(f"Received question: {request.question}")
    try:
        inputs = {"question": request.question}
        # Run the graph
        result = rag_graph.invoke(inputs)
        
        # Check intent to see if it's itinerary planning without a form submission
        intent = result.get("query_intent", {}).get("intent_type")
        if intent == "itinerary_planning" and not result.get("moip_result"):
            return {
                "response": "It looks like you want to plan a trip! Please fill out the details below so I can create a customized itinerary for you.",
                "is_itinerary_request": True
            }
        
        # Extract the final response and evidence
        response_text = result.get("evidenced_response") or result.get("draft") or "No response generated."
        evidence = result.get("structured_facts") or []
        
        return {
            "response": response_text,
            "evidence": evidence,
            "is_itinerary_request": False
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plan_itinerary")
async def plan_itinerary(request: PlanItineraryRequest):
    logger.info(f"Planning itinerary from {request.start_place} for {request.days} days.")
    try:
        from aco import run_aco
        from evaluation import evaluate_and_rank, split_multiday
        from utils import split_text_places, fuzzy_match_place, normalize_weights
        
        G = get_moip_graph()
        places = sorted(list(G.nodes()))
        
        start = request.start_place
        if start not in places and places:
            start = places[0] # Fallback
            
        typed_places = [fuzzy_match_place(p, places) for p in split_text_places(request.must_visit)]
        must_visit = list(dict.fromkeys([p for p in typed_places if p]))
        
        selected = list(dict.fromkeys([start] + must_visit))
        
        # For simplicity, we skip weather filtering here and assume all nodes are accepted
        accepted = list(G.nodes())
        if start not in accepted:
            accepted.insert(0, start)
            
        preferences = {
            "attraction": request.attraction_priority,
            "budget": request.budget_priority,
            "time": request.time_priority,
            "popular": request.popular_priority,
        }
        weights = normalize_weights(preferences)
        max_total_time_min = int(request.days) * float(request.hours_per_day) * 60
        route_limit = max(int(request.max_places), len(list(dict.fromkeys([start] + must_visit))))
        
        # Run ACO
        all_routes, _ = run_aco(
            G, start, accepted, preferences, 50, 15, max_total_time_min, route_limit, must_visit, seed=42, copy_pheromone=True
        )
        
        if not all_routes:
            raise Exception("No feasible route found for the given constraints.")
            
        moip_data_dir = os.path.normpath(os.path.join(moip_path, "..", "data"))
        ranked = evaluate_and_rank(all_routes, weights, os.path.join(moip_data_dir, "past_trips_dataset.csv"), use_pareto=True, min_results=1)
        
        if not ranked:
            raise Exception("Failed to rank routes.")
            
        best = ranked[0]
        itinerary = split_multiday(G, best["route"], request.days, request.hours_per_day)
        
        # Attach coordinates
        import pandas as pd
        pois_df = pd.read_csv(os.path.join(moip_data_dir, "pois.csv"))
        pois_df.set_index("name", inplace=True)
        
        for day in itinerary:
            detailed_places = []
            for place_name in day.get("places", []):
                lat, lon = None, None
                if place_name in pois_df.index:
                    lat = float(pois_df.loc[place_name, "latitude"])
                    lon = float(pois_df.loc[place_name, "longitude"])
                detailed_places.append({
                    "name": place_name,
                    "lat": lat,
                    "lon": lon
                })
            day["detailed_places"] = detailed_places
        
        moip_result = {
            "best": best,
            "itinerary": itinerary
        }
        
        # Now pass this back into the LangGraph to format it
        inputs = {
            "question": request.question,
            "query_intent": {"intent_type": "itinerary_planning"},
            "moip_result": moip_result
        }
        result = rag_graph.invoke(inputs)
        response_text = result.get("evidenced_response") or "Successfully generated itinerary."
        
        return {
            "response": response_text,
            "evidence": [],
            "is_itinerary_request": False,
            "moip_result": result.get("moip_result", moip_result)
        }

    except Exception as e:
        import traceback
        logger.error(f"Error planning itinerary: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files from ui directory
ui_dir = os.path.join(base_dir, "ui")
app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
