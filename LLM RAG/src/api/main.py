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

# Add MOIP path.
# The optimiser project has been re-laid-out at least once (it used to sit under
# an extra "Multi-Objective-Itinerary-Planning--master" folder), so resolve it by
# looking for the module we actually import rather than hard-coding one shape.
_MOIP_ROOT = os.path.normpath(os.path.join(base_dir, "..", "Multi-Objective-Itinerary-Planning"))
_MOIP_CANDIDATES = [
    os.path.join(_MOIP_ROOT, "src"),                                              # current layout
    os.path.join(_MOIP_ROOT, "Multi-Objective-Itinerary-Planning--master", "src"),  # previous layout
]

moip_path = next(
    (os.path.normpath(p) for p in _MOIP_CANDIDATES
     if os.path.isfile(os.path.join(p, "aco.py"))),
    None,
)
if moip_path is None:
    raise RuntimeError(
        "Could not locate the Multi-Objective-Itinerary-Planning source folder. "
        "Looked for aco.py in:\n  " + "\n  ".join(os.path.normpath(p) for p in _MOIP_CANDIDATES)
    )
sys.path.append(moip_path)

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
    # Mirrors the Streamlit app's controls. All optional so older payloads work.
    use_all_pois: bool = False
    num_ants: int = 80
    iterations: int = 30
    trip_start_date: Optional[str] = None
    avoid_bad_weather: bool = True

# Leave empty to skip weather entirely (same behaviour as src/app.py).
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")

MOIP_DATA_DIR = os.path.normpath(os.path.join(moip_path, "..", "data"))
PAST_TRIPS_FILE = os.path.join(MOIP_DATA_DIR, "past_trips_dataset.csv")

# Lazy-loaded MOIP graph
G_MOIP = None
def get_moip_graph():
    global G_MOIP
    if G_MOIP is None:
        from build_graph import build_graph
        from trips import initialize_pheromones_from_past_trips
        G = build_graph(
            os.path.join(MOIP_DATA_DIR, "pois.csv"),
            os.path.join(MOIP_DATA_DIR, "route_distance_matrix.csv"),
            os.path.join(MOIP_DATA_DIR, "route_duration_matrix.csv")
        )
        G, _, _ = initialize_pheromones_from_past_trips(G, PAST_TRIPS_FILE, boost=5.0)
        G_MOIP = G
    return G_MOIP


# ---------------------------------------------------------------------------
# Presentation helpers — ported verbatim from the MOIP Streamlit app (src/app.py)
# so the web planner renders exactly what that app renders.
# ---------------------------------------------------------------------------

def place_label(G, place):
    """'Colombo Lotus Tower (Landmark / Tower)'"""
    if place not in G.nodes:
        return place
    cat = str(G.nodes[place].get("category", "")).strip()
    return f"{place} ({cat})" if cat else place


def format_full_route(G, route):
    return " → ".join(place_label(G, p) for p in route)


def format_day_places(G, places):
    """1 place -> 'X (cat)'; 2+ -> 'start-First → ... → end-Last'"""
    if not places:
        return "(no places)"
    labels = [place_label(G, p) for p in places]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"start-{labels[0]} → end-{labels[1]}"
    mid = " → ".join(labels[1:-1])
    return f"start-{labels[0]} → {mid} → end-{labels[-1]}"


def format_day_route_line(G, places):
    return " → ".join(place_label(G, p) for p in places) if places else ""


def refresh_route_metrics(G, route_result):
    from aco import route_metrics
    refreshed = route_metrics(G, route_result["route"])
    if refreshed is None:
        return None
    return {**route_result, **refreshed}


def score_row(route_result, rank=None):
    row = {}
    if rank is not None:
        row["Rank"] = rank
    row.update({
        "Overall Score": round(route_result.get("preference_score", 0), 4),
        "Satisfaction": round(route_result.get("total_satisfaction", 0), 2),
        "Cost (LKR)": round(route_result.get("total_cost", 0), 0),
        "Time (h)": round(route_result.get("total_time_min", 0) / 60, 2),
        "Distance (km)": round(route_result.get("total_distance_km", 0), 1),
        "Hist. Similarity": round(route_result.get("historical_similarity", 0), 3),
        "#Places": len(route_result.get("route") or []),
        "Route": " → ".join(route_result.get("route") or []),
    })
    return row


def json_safe(obj):
    """
    Make optimiser output JSON-encodable.

    The accuracy comparison table comes from pandas, so it carries NaN (which is
    not valid JSON) and numpy scalars (which json cannot encode at all).
    """
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:                       # numpy scalar -> python scalar
            return json_safe(obj.item())
        except Exception:
            return str(obj)
    return obj


def serialise_itinerary(G, itinerary, pois_df):
    """Day plans + the display strings and coordinates the browser needs."""
    days = []
    prev_end = None
    for day in itinerary:
        places = day.get("places") or []
        detailed = []
        for name in places:
            lat = lon = None
            if name in pois_df.index:
                lat = float(pois_df.loc[name, "latitude"])
                lon = float(pois_df.loc[name, "longitude"])
            detailed.append({
                "name": name,
                "label": place_label(G, name),
                "lat": lat,
                "lon": lon,
            })
        connected_from = None
        if prev_end and places and prev_end != places[0]:
            connected_from = f"{place_label(G, prev_end)} → {place_label(G, places[0])}"
        days.append({
            "day": day.get("day", 0),
            "places": places,
            "detailed_places": detailed,
            "legs": day.get("legs") or [],
            "time_min": day.get("time_min", 0),
            "time_h": round(day.get("time_min", 0) / 60, 2),
            "distance_km": round(day.get("distance_km", 0), 2),
            "within_limit": bool(day.get("within_limit", True)),
            "places_summary": format_day_places(G, places),
            "route_line": format_day_route_line(G, places),
            "connected_from": connected_from,
        })
        if places:
            prev_end = places[-1]
    return days

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

@app.get("/api/pois")
async def list_pois():
    """
    POI list for the Start-place dropdown.

    The old free-text field silently fell back to places[0] ("Aberdeen Falls")
    whenever the typed name didn't match a node exactly, which is why the start
    place looked like it was being ignored. The Streamlit app uses a selectbox;
    this endpoint gives the web UI the same guarantee.
    """
    try:
        G = get_moip_graph()
        places = sorted(list(G.nodes()))
        return {
            "count": len(places),
            "default": "Colombo Lotus Tower" if "Colombo Lotus Tower" in places else (places[0] if places else None),
            "places": [
                {"name": p, "category": str(G.nodes[p].get("category", "")).strip(), "label": place_label(G, p)}
                for p in places
            ],
        }
    except Exception as e:
        logger.error(f"Error listing POIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/plan_itinerary")
async def plan_itinerary(request: PlanItineraryRequest):
    """
    Full MOIP pipeline, mirroring src/app.py (the Streamlit optimiser) so the web
    planner shows the same results: subset selection, specialised ACO runs, route
    variants, top-5 ranking, per-day plans, and exhaustive accuracy evaluation.
    """
    logger.info(f"Planning itinerary from {request.start_place} for {request.days} days.")
    try:
        import pandas as pd
        from aco import (
            run_aco,
            optimize_required_route,
            generate_diverse_required_routes,
            generate_route_variants,
            generate_multiple_subset_options,
        )
        from evaluation import evaluate_and_rank, split_multiday
        from utils import split_text_places, fuzzy_match_place, normalize_weights
        from scoring import compute_detailed_scores, explain_route
        from accuracy_ui import compute_accuracy_for_input

        G = get_moip_graph()
        places = sorted(list(G.nodes()))

        # --- start place: validate instead of silently falling back -------
        start = (request.start_place or "").strip()
        if start not in places:
            try:
                start = fuzzy_match_place(start, places)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Start place '{request.start_place}' is not a known POI. "
                           f"Pick one from the dropdown.",
                )

        # --- candidates / must-visit --------------------------------------
        try:
            typed_places = [fuzzy_match_place(p, places) for p in split_text_places(request.must_visit)]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        typed_places = [p for p in typed_places if p]

        if request.use_all_pois or not typed_places:
            must_visit = []
            candidate_pool = list(places)
            original_must_visit = []
        else:
            must_visit = list(dict.fromkeys([p for p in typed_places if p != start]))
            original_must_visit = list(must_visit)
            candidate_pool = list(dict.fromkeys([start] + must_visit))

        preferences = {
            "attraction": request.attraction_priority,
            "budget": request.budget_priority,
            "time": request.time_priority,
            "popular": request.popular_priority,
        }
        weights = normalize_weights(preferences)
        max_total_time_min = int(request.days) * float(request.hours_per_day) * 60
        num_ants = int(request.num_ants)
        iterations = int(request.iterations)

        # Weather is skipped unless OPENWEATHERMAP_API_KEY is set (same as app.py).
        removed_weather = []

        # --- subset selection when everything doesn't fit -----------------
        subset_options, subset_info = [], None
        if must_visit:
            subset_options = generate_multiple_subset_options(
                G, start, must_visit, max_total_time_min, weights, max_options=5
            )
            if not subset_options:
                raise HTTPException(
                    status_code=400,
                    detail="No places fit in the available time. Increase days/hours or choose closer places.",
                )
            primary = subset_options[0]
            must_visit = primary["selected"]
            subset_info = {
                "fitted_all": primary.get("fitted_all", False),
                "selected": primary["selected"],
                "dropped": primary["dropped"],
                "estimated_time_h": primary.get("estimated_time_h"),
                "budget_h": primary.get("budget_h"),
                "full_needed_h": primary.get("full_needed_h"),
                "reason": (
                    "All places fit the day budget."
                    if primary.get("fitted_all")
                    else f"Not all places fit {primary.get('budget_h')} h. Packed maximum "
                         f"{len(primary.get('selected') or [])} places; other max-pack "
                         f"combinations ranked as alternatives."
                ),
                "options": [{
                    "label": o.get("label"),
                    "selected": o.get("selected") or [],
                    "dropped": o.get("dropped") or [],
                    "estimated_time_h": o.get("estimated_time_h"),
                    "total_satisfaction": o.get("total_satisfaction"),
                } for o in subset_options],
            }
            accepted = list(dict.fromkeys([start] + must_visit))
        else:
            accepted = list(dict.fromkeys([start] + [p for p in candidate_pool if p != start]))
            must_visit = []

        must_count = len(list(dict.fromkeys([start] + must_visit)))
        time_based_limit = max(must_count, int(max_total_time_min // 90))
        route_limit = max(must_count, min(time_based_limit, len(accepted)))

        # --- ACO ----------------------------------------------------------
        all_routes = []

        def _run_for_subset(subset_places, seed_base=42):
            local_must = list(subset_places)
            local_accepted = list(dict.fromkeys([start] + local_must))
            local_limit = max(len(local_accepted), min(len(local_accepted), int(max_total_time_min // 90)))
            collected = []
            main, _ = run_aco(
                G, start, local_accepted, preferences,
                max(15, num_ants // 2), max(8, iterations // 2),
                max_total_time_min, local_limit, local_must or None,
                seed=seed_base, copy_pheromone=True,
            )
            collected.extend(main)
            if local_must:
                collected.extend(generate_diverse_required_routes(
                    G, start, local_must, max_total_time_min, max_variants=6
                ))
                exact = optimize_required_route(G, start, local_must, max_total_time_min)
                if exact:
                    collected.append(exact)
            for r in collected:
                r["subset_places"] = local_must
            return collected

        if must_visit:
            all_routes.extend(_run_for_subset(must_visit, 42))
            for i, opt in enumerate(subset_options[1:], start=1):
                all_routes.extend(_run_for_subset(opt["selected"], 42 + i * 11))
        else:
            main, _ = run_aco(
                G, start, accepted, preferences,
                num_ants, iterations,
                max_total_time_min, route_limit, None,
                seed=42, copy_pheromone=True,
            )
            all_routes.extend(main)

        specialised = [
            {"attraction": 10, "budget": 2, "time": 3, "popular": 3},
            {"attraction": 3, "budget": 2, "time": 10, "popular": 3},
            {"attraction": 3, "budget": 10, "time": 3, "popular": 3},
            {"attraction": 3, "budget": 2, "time": 3, "popular": 10},
        ]
        for i, pref in enumerate(specialised):
            extra, _ = run_aco(
                G, start, accepted, pref,
                max(12, num_ants // 3), max(6, iterations // 3),
                max_total_time_min, route_limit, must_visit or None,
                seed=100 + i * 17, copy_pheromone=True,
            )
            all_routes.extend(extra)

        unique_so_far, seen_keys = [], set()
        for r in all_routes:
            key = tuple(r["route"])
            if key not in seen_keys and len(r["route"]) > 1:
                seen_keys.add(key)
                unique_so_far.append(r)
            if len(unique_so_far) >= 6:
                break
        for r in unique_so_far:
            for v in generate_route_variants(G, r["route"], max_total_time_min, max_variants=3):
                all_routes.append(v)

        if not all_routes:
            raise HTTPException(status_code=400, detail="No feasible route found for the given constraints.")

        ranked = evaluate_and_rank(all_routes, weights, PAST_TRIPS_FILE, use_pareto=True, min_results=5)
        ranked = [x for x in (refresh_route_metrics(G, r) for r in ranked) if x is not None]
        if not ranked:
            raise HTTPException(status_code=400, detail="No feasible route found. Increase days/hours or reduce places.")

        for r in ranked:
            r["detailed"] = compute_detailed_scores(G, r, weights, {})
            r["explanation"] = explain_route(G, r, r["detailed"], weights, {})

        best = ranked[0]
        itinerary = split_multiday(G, best["route"], int(request.days), float(request.hours_per_day))

        # --- coordinates ---------------------------------------------------
        pois_df = pd.read_csv(os.path.join(MOIP_DATA_DIR, "pois.csv"))
        pois_df.set_index("name", inplace=True)

        best_days = serialise_itinerary(G, itinerary, pois_df)

        # Top 5 options: best + up to 4 alternatives
        options = [{
            "rank": 1,
            "route": best["route"],
            "route_label": format_full_route(G, best["route"]),
            "days": best_days,
            "scores": score_row(best, rank=1),
            "explanation": best.get("explanation"),
        }]
        for i, route in enumerate(ranked[1:5], start=2):
            itin = split_multiday(G, route["route"], int(request.days), float(request.hours_per_day))
            options.append({
                "rank": i,
                "route": route["route"],
                "route_label": format_full_route(G, route["route"]),
                "days": serialise_itinerary(G, itin, pois_df),
                "scores": score_row(route, rank=i),
                "explanation": route.get("explanation"),
            })

        # --- evaluation & accuracy ----------------------------------------
        accuracy = None
        if must_visit:
            try:
                acc = compute_accuracy_for_input(
                    G, start, must_visit, weights, ranked, PAST_TRIPS_FILE,
                    max_total_time_min=max_total_time_min,
                )
                if acc:
                    table = acc.get("comparison_table")
                    if table is not None and hasattr(table, "to_dict"):
                        acc["comparison_table"] = table.to_dict("records")
                    accuracy = acc
            except Exception as acc_err:      # accuracy is optional, never fatal
                logger.warning(f"Accuracy evaluation skipped: {acc_err}")

        # Backwards-compatible payload for the existing renderer
        moip_result = json_safe({"best": best, "itinerary": best_days})

        # --- LLM summary (RAG graph unchanged) -----------------------------
        inputs = {
            "question": request.question,
            "query_intent": {"intent_type": "itinerary_planning"},
            "moip_result": {"best": best, "itinerary": itinerary},
        }
        try:
            result = rag_graph.invoke(inputs)
            response_text = result.get("evidenced_response") or "Successfully generated itinerary."
        except Exception as rag_err:
            # The optimiser result is already complete; a failed summary must not
            # discard it (this is error handling only - the graph call is unchanged).
            logger.error(f"LLM summary failed, returning itinerary without it: {rag_err}")
            response_text = ""

        return {
            "response": response_text,
            "evidence": [],
            "is_itinerary_request": False,
            "moip_result": moip_result,
            "planner": json_safe({
                "start": start,
                "start_label": place_label(G, start),
                "loaded_pois": len(places),
                "trip_days": int(request.days),
                "hours_per_day": float(request.hours_per_day),
                "trip_start_date": request.trip_start_date,
                "num_ants": num_ants,
                "iterations": iterations,
                "weights": weights,
                "must_visit_used": must_visit,
                "original_must_visit": original_must_visit,
                "removed_weather": removed_weather,
                "weather_enabled": bool(OPENWEATHERMAP_API_KEY.strip()),
                "subset_info": subset_info,
                "best_route_label": format_full_route(G, best["route"]),
                "options": options,
                "score_rows": [o["scores"] for o in options],
                "accuracy": accuracy,
            }),
        }

    except HTTPException:
        raise
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
