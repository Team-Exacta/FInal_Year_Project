import os, sys, traceback, copy
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.append(os.path.dirname(__file__))

from build_graph import build_graph
from trips import initialize_pheromones_from_past_trips
from utils import split_text_places, fuzzy_match_place, normalize_weights
from weather import filter_places_by_weather
from aco import (
    run_aco,
    route_metrics,
    optimize_required_route,
    generate_diverse_required_routes,
    generate_route_variants,
)
from evaluation import evaluate_and_rank, split_multiday
from map_utils import create_route_map
from scoring import compute_detailed_scores, explain_route

POI_FILE = "data/pois.csv"
DISTANCE_FILE = "data/route_distance_matrix.csv"
DURATION_FILE = "data/route_duration_matrix.csv"
PAST_TRIPS_FILE = "data/past_trips_dataset.csv"
OPENWEATHERMAP_API_KEY = ""
ROUTE_SEPARATOR = " → "


def refresh_route_metrics(G, route_result):
    refreshed = route_metrics(G, route_result["route"])
    return {**route_result, **refreshed}


def place_with_category(G, place):
    category = str(G.nodes[place].get("category", "")).strip()
    return f"{place} ({category})" if category else place


def format_day_route(G, places):
    return ROUTE_SEPARATOR.join(place_with_category(G, place) for place in places)


def option_summary(route_result, scores=None):
    base = {
        "Overall Score": round(route_result.get("preference_score", 0), 4),
        "Satisfaction": round(route_result["total_satisfaction"], 2),
        "Cost (LKR)": round(route_result["total_cost"], 0),
        "Time (h)": round(route_result["total_time_min"] / 60, 2),
        "Distance (km)": round(route_result["total_distance_km"], 1),
        "Historical Similarity": round(route_result.get("historical_similarity", 0), 3),
        "Places": len(route_result["route"]),
    }
    if scores:
        base["Time Efficiency"] = scores.get("time_efficiency")
        base["Distance Efficiency"] = scores.get("distance_efficiency")
        base["Weather Safety"] = scores.get("weather_safety")
        base["Stay Balance"] = scores.get("stay_balance")
    return base


@st.cache_resource(show_spinner=False)
def load_graph():
    """Load graph once and cache — prevents UI blink on every interaction."""
    G = build_graph(POI_FILE, DISTANCE_FILE, DURATION_FILE)
    G, matched, missing = initialize_pheromones_from_past_trips(G, PAST_TRIPS_FILE, boost=5.0)
    return G, matched, missing


st.set_page_config(page_title="Sri Lanka ACO Itinerary Optimizer", layout="wide")
st.title("Sri Lanka Multi-Objective Itinerary Optimization")
st.caption("ACO + past-trip pheromones · preference-aware ranking · weather-aware filtering · multi-day split")

with st.sidebar:
    st.header("Weather")
    avoid_bad_weather = st.checkbox("Avoid rainy / storm / flood-risk places", value=True)
    st.header("ACO Settings")
    num_ants = st.number_input("Number of ants", 10, 1000, 80, 10)
    iterations = st.number_input("Iterations", 5, 300, 30, 5)

try:
    G, matched, missing = load_graph()
except Exception as e:
    st.error(f"Failed to load graph: {e}")
    st.code(traceback.format_exc())
    st.stop()

places = sorted(list(G.nodes()))
st.success(f"Loaded {len(places)} POIs and road network successfully.")

with st.form("plan_form"):
    col1, col2 = st.columns(2)
    with col1:
        start = st.selectbox(
            "Start place",
            places,
            index=places.index("Colombo Lotus Tower") if "Colombo Lotus Tower" in places else 0,
        )
        trip_days = st.number_input("Number of days", 1, 30, 4)
        hours_per_day = st.number_input("Hours per day", 4.0, 16.0, 8.0, 0.5)
        max_places_to_visit = st.number_input("Max places to visit", 2, 30, 12)
    with col2:
        use_all_pois = st.checkbox("Use all POIs as candidates (ignore must-visit)", value=False)
        must_visit_raw = st.text_area(
            "Must-visit places (comma or newline separated)",
            value="",
            height=120,
            help="Example: Sigiriya Lion Rock, Temple of the Sacred Tooth Relic, Ella Rock, Mirissa Beach",
        )
        st.markdown("**User Preference Priorities** (0–10)")
        attraction_priority = st.slider("Attraction / Satisfaction priority", 0, 10, 8)
        budget_priority = st.slider("Budget priority", 0, 10, 4)
        travel_time_priority = st.slider("Travel time priority", 0, 10, 7)
        popular_route_priority = st.slider("Follow popular routes priority", 0, 10, 5)
    submitted = st.form_submit_button("Generate Optimized Routes")

if submitted:
    try:
        typed_places = [fuzzy_match_place(p, places) for p in split_text_places(must_visit_raw)]
        if typed_places:
            must_visit = list(dict.fromkeys(typed_places))
            selected = list(dict.fromkeys([start] + must_visit))
        elif use_all_pois:
            must_visit = []
            selected = list(places)
        else:
            # Default: start + a sensible pool of popular places so routes are non-trivial
            must_visit = []
            popular_defaults = [
                "Sigiriya Lion Rock",
                "Temple of the Sacred Tooth Relic",
                "Dambulla Cave Temple",
                "Gregory Lake",
                "Ella Rock",
                "Nine Arches Bridge",
                "Yala National Park",
                "Mirissa Beach",
                "Galle Dutch Fort",
                "Hikkaduwa Beach",
                "Negombo Beach",
                "Pinnawala Elephant Orphanage",
            ]
            selected = list(dict.fromkeys([start] + [p for p in popular_defaults if p in G]))

        if start not in selected:
            selected.insert(0, start)

        accepted, removed, weather_report = filter_places_by_weather(
            G, selected, OPENWEATHERMAP_API_KEY, avoid_bad_weather
        )
        if start not in accepted:
            accepted.insert(0, start)
        if len(accepted) < 2:
            st.error("Not enough places after weather filtering.")
            st.stop()

        preferences = {
            "attraction": attraction_priority,
            "budget": budget_priority,
            "time": travel_time_priority,
            "popular": popular_route_priority,
        }
        weights = normalize_weights(preferences)
        max_total_time_min = int(trip_days) * float(hours_per_day) * 60
        route_limit = max(int(max_places_to_visit), len(list(dict.fromkeys([start] + must_visit))))

        with st.spinner("Running multi-preference ACO, ranking routes, building day plans & map..."):
            all_routes = []

            # 1) Main ACO with user weights
            main_routes, _ = run_aco(
                G,
                start,
                accepted,
                preferences,
                int(num_ants),
                int(iterations),
                max_total_time_min,
                route_limit,
                must_visit,
                seed=42,
                copy_pheromone=True,
            )
            all_routes.extend(main_routes)

            # 2) Diverse must-visit orderings (guarantees alternatives when must-visits exist)
            if must_visit:
                diverse = generate_diverse_required_routes(
                    G, start, must_visit, max_total_time_min, max_variants=8
                )
                all_routes.extend(diverse)
                exact = optimize_required_route(G, start, must_visit, max_total_time_min)
                if exact:
                    all_routes.append(exact)

            # 3) Specialised preference runs for different emphases
            specialised = [
                {"attraction": 10, "budget": 2, "time": 3, "popular": 3},  # max satisfaction
                {"attraction": 3, "budget": 2, "time": 10, "popular": 3},  # min time
                {"attraction": 3, "budget": 10, "time": 3, "popular": 3},  # min cost
                {"attraction": 3, "budget": 2, "time": 3, "popular": 10},  # max popular
                {"attraction": 5, "budget": 5, "time": 8, "popular": 5},  # balanced
            ]
            for i, pref in enumerate(specialised):
                extra, _ = run_aco(
                    G,
                    start,
                    accepted,
                    pref,
                    max(15, int(num_ants) // 3),
                    max(8, int(iterations) // 3),
                    max_total_time_min,
                    route_limit,
                    must_visit,
                    seed=100 + i * 17,
                    copy_pheromone=True,
                )
                all_routes.extend(extra)

            # 4) Neighbourhood variants of the current best candidates (extra diversity)
            unique_so_far = []
            seen_keys = set()
            for r in all_routes:
                key = tuple(r["route"])
                if key not in seen_keys and len(r["route"]) > 2:
                    seen_keys.add(key)
                    unique_so_far.append(r)
                if len(unique_so_far) >= 5:
                    break
            for r in unique_so_far:
                variants = generate_route_variants(
                    G, r["route"], max_total_time_min, max_variants=4
                )
                for v in variants:
                    if all(p in v["route"] for p in must_visit if p != start):
                        all_routes.append(v)

            # Rank — always try for ≥ 5 distinct routes
            ranked = evaluate_and_rank(
                all_routes, weights, PAST_TRIPS_FILE, use_pareto=True, min_results=5
            )
            ranked = [refresh_route_metrics(G, r) for r in ranked]

        if not ranked:
            requested = (
                route_metrics(G, list(dict.fromkeys([start] + must_visit))) if must_visit else None
            )
            if requested:
                st.error(
                    f"No feasible route found. Requested places need ~{requested['total_time_min']/60:.1f} h "
                    f"but only {max_total_time_min/60:.1f} h available. Increase days/hours or reduce places."
                )
            else:
                st.error("No feasible route found.")
            st.stop()

        # Attach detailed scores + explanations
        for r in ranked:
            r["detailed"] = compute_detailed_scores(G, r, weights, weather_report)
            r["explanation"] = explain_route(G, r, r["detailed"], weights, weather_report)

        best = ranked[0]
        itinerary = split_multiday(G, best["route"], int(trip_days), float(hours_per_day))

        # Build alternatives from ranks 2..5
        alternative_itineraries = []
        for route in ranked[1:5]:
            alternative_itineraries.append(
                {
                    "route": route,
                    "itinerary": split_multiday(
                        G, route["route"], int(trip_days), float(hours_per_day)
                    ),
                }
            )

        route_map = create_route_map(G, best["route"], "outputs/best_route_map.html")

        st.session_state.update(
            {
                "ranked": ranked,
                "best": best,
                "weights": weights,
                "itinerary": itinerary,
                "alternative_itineraries": alternative_itineraries,
                "route_map": route_map,
                "removed": removed,
                "weather_report": weather_report,
                "preferences": preferences,
            }
        )
    except Exception as e:
        st.error(str(e))
        st.code(traceback.format_exc())

# -------------------- DISPLAY --------------------
if "ranked" in st.session_state:
    best = st.session_state["best"]
    weights = st.session_state["weights"]
    itinerary = st.session_state["itinerary"]
    alternative_itineraries = st.session_state.get("alternative_itineraries", [])
    route_map = st.session_state["route_map"]
    removed = st.session_state["removed"]
    weather_report = st.session_state["weather_report"]
    ranked = st.session_state["ranked"]

    st.divider()
    st.subheader("1. Weather Filtering Result")
    if removed:
        st.warning(
            "Removed due to unfavourable weather: "
            + ", ".join(removed[:30])
            + (" …" if len(removed) > 30 else "")
        )
    else:
        st.info("No places removed by weather filtering.")

    st.subheader("2. Normalised User Preference Weights")
    st.write(weights)

    # ---- BEST ROUTE ----
    st.subheader("3. Best Optimised Route (Rank #1)")
    st.success(format_day_route(G, best["route"]))

    scores = best.get("detailed", {})
    st.markdown("**Route Score Summary**")
    st.dataframe(pd.DataFrame([option_summary(best, scores)]), use_container_width=True)

    st.markdown("**Why this route is suitable**")
    st.write(best.get("explanation", ""))

    if best.get("legs"):
        st.markdown("**Detailed Leg Breakdown**")
        st.dataframe(pd.DataFrame(best["legs"]), use_container_width=True)

    st.subheader("4. Best Route – Day-by-Day Plan")
    for day in itinerary:
        day_hours = round(day["time_min"] / 60, 2)
        day_distance = round(day.get("distance_km", 0), 2)
        status = "within limit" if day.get("within_limit", True) else "exceeds daily limit"
        st.markdown(
            f"**Day {day['day']}** – Estimated time: **{day_hours} h**, distance: **{day_distance} km** ({status})"
        )
        st.write(format_day_route(G, day["places"]))
        if day.get("legs"):
            st.dataframe(pd.DataFrame(day["legs"]), use_container_width=True)

    st.subheader("5. Best Route on Map")
    # returned_objects=[] + stable key prevents the common streamlit-folium blink / full-page reload
    st_folium(
        route_map,
        width=1200,
        height=650,
        returned_objects=[],
        key="best_route_map_display",
    )

    # ---- RANKED LIST + ALTERNATIVES ----
    st.subheader("6. Ranked List of Candidate Routes (Best → Worst)")
    rank_rows = []
    for i, r in enumerate(ranked[:8], 1):
        d = r.get("detailed", {})
        rank_rows.append(
            {
                "Rank": i,
                "Route (ordered POIs)": " → ".join(r["route"]),
                "Overall Score": round(r.get("preference_score", 0), 4),
                "Time (h)": round(r["total_time_min"] / 60, 2),
                "Distance (km)": round(r["total_distance_km"], 1),
                "Satisfaction": round(r["total_satisfaction"], 2),
                "Cost": round(r["total_cost"], 0),
                "Hist. Similarity": round(r.get("historical_similarity", 0), 3),
                "Time Eff.": d.get("time_efficiency"),
                "Dist. Eff.": d.get("distance_efficiency"),
                "Weather": d.get("weather_safety"),
            }
        )
    st.dataframe(pd.DataFrame(rank_rows), use_container_width=True)

    st.subheader("7. Alternative Itinerary Options (for flexibility)")
    if alternative_itineraries:
        for option_number, option in enumerate(alternative_itineraries, 1):
            route = option["route"]
            option_days = option.get("itinerary") or []
            st.markdown(f"### Alternative Option {option_number}")
            st.write(format_day_route(G, route["route"]))
            st.dataframe(
                pd.DataFrame([option_summary(route, route.get("detailed"))]),
                use_container_width=True,
            )
            st.caption(route.get("explanation", ""))
            for day in option_days:
                day_hours = round(day["time_min"] / 60, 2)
                day_distance = round(day.get("distance_km", 0), 2)
                status = "within limit" if day.get("within_limit", True) else "exceeds daily limit"
                st.markdown(
                    f"**Day {day['day']}** – {day_hours} h, {day_distance} km ({status})"
                )
                st.write(format_day_route(G, day["places"]))
    else:
        st.info(
            "Only one distinct feasible route was found for these inputs. "
            "Try more days/hours, fewer must-visit places, or enable “Use all POIs” to see alternatives."
        )

    st.success(
        "Optimisation complete. Use the ranked list and map above to choose the itinerary that best matches your priorities."
    )
