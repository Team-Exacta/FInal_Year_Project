import os, sys, traceback, copy
from datetime import date, datetime, timedelta
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.append(os.path.dirname(__file__))

from build_graph import build_graph
from trips import initialize_pheromones_from_past_trips
from utils import split_text_places, fuzzy_match_place, normalize_weights
from weather import (
    filter_places_by_weather,
    build_weather_table,
    places_bad_on_any_trip_day,
)
from aco import (
    run_aco,
    route_metrics,
    optimize_required_route,
    generate_diverse_required_routes,
    generate_route_variants,
    select_places_for_budget,
    generate_multiple_subset_options,
)
from evaluation import evaluate_and_rank, split_multiday
from map_utils import create_multiday_map, create_route_map
from scoring import compute_detailed_scores, explain_route
from accuracy_ui import can_compute_accuracy, compute_accuracy_for_input, permutation_count

POI_FILE = "data/pois.csv"
DISTANCE_FILE = "data/route_distance_matrix.csv"
DURATION_FILE = "data/route_duration_matrix.csv"
PAST_TRIPS_FILE = "data/past_trips_dataset.csv"

# ============================================================
# OPENWEATHERMAP API KEY — replace with your key
# Sign up free: https://home.openweathermap.org/users/sign_up
# Create key:   https://home.openweathermap.org/api_keys
# ============================================================
# Leave empty "" to skip weather completely
OPENWEATHERMAP_API_KEY = ""

ROUTE_SEPARATOR = " → "


def refresh_route_metrics(G, route_result):
    refreshed = route_metrics(G, route_result["route"])
    if refreshed is None:
        return None
    return {**route_result, **refreshed}


def place_label(G, place):
    """Place with category, e.g. Colombo Lotus Tower (Landmark / Tower)"""
    if place not in G.nodes:
        return place
    cat = str(G.nodes[place].get("category", "")).strip()
    return f"{place} ({cat})" if cat else place


def format_full_route(G, route):
    """Colombo Lotus Tower (Landmark / Tower) → Museum (...) → ..."""
    return " → ".join(place_label(G, p) for p in route)


def format_day_places(G, places):
    """
    1 place:  Place (cat)
    2+ places: start-First (cat) → ... → end-Last (cat)
    """
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
    """Full day route line: A (cat) → B (cat) → ..."""
    if not places:
        return ""
    return " → ".join(place_label(G, p) for p in places)


def render_day_by_day(G, itinerary, title="Best Route – Day-by-Day Plan"):
    """Streamlit day-by-day block; shows connection from previous day."""
    st.subheader(title)
    prev_end = None
    for day in itinerary:
        d = day.get("day", 0)
        h = round(day.get("time_min", 0) / 60, 2)
        km = round(day.get("distance_km", 0), 2)
        status = "within limit" if day.get("within_limit", True) else "exceeds daily limit"
        places = day.get("places") or []
        st.markdown(f"**Day {d} – Estimated time: {h} h, distance: {km} km ({status})**")
        if prev_end and places and prev_end != places[0]:
            st.caption(
                f"connected from previous day: {place_label(G, prev_end)} → {place_label(G, places[0])}"
            )
        st.write(format_day_places(G, places))
        if len(places) >= 1:
            st.caption("route - " + format_day_route_line(G, places))
        if places:
            prev_end = places[-1]


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


@st.cache_resource(show_spinner=False)
def load_graph():
    G = build_graph(POI_FILE, DISTANCE_FILE, DURATION_FILE)
    G, matched, missing = initialize_pheromones_from_past_trips(G, PAST_TRIPS_FILE, boost=5.0)
    return G, matched, missing


st.set_page_config(page_title="Sri Lanka ACO Itinerary Optimizer", layout="wide")
st.title("Sri Lanka Multi-Objective Itinerary Optimization")
st.caption("ACO · real road matrix · day-wise weather · multi-day coloured map · evaluation & accuracy")

with st.sidebar:
    st.header("Weather")
    avoid_bad_weather = st.checkbox("Avoid places with bad weather", value=True)
    st.markdown(
        """
**OpenWeatherMap API**
1. Sign up: [openweathermap.org/users/sign_up](https://home.openweathermap.org/users/sign_up)
2. Keys: [openweathermap.org/api_keys](https://home.openweathermap.org/api_keys)
3. Paste key in `src/app.py` → `OPENWEATHERMAP_API_KEY`
"""
    )
    st.caption("Key set: Yes" if OPENWEATHERMAP_API_KEY.strip() else "Key set: No (weather ignored)")
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
st.success(f"Loaded {len(places)} POIs and road network.")

with st.form("plan_form"):
    col1, col2 = st.columns(2)
    with col1:
        start = st.selectbox(
            "Start place",
            places,
            index=places.index("Colombo Lotus Tower") if "Colombo Lotus Tower" in places else 0,
        )
        trip_start_date = st.date_input("Trip start date", value=date.today())
        trip_days = st.number_input("Number of days", 1, 30, 4)
        hours_per_day = st.number_input("Hours per day", 4.0, 16.0, 8.0, 0.5)
    with col2:
        use_all_pois = st.checkbox("Use all POIs as candidates (ignore must-visit)", value=False)
        must_visit_raw = st.text_area(
            "Candidate / must-visit places (comma or newline)",
            value="",
            height=120,
            help="Example: Aberdeen Falls, Arugam Bay, Hikkaduwa Beach, Coconut Tree Hill",
        )
        st.markdown("**Preference priorities (0–10)**")
        attraction_priority = st.slider("Attraction / Satisfaction", 0, 10, 8)
        budget_priority = st.slider("Budget", 0, 10, 4)
        travel_time_priority = st.slider("Travel time", 0, 10, 7)
        popular_route_priority = st.slider("Popular routes", 0, 10, 5)

    submitted = st.form_submit_button("Generate Optimized Routes", type="primary")

if submitted:
    try:
        typed_places = [fuzzy_match_place(p, places) for p in split_text_places(must_visit_raw)]
        typed_places = [p for p in typed_places if p]

        if use_all_pois or not typed_places:
            must_visit = []
            candidate_pool = list(places)
            original_must_visit = []
        else:
            must_visit = list(dict.fromkeys([p for p in typed_places if p != start]))
            original_must_visit = list(must_visit)
            candidate_pool = list(dict.fromkeys([start] + must_visit))

        preferences = {
            "attraction": attraction_priority,
            "budget": budget_priority,
            "time": travel_time_priority,
            "popular": popular_route_priority,
        }
        weights = normalize_weights(preferences)
        max_total_time_min = int(trip_days) * float(hours_per_day) * 60

        # ---------- Weather table (per place × day) ----------
        weather_table_rows = []
        bad_on_day = {d: set() for d in range(1, int(trip_days) + 1)}
        weather_report = {}
        removed_weather = []

        check_places = list(dict.fromkeys([start] + (original_must_visit or candidate_pool[:25])))
        if OPENWEATHERMAP_API_KEY.strip() and avoid_bad_weather:
            with st.spinner("Fetching OpenWeatherMap forecast for trip days..."):
                weather_table_rows, bad_on_day, weather_report = build_weather_table(
                    G, check_places, OPENWEATHERMAP_API_KEY, trip_start_date, int(trip_days)
                )
            # Places bad on ALL days → drop from candidates entirely
            always_bad = set(check_places)
            for d in range(1, int(trip_days) + 1):
                always_bad &= bad_on_day.get(d, set())
            # Places bad on majority of days → also drop
            bad_counts = {}
            for d, s in bad_on_day.items():
                for p in s:
                    bad_counts[p] = bad_counts.get(p, 0) + 1
            majority_bad = {p for p, c in bad_counts.items() if c >= max(1, int(trip_days) // 2 + 1)}
            removed_weather = sorted(always_bad | majority_bad)
            if start in removed_weather:
                removed_weather = [p for p in removed_weather if p != start]
            if must_visit:
                must_visit = [p for p in must_visit if p not in removed_weather]
            candidate_pool = [p for p in candidate_pool if p not in removed_weather]
            if start not in candidate_pool:
                candidate_pool.insert(0, start)
        # No API key → do not consider weather at all (no message spam)

        # ---------- Subset selection if not all fit ----------
        subset_options = []
        subset_info = None
        if must_visit:
            subset_options = generate_multiple_subset_options(
                G, start, must_visit, max_total_time_min, weights, max_options=5
            )
            if not subset_options:
                st.error("No places fit in the available time. Increase days/hours or choose closer places.")
                st.stop()
            primary = subset_options[0]
            must_visit = primary["selected"]
            subset_info = {
                "fitted_all": primary.get("fitted_all", False),
                "selected": primary["selected"],
                "dropped": primary["dropped"],
                "estimated_time_h": primary.get("estimated_time_h"),
                "budget_h": primary.get("budget_h"),
                "full_needed_h": primary.get("full_needed_h"),
                "options": subset_options,
                "reason": (
                    "All places fit the day budget."
                    if primary.get("fitted_all")
                    else f"Not all places fit {primary.get('budget_h')} h. Packed maximum {len(primary.get('selected') or [])} places; other max-pack combinations ranked as alternatives."
                ),
            }
            accepted = list(dict.fromkeys([start] + must_visit))
        else:
            accepted = list(dict.fromkeys([start] + [p for p in candidate_pool if p != start]))
            # Cap by time budget roughly
            must_visit = []

        must_count = len(list(dict.fromkeys([start] + must_visit)))
        time_based_limit = max(must_count, int(max_total_time_min // 90))
        route_limit = max(must_count, min(time_based_limit, len(accepted)))

        with st.spinner("Running ACO, ranking top routes, building day plans & map..."):
            all_routes = []

            def _run_for_subset(subset_places, seed_base=42):
                local_must = list(subset_places)
                local_accepted = list(dict.fromkeys([start] + local_must))
                local_limit = max(len(local_accepted), min(len(local_accepted), int(max_total_time_min // 90)))
                collected = []
                main, _ = run_aco(
                    G, start, local_accepted, preferences,
                    max(15, int(num_ants) // 2), max(8, int(iterations) // 2),
                    max_total_time_min, local_limit, local_must or None,
                    seed=seed_base, copy_pheromone=True,
                )
                collected.extend(main)
                if local_must:
                    diverse = generate_diverse_required_routes(
                        G, start, local_must, max_total_time_min, max_variants=6
                    )
                    collected.extend(diverse)
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
                    int(num_ants), int(iterations),
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
                    max(12, int(num_ants) // 3), max(6, int(iterations) // 3),
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

            ranked = evaluate_and_rank(
                all_routes, weights, PAST_TRIPS_FILE, use_pareto=True, min_results=5
            )
            ranked = [x for x in (refresh_route_metrics(G, r) for r in ranked) if x is not None]

            # Prefer routes that avoid placing a POI on a day when its weather is bad
            if OPENWEATHERMAP_API_KEY.strip() and avoid_bad_weather and bad_on_day:
                def weather_penalty(route_dict):
                    itin = split_multiday(G, route_dict["route"], int(trip_days), float(hours_per_day))
                    pen = 0
                    for day_plan in itin:
                        d = day_plan["day"]
                        bad_set = bad_on_day.get(d, set())
                        for p in day_plan["places"]:
                            if p in bad_set:
                                pen += 1
                    return pen
                ranked.sort(
                    key=lambda r: (weather_penalty(r), -r.get("preference_score", 0))
                )
                # re-attach ranks after weather-aware sort but keep preference for display
                # soft: only reorder, scores stay

        if not ranked:
            st.error("No feasible route found. Increase days/hours or reduce places.")
            st.stop()

        for r in ranked:
            r["detailed"] = compute_detailed_scores(G, r, weights, {})
            r["explanation"] = explain_route(G, r, r["detailed"], weights, {})

        best = ranked[0]
        itinerary = split_multiday(G, best["route"], int(trip_days), float(hours_per_day))
        # Day-specific weather: if a place is bad on its assigned day, note it
        day_weather_notes = []
        for day_plan in itinerary:
            d = day_plan["day"]
            bad_set = bad_on_day.get(d, set())
            bad_here = [p for p in day_plan["places"] if p in bad_set]
            if bad_here:
                day_weather_notes.append(f"Day {d}: bad weather risk at {', '.join(bad_here)}")

        alternative_itineraries = []
        for route in ranked[1:5]:  # up to 4 alternatives → 5 total
            alternative_itineraries.append({
                "route": route,
                "itinerary": split_multiday(G, route["route"], int(trip_days), float(hours_per_day)),
            })

        route_map = create_multiday_map(G, itinerary, "outputs/best_route_map.html")

        accuracy_result = None
        if must_visit and can_compute_accuracy(start, must_visit):
            with st.spinner("Evaluation & accuracy (exhaustive Top-1 / score gap)..."):
                accuracy_result = compute_accuracy_for_input(
                    G, start, must_visit, weights, ranked, PAST_TRIPS_FILE,
                    max_total_time_min=max_total_time_min,
                )
        elif must_visit:
            with st.spinner("Accuracy evaluation..."):
                accuracy_result = compute_accuracy_for_input(
                    G, start, must_visit, weights, ranked, PAST_TRIPS_FILE,
                    max_total_time_min=max_total_time_min,
                )

        st.session_state.update({
            "ranked": ranked,
            "best": best,
            "weights": weights,
            "itinerary": itinerary,
            "alternative_itineraries": alternative_itineraries,
            "route_map": route_map,
            "accuracy_result": accuracy_result,
            "subset_info": subset_info,
            "subset_options": subset_options,
            "weather_table_rows": weather_table_rows,
            "removed_weather": removed_weather,
            "day_weather_notes": day_weather_notes,
            "trip_start_date": str(trip_start_date),
            "trip_days": int(trip_days),
            "hours_per_day": float(hours_per_day),
            "start_used": start,
            "must_visit_used": must_visit,
            "original_must_visit": original_must_visit,
        })
    except Exception as e:
        st.error(str(e))
        st.code(traceback.format_exc())

# -------------------- DISPLAY --------------------
if "ranked" in st.session_state:
    best = st.session_state["best"]
    ranked = st.session_state["ranked"]
    itinerary = st.session_state["itinerary"]
    alternative_itineraries = st.session_state.get("alternative_itineraries", [])
    route_map = st.session_state["route_map"]
    weights = st.session_state["weights"]

    st.divider()

    # Weather table ONLY if API key was used
    weather_rows = st.session_state.get("weather_table_rows") or []
    if weather_rows:
        st.subheader("Weather forecast (places × trip days)")
        st.caption(
            f"Start date: {st.session_state.get('trip_start_date')} · "
            f"{st.session_state.get('trip_days')} days · OpenWeatherMap 5-day forecast"
        )
        st.dataframe(pd.DataFrame(weather_rows), use_container_width=True)
        removed_w = st.session_state.get("removed_weather") or []
        if removed_w:
            st.warning("Avoided (bad weather on most/all trip days): " + ", ".join(removed_w))
        notes = st.session_state.get("day_weather_notes") or []
        for n in notes:
            st.warning(n)

    subset_info = st.session_state.get("subset_info")
    if subset_info and not subset_info.get("fitted_all", True):
        st.warning(subset_info.get("reason", ""))
        opts = st.session_state.get("subset_options") or []
        if opts:
            st.caption("Alternative place combinations considered:")
            st.dataframe(
                pd.DataFrame([{
                    "Strategy": o.get("label"),
                    "Selected": ", ".join(o.get("selected") or []),
                    "Dropped": ", ".join(o.get("dropped") or []),
                    "Est. h": o.get("estimated_time_h"),
                    "Satisfaction": o.get("total_satisfaction"),
                } for o in opts]),
                use_container_width=True,
            )

    # ---- BEST ROUTE (required format) ----
    st.subheader("Best route")
    st.write(format_full_route(G, best["route"]))

    render_day_by_day(G, itinerary, title="Best Route – Day-by-Day Plan")

    st.markdown("**Best route scores**")
    st.dataframe(pd.DataFrame([score_row(best, rank=1)]), use_container_width=True)

    # Map — one colour per day
    st.subheader("Best route map (different colour per day)")
    st.caption("Day 1 = red · Day 2 = blue · Day 3 = green · Day 4 = purple · … · real roads via OSRM")
    st_folium(route_map, width=1200, height=560, returned_objects=[], key="best_map")

    # Exactly 5 route options when possible (best + 4 alternatives)
    st.subheader("Route options (1–5)")
    top5 = ranked[:5]
    # pad alternatives if session list short
    while len(alternative_itineraries) < max(0, len(top5) - 1):
        idx = len(alternative_itineraries) + 1
        if idx >= len(ranked):
            break
        alternative_itineraries.append({
            "route": ranked[idx],
            "itinerary": split_multiday(
                G, ranked[idx]["route"],
                int(st.session_state.get("trip_days", 4)),
                float(st.session_state.get("hours_per_day", 8.0)),
            ),
        })

    score_rows = [score_row(best, rank=1)]
    for i, option in enumerate(alternative_itineraries[:4], start=2):
        route = option["route"]
        itin = option["itinerary"]
        st.markdown(f"### Route {i}")
        st.write(format_full_route(G, route["route"]))
        render_day_by_day(G, itin, title=f"Route {i} – Day-by-Day Plan")
        score_rows.append(score_row(route, rank=i))

    st.markdown("**Scores for all 5 routes**")
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True)

    # Evaluation / accuracy (Top-1, Top-3, score gap, ACO rank only)
    st.subheader("Evaluation & accuracy")
    acc = st.session_state.get("accuracy_result")
    if acc and (acc.get("top1") is not None or acc.get("score_gap") is not None):
        st.caption(
            "Accuracy vs true best order (exhaustive check of all place sequences). "
            "Lower score gap is better; Top-1 = Yes means ACO found an optimal order."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top-1", "Yes" if acc.get("top1") else "No")
        c2.metric("Top-3 hit", "Yes" if acc.get("top3_hit") else "No")
        gap = acc.get("score_gap")
        c3.metric("Score gap", f"{gap*100:.2f}%" if gap is not None else "—")
        c4.metric("ACO rank", acc.get("aco_rank") or "—")
        if acc.get("true_best_route"):
            st.caption("True best route (exhaustive): " + " → ".join(acc["true_best_route"]))
        if acc.get("aco_route") or acc.get("true_best_score") is not None:
            if acc.get("true_best_score") is not None:
                st.caption(
                    f"True best score: {acc.get('true_best_score')} · "
                    f"ACO fair score: {acc.get('aco_fair_score', '—')}"
                )
    else:
        st.caption(
            "Accuracy needs a small fixed must-visit list (≤ 7 intermediate places) "
            "so every order can be checked."
        )

    st.caption(f"Weights used: {weights}")
