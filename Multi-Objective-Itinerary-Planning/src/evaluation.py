import pandas as pd
from utils import min_max


def deduplicate(routes):
    seen, out = set(), []
    for route in routes:
        key = tuple(route["route"])
        if key not in seen:
            seen.add(key)
            out.append(route)
    return out


def load_historical_routes(past_trips_file):
    try:
        past = pd.read_csv(past_trips_file)
    except Exception:
        return []
    trip_col = "trip_id" if "trip_id" in past.columns else past.columns[0]
    order_col = "order" if "order" in past.columns else past.columns[1]
    poi_col = "poi" if "poi" in past.columns else past.columns[2]
    routes = []
    for _, group in past.groupby(trip_col):
        group = group.sort_values(order_col)
        routes.append([str(x).strip() for x in group[poi_col].tolist()])
    return routes


def jaccard(a, b):
    a_set, b_set = set(a), set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def max_historical_similarity(route, historical_routes):
    if not historical_routes:
        return 0.0
    return max(jaccard(route, historical) for historical in historical_routes)


def dominates(a, b):
    return (
        a["total_satisfaction"] >= b["total_satisfaction"]
        and a["total_cost"] <= b["total_cost"]
        and a["total_time_min"] <= b["total_time_min"]
        and a["historical_similarity"] >= b["historical_similarity"]
        and (
            a["total_satisfaction"] > b["total_satisfaction"]
            or a["total_cost"] < b["total_cost"]
            or a["total_time_min"] < b["total_time_min"]
            or a["historical_similarity"] > b["historical_similarity"]
        )
    )


def pareto_front(routes):
    return [route for route in routes if not any(dominates(other, route) for other in routes if other is not route)]


def _score_routes(candidates, weights):
    if not candidates:
        return
    mins = {
        "s": min(r["total_satisfaction"] for r in candidates),
        "c": min(r["total_cost"] for r in candidates),
        "t": min(r["total_time_min"] for r in candidates),
        "h": min(r["historical_similarity"] for r in candidates),
        "p": min(len(r["route"]) for r in candidates),
        "d": min(r.get("total_distance_km", 0) for r in candidates),
    }
    maxs = {
        "s": max(r["total_satisfaction"] for r in candidates),
        "c": max(r["total_cost"] for r in candidates),
        "t": max(r["total_time_min"] for r in candidates),
        "h": max(r["historical_similarity"] for r in candidates),
        "p": max(len(r["route"]) for r in candidates),
        "d": max(r.get("total_distance_km", 0) for r in candidates),
    }
    for route in candidates:
        satisfaction = min_max(route["total_satisfaction"], mins["s"], maxs["s"], True)
        cost = min_max(route["total_cost"], mins["c"], maxs["c"], False)
        time = min_max(route["total_time_min"], mins["t"], maxs["t"], False)
        distance = min_max(route.get("total_distance_km", 0), mins["d"], maxs["d"], False)
        historical_similarity = min_max(route["historical_similarity"], mins["h"], maxs["h"], True)
        places_covered = min_max(len(route["route"]), mins["p"], maxs["p"], True)

        user_preference_score = (
            weights.get("attraction", 0.25) * satisfaction
            + weights.get("budget", 0.15) * cost
            + weights.get("time", 0.25) * time
            + weights.get("popular", 0.15) * historical_similarity
            + 0.10 * distance          # explicit distance preference
        )
        route["preference_score"] = 0.75 * user_preference_score + 0.25 * places_covered


def evaluate_and_rank(routes, weights, past_trips_file, use_pareto=True, min_results=4):
    """
    Rank routes from best to worst.
    Always returns at least min_results distinct routes when possible
    so the UI can show alternative itineraries.
    """
    routes = deduplicate(routes)
    if not routes:
        return []

    historical = load_historical_routes(past_trips_file)
    for route in routes:
        route["historical_similarity"] = max_historical_similarity(route["route"], historical)

    _score_routes(routes, weights)
    all_sorted = sorted(routes, key=lambda r: r["preference_score"], reverse=True)

    if not use_pareto:
        return all_sorted

    front = pareto_front(routes)
    _score_routes(front, weights)
    front_sorted = sorted(front, key=lambda r: r["preference_score"], reverse=True)

    seen = {tuple(r["route"]) for r in front_sorted}
    result = list(front_sorted)
    for r in all_sorted:
        key = tuple(r["route"])
        if key not in seen:
            result.append(r)
            seen.add(key)
        if len(result) >= min_results:
            break
    return result


def _travel_time(G, source, destination):
    if G.has_edge(source, destination):
        return float(G[source][destination].get("travel_time_min", 0))
    if G.has_edge(destination, source):
        return float(G[destination][source].get("travel_time_min", 0))
    return 0.0


def _travel_distance(G, source, destination):
    if G.has_edge(source, destination):
        return float(G[source][destination].get("distance_km", 0))
    if G.has_edge(destination, source):
        return float(G[destination][source].get("distance_km", 0))
    return 0.0


def split_multiday(G, route, days, hours_per_day):
    """
    Greedy day-wise split that keeps REAL inter-day travel times.
    """
    max_day_min = hours_per_day * 60
    itinerary = []
    current_places, current_legs = [], []
    current_time, current_distance = 0.0, 0.0
    day = 1

    for index, place in enumerate(route):
        stay = float(G.nodes[place].get("duration_time_min", 0))
        previous = route[index - 1] if index > 0 else None
        travel = _travel_time(G, previous, place) if previous else 0.0
        distance = _travel_distance(G, previous, place) if previous else 0.0
        add = stay + travel

        if current_places and (current_time + add > max_day_min) and day < days:
            itinerary.append({
                "day": day,
                "places": current_places,
                "legs": current_legs,
                "time_min": current_time,
                "distance_km": current_distance,
                "within_limit": current_time <= max_day_min,
            })
            day += 1
            # keep real travel to the first place of the new day
            current_places = [place]
            current_legs = [{
                "Place": place,
                "Travel From Previous (min)": round(travel, 2),
                "Visit Time (min)": round(stay, 2),
                "Distance From Previous (km)": round(distance, 2),
            }]
            current_time = add
            current_distance = distance
        else:
            current_places.append(place)
            current_legs.append({
                "Place": place,
                "Travel From Previous (min)": round(travel, 2),
                "Visit Time (min)": round(stay, 2),
                "Distance From Previous (km)": round(distance, 2),
            })
            current_time += add
            current_distance += distance

    if current_places:
        itinerary.append({
            "day": day,
            "places": current_places,
            "legs": current_legs,
            "time_min": current_time,
            "distance_km": current_distance,
            "within_limit": current_time <= max_day_min,
        })
    return itinerary
