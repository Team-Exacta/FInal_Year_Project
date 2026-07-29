"""Rich multi-objective scoring + human-readable explanations."""
from utils import min_max

def compute_detailed_scores(G, route_result, weights, weather_report=None):
    """
    Returns a dict with component scores (0-1 higher better) and overall score.
    Components: preference match, time efficiency, distance efficiency,
                weather safety, stay quality, historical popularity.
    """
    route = route_result["route"]
    n = max(len(route), 1)
    total_time = max(route_result.get("total_time_min", 1), 1)
    total_dist = max(route_result.get("total_distance_km", 1), 1)
    total_sat = route_result.get("total_satisfaction", 0)
    total_cost = route_result.get("total_cost", 0)
    hist_sim = route_result.get("historical_similarity", 0)

    # Preference / satisfaction (higher better)
    sat_score = min(1.0, total_sat / (n * 3.5))          # rough normalisation

    # Time efficiency (lower time better → invert)
    # Assume ideal ~ 90 min per place + modest travel
    ideal_time = n * 90 + (n - 1) * 40
    time_score = max(0.0, 1.0 - abs(total_time - ideal_time) / max(total_time, ideal_time))

    # Distance efficiency
    ideal_dist = (n - 1) * 40
    dist_score = max(0.0, 1.0 - abs(total_dist - ideal_dist) / max(total_dist, ideal_dist, 1))

    # Cost / budget
    cost_score = 1.0 / (1.0 + total_cost / 5000)

    # Historical / popular
    popular_score = hist_sim

    # Weather safety (fraction of places that are not bad weather)
    weather_score = 1.0
    if weather_report:
        bad = 0
        checked = 0
        for p in route:
            info = weather_report.get(p, {})
            if info.get("available"):
                checked += 1
                if info.get("is_bad"):
                    bad += 1
        if checked:
            weather_score = 1.0 - (bad / checked)

    # Stay duration balance (prefer places with reasonable visit times)
    stays = [float(G.nodes[p].get("duration_time_min", 60)) for p in route]
    avg_stay = sum(stays) / n
    stay_score = 1.0 - min(1.0, abs(avg_stay - 120) / 180)

    # Weighted overall (user weights dominate)
    w = weights
    overall = (
        w.get("attraction", 0.25) * sat_score
        + w.get("budget", 0.15) * cost_score
        + w.get("time", 0.25) * time_score
        + w.get("popular", 0.15) * popular_score
        + 0.10 * dist_score
        + 0.05 * weather_score
        + 0.05 * stay_score
    )

    return {
        "overall_score": round(overall, 4),
        "satisfaction_score": round(sat_score, 3),
        "time_efficiency": round(time_score, 3),
        "distance_efficiency": round(dist_score, 3),
        "budget_score": round(cost_score, 3),
        "popular_score": round(popular_score, 3),
        "weather_safety": round(weather_score, 3),
        "stay_balance": round(stay_score, 3),
        "total_time_h": round(total_time / 60, 2),
        "total_distance_km": round(total_dist, 1),
        "total_satisfaction": round(total_sat, 2),
        "total_cost": round(total_cost, 0),
    }


def explain_route(G, route_result, scores, weights, weather_report=None):
    """Generate a short human-readable explanation of why the route is suitable."""
    route = route_result["route"]
    parts = []

    parts.append(f"This route visits {len(route)} places in a logical order that balances your priorities.")

    # Preference emphasis
    top_pref = max(weights, key=weights.get)
    if top_pref == "attraction":
        parts.append("High weight on attraction → places with stronger visitor satisfaction are preferred.")
    elif top_pref == "time":
        parts.append("High weight on travel time → the sequence minimises long road legs where possible.")
    elif top_pref == "budget":
        parts.append("High weight on budget → lower-cost attractions are favoured.")
    elif top_pref == "popular":
        parts.append("High weight on popular routes → the order is influenced by real past tourist sequences.")

    # Time / distance
    t = scores["total_time_h"]
    d = scores["total_distance_km"]
    parts.append(f"Total estimated time is {t} hours covering about {d} km.")

    if scores["time_efficiency"] >= 0.6:
        parts.append("Travel time is reasonably efficient for the number of stops.")
    elif scores["time_efficiency"] < 0.4:
        parts.append("Note: some legs are long because the selected places are geographically spread out (especially east-coast / highland locations).")

    # Weather
    if weather_report and scores["weather_safety"] < 1.0:
        bad = [p for p in route if weather_report.get(p, {}).get("is_bad")]
        if bad:
            parts.append(f"Weather caution: current conditions suggest possible rain/storm near {', '.join(bad[:3])}.")
    else:
        parts.append("No severe weather warnings currently affect the selected places.")

    # Day feasibility
    parts.append("The full route is later split into daily itineraries according to your hours-per-day limit.")

    return " ".join(parts)
