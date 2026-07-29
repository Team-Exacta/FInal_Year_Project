import itertools
import random
from utils import normalize_weights


def _edge(G, u, v):
    if G.has_edge(u, v):
        return G[u][v]
    if G.has_edge(v, u):
        return G[v][u]
    return None


def route_metrics(G, route):
    total_distance = total_time = total_cost = total_satisfaction = pheromone_sum = 0.0
    legs = []

    for index, place in enumerate(route):
        node = G.nodes[place]
        visit_time = float(node.get("duration_time_min", 0))
        total_cost += float(node.get("cost", 0))
        total_satisfaction += float(node.get("satisfaction_score", 0))
        total_time += visit_time

        leg = {
            "Order": index + 1,
            "Place": place,
            "Visit Time (min)": round(visit_time, 2),
            "Travel From Previous (min)": 0.0,
            "Distance From Previous (km)": 0.0,
        }
        if index > 0:
            edge = _edge(G, route[index - 1], place)
            if edge:
                travel_time = float(edge.get("travel_time_min", 0))
                distance = float(edge.get("distance_km", 0))
                total_time += travel_time
                total_distance += distance
                pheromone_sum += float(edge.get("pheromone", 1.0))
                leg["Travel From Previous (min)"] = round(travel_time, 2)
                leg["Distance From Previous (km)"] = round(distance, 2)
        legs.append(leg)

    return {
        "route": route,
        "legs": legs,
        "total_distance_km": total_distance,
        "total_time_min": total_time,
        "total_cost": total_cost,
        "total_satisfaction": total_satisfaction,
        "pheromone_sum": pheromone_sum,
    }


def optimize_required_route(G, start, required_places, max_total_time_min=None, exhaustive_limit=8):
    """Return the lowest-time feasible ordering of required places from start."""
    required = list(dict.fromkeys([place for place in required_places if place != start]))
    if not required:
        return route_metrics(G, [start])

    def feasible_metrics(route):
        metrics = route_metrics(G, route)
        if max_total_time_min and metrics["total_time_min"] > max_total_time_min:
            return None
        return metrics

    if len(required) <= exhaustive_limit:
        best = None
        for order in itertools.permutations(required):
            metrics = feasible_metrics([start] + list(order))
            if metrics and (best is None or metrics["total_time_min"] < best["total_time_min"]):
                best = metrics
        return best

    # Greedy nearest-neighbour for larger sets
    route = [start]
    unvisited = required[:]
    current = start
    while unvisited:
        next_place = min(
            unvisited,
            key=lambda place: float((_edge(G, current, place) or {}).get("travel_time_min", float("inf"))),
        )
        route.append(next_place)
        unvisited.remove(next_place)
        current = next_place
    return feasible_metrics(route)


def generate_diverse_required_routes(G, start, required_places, max_total_time_min=None, max_variants=6):
    """
    Produce several distinct feasible orderings of the required places.
    Ensures the ranked list has real alternatives even when must-visits are set.
    """
    required = list(dict.fromkeys([p for p in required_places if p != start]))
    results = []
    seen = set()

    def _add(metrics):
        if not metrics:
            return
        key = tuple(metrics["route"])
        if key in seen:
            return
        seen.add(key)
        results.append(metrics)

    # 1) Exact best (or greedy) order
    _add(optimize_required_route(G, start, required, max_total_time_min))

    # 2) All permutations when the set is small
    if 1 < len(required) <= 7:
        for order in itertools.permutations(required):
            m = route_metrics(G, [start] + list(order))
            if max_total_time_min and m["total_time_min"] > max_total_time_min:
                continue
            _add(m)
            if len(results) >= max_variants:
                return results

    # 3) Randomised greedy starts
    rng = random.Random(123)
    for _ in range(max_variants * 4):
        if len(results) >= max_variants:
            break
        unvisited = required[:]
        rng.shuffle(unvisited)
        route = [start]
        current = start
        if unvisited:
            first = unvisited.pop(0)
            route.append(first)
            current = first
        while unvisited:
            next_place = min(
                unvisited,
                key=lambda p: float((_edge(G, current, p) or {}).get("travel_time_min", float("inf"))),
            )
            route.append(next_place)
            unvisited.remove(next_place)
            current = next_place
        m = route_metrics(G, route)
        if max_total_time_min and m["total_time_min"] > max_total_time_min:
            continue
        _add(m)

    # 4) Adjacent-swap neighbourhood of the best route
    if results:
        base = list(results[0]["route"])
        for i in range(1, len(base) - 1):
            variant = base[:]
            variant[i], variant[i + 1] = variant[i + 1], variant[i]
            m = route_metrics(G, variant)
            if max_total_time_min and m["total_time_min"] > max_total_time_min:
                continue
            _add(m)
            if len(results) >= max_variants:
                break

    return results


def _ordered_candidates(unvisited, pending_must_visit):
    must = [p for p in unvisited if p in pending_must_visit]
    optional = [p for p in unvisited if p not in pending_must_visit]
    return must if must else optional


def construct_route(G, start, candidate_places, weights, max_total_time_min, max_places, must_visit=None, rng=None):
    rng = rng or random
    route = [start]
    unvisited = [p for p in candidate_places if p != start]
    must_visit = [p for p in (must_visit or []) if p != start]
    current = start

    while unvisited and len(route) < max_places:
        pending_must_visit = [p for p in must_visit if p not in route]
        pool = _ordered_candidates(unvisited, pending_must_visit)
        candidates, scores = [], []

        for nxt in pool:
            edge = _edge(G, current, nxt)
            if not edge:
                continue
            test = route_metrics(G, route + [nxt])
            if max_total_time_min and test["total_time_min"] > max_total_time_min:
                continue

            pheromone = float(edge.get("pheromone", 1.0))
            travel_time = max(float(edge.get("travel_time_min", 1)), 1)
            distance = max(float(edge.get("distance_km", 1)), 1)
            attraction = float(G.nodes[nxt].get("satisfaction_score", 0))
            cost = float(G.nodes[nxt].get("cost", 0))
            budget = 1 / (1 + cost / 1000)
            must_bonus = 100 if nxt in pending_must_visit else 0
            score = (
                must_bonus
                + weights["popular"] * pheromone
                + weights["time"] * (100 / travel_time)
                + weights["attraction"] * attraction * 20
                + weights["budget"] * budget * 20
                + 5 / distance
            )
            candidates.append(nxt)
            scores.append(max(score, 0.0001))

        if not candidates:
            break

        selected = rng.choices(candidates, weights=scores, k=1)[0]
        route.append(selected)
        unvisited.remove(selected)
        current = selected

    return route_metrics(G, route)


def evaporate(G, rho=0.1):
    for u, v in G.edges():
        G[u][v]["pheromone"] = max(0.01, float(G[u][v].get("pheromone", 1.0)) * (1 - rho))


def deposit(G, route, amount):
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if G.has_edge(u, v):
            G[u][v]["pheromone"] += amount
        elif G.has_edge(v, u):
            G[v][u]["pheromone"] += amount


def run_aco(
    G,
    start,
    candidate_places,
    raw_preferences,
    num_ants=100,
    iterations=40,
    max_total_time_min=None,
    max_places=10,
    must_visit=None,
    seed=42,
    copy_pheromone=True,
):
    """
    Run ACO. When copy_pheromone=True, pheromones are restored after the run
    so specialised preference runs stay independent.
    """
    weights = normalize_weights(raw_preferences)
    rng = random.Random(seed)
    if start not in candidate_places:
        candidate_places = [start] + list(candidate_places)
    all_routes = []
    must_visit = must_visit or []

    pheromone_backup = None
    if copy_pheromone:
        pheromone_backup = {(u, v): float(d.get("pheromone", 1.0)) for u, v, d in G.edges(data=True)}

    for _ in range(iterations):
        iteration_routes = []
        for _ant in range(num_ants):
            route = construct_route(
                G, start, candidate_places, weights, max_total_time_min, max_places, must_visit, rng
            )
            if len(route["route"]) > 1 and all(
                place in route["route"] for place in must_visit if place != start
            ):
                iteration_routes.append(route)
                all_routes.append(route)
        if iteration_routes:
            best = max(
                iteration_routes,
                key=lambda x: (x["total_satisfaction"] / max(x["total_time_min"], 1.0))
                * (1.0 + 0.05 * len(x["route"])),
            )
            evaporate(G)
            deposit(G, best["route"], best["total_satisfaction"] / max(best["total_time_min"], 1))

    if pheromone_backup is not None:
        for (u, v), val in pheromone_backup.items():
            if G.has_edge(u, v):
                G[u][v]["pheromone"] = val

    return all_routes, weights


def generate_route_variants(G, base_route, max_total_time_min=None, max_variants=5):
    """Create distinct neighbour routes by adjacent swaps / segment reversals."""
    if not base_route or len(base_route) < 3:
        return []
    results = []
    seen = {tuple(base_route)}
    route = list(base_route)

    for i in range(1, len(route) - 1):
        variant = route[:]
        variant[i], variant[i + 1] = variant[i + 1], variant[i]
        key = tuple(variant)
        if key in seen:
            continue
        m = route_metrics(G, variant)
        if max_total_time_min and m["total_time_min"] > max_total_time_min:
            continue
        seen.add(key)
        results.append(m)
        if len(results) >= max_variants:
            return results

    for length in (2, 3, 4):
        for i in range(1, len(route) - length):
            variant = route[:i] + list(reversed(route[i : i + length])) + route[i + length :]
            key = tuple(variant)
            if key in seen:
                continue
            m = route_metrics(G, variant)
            if max_total_time_min and m["total_time_min"] > max_total_time_min:
                continue
            seen.add(key)
            results.append(m)
            if len(results) >= max_variants:
                return results

    return results
