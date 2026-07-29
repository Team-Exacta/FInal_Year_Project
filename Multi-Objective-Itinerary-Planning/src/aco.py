import itertools
import random
from utils import normalize_weights


def _edge(G, u, v):
    if G.has_edge(u, v):
        return G[u][v]
    if G.has_edge(v, u):
        return G[v][u]
    return None


def is_valid_route(G, route):
    """True only if every consecutive pair has a real road edge in the matrices."""
    if not route or any(p not in G for p in route):
        return False
    for i in range(1, len(route)):
        if not _edge(G, route[i - 1], route[i]):
            return False
    return True


def route_metrics(G, route):
    """
    Metrics for a route using ONLY real matrix edges.
    Returns None if any consecutive pair has no actual road link.
    """
    if not route or any(p not in G for p in route):
        return None

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
            if not edge:
                return None  # not an actual connected road route
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
        "is_actual_route": True,
    }


def estimate_route_time(G, route):
    """Total time for a route; inf if not an actual connected route."""
    m = route_metrics(G, route)
    if m is None:
        return float("inf")
    return m["total_time_min"]


def _nn_order(G, start, places):
    """Nearest-neighbour order of places starting from start. Returns route list."""
    order = [start]
    rem = [p for p in places if p != start]
    cur = start
    while rem:
        candidates = [x for x in rem if _edge(G, cur, x)]
        if not candidates:
            # disconnected – append remaining in given order so caller can reject
            order.extend(rem)
            break
        nxt = min(
            candidates,
            key=lambda x: float(_edge(G, cur, x).get("travel_time_min", 1e18)),
        )
        order.append(nxt)
        rem.remove(nxt)
        cur = nxt
    return order


def _subset_fits(G, start, places, max_total_time_min):
    """True if NN tour of start+places is a real route and fits the time budget."""
    if not places:
        return False
    order = _nn_order(G, start, places)
    t = estimate_route_time(G, order)
    return t != float("inf") and t <= max_total_time_min


def _place_value(G, p, weights):
    sat = float(G.nodes[p].get("satisfaction_score", 0))
    cost = float(G.nodes[p].get("cost", 0))
    w_att = float(weights.get("attraction", 0.25))
    w_bud = float(weights.get("budget", 0.15))
    return w_att * sat * 10 + w_bud * (1.0 / (1.0 + cost / 1000.0))


def _greedy_subset(G, start, ranked, max_total_time_min):
    """Greedy fill: add places in ranked order while the set still fits."""
    selected = []
    for p in ranked:
        trial = selected + [p]
        if _subset_fits(G, start, trial, max_total_time_min):
            selected.append(p)
    return selected


def _try_insert_more(G, start, selected, candidates, max_total_time_min):
    """
    After a greedy pack, try inserting any remaining candidates again
    (order may allow more places once set is known).
    """
    selected = list(selected)
    changed = True
    while changed:
        changed = False
        for p in candidates:
            if p in selected:
                continue
            trial = selected + [p]
            if _subset_fits(G, start, trial, max_total_time_min):
                selected.append(p)
                changed = True
    return selected


def _maximize_places_pack(G, start, required, max_total_time_min, weights=None):
    """
    Prefer MAXIMUM number of places that fit the time budget.

    Tries several orderings that pack denser routes (near / short visits first),
    then keeps the largest set; ties broken by total satisfaction.
    """
    weights = weights or {}
    if not required:
        return []

    def visit_dur(p):
        return float(G.nodes[p].get("duration_time_min", 90))

    def near_start(p):
        return float((_edge(G, start, p) or {}).get("travel_time_min", 1e18))

    def sat_of(p):
        return float(G.nodes[p].get("satisfaction_score", 0))

    def density(p):
        # higher satisfaction per minute of visit → pack more value densely
        d = max(visit_dur(p), 1.0)
        return sat_of(p) / d

    rankings = [
        sorted(required, key=near_start),  # closest first → usually max count
        sorted(required, key=visit_dur),   # shortest visits first
        sorted(required, key=lambda p: (near_start(p), visit_dur(p))),
        sorted(required, key=density, reverse=True),
        sorted(required, key=lambda p: _place_value(G, p, weights), reverse=True),
        sorted(required, key=sat_of, reverse=True),
        sorted(required, key=lambda p: float(G.nodes[p].get("cost", 0))),
    ]

    best = []
    best_sat = -1.0
    for ranked in rankings:
        sel = _greedy_subset(G, start, ranked, max_total_time_min)
        sel = _try_insert_more(G, start, sel, required, max_total_time_min)
        sat = sum(sat_of(p) for p in sel)
        if len(sel) > len(best) or (len(sel) == len(best) and sat > best_sat):
            best = sel
            best_sat = sat

    # If still empty, take any single place that fits
    if not best:
        for p in sorted(required, key=near_start):
            if _subset_fits(G, start, [p], max_total_time_min):
                return [p]
    return best


def select_places_for_budget(G, start, must_visit, max_total_time_min, weights=None):
    """
    If all must-visit places fit, return them all.
    Otherwise return the largest subset that fits (max places, then satisfaction).

    Returns (selected_list, info_dict).
    """
    weights = weights or {}
    required = list(dict.fromkeys([p for p in must_visit if p != start and p in G]))
    if not required:
        return [], {"fitted_all": True, "dropped": [], "reason": "no must-visit places"}

    full_order = _nn_order(G, start, required)
    full_time = estimate_route_time(G, full_order)

    if full_time != float("inf") and full_time <= max_total_time_min:
        return required, {
            "fitted_all": True,
            "dropped": [],
            "selected": required,
            "estimated_time_h": round(full_time / 60, 2),
            "budget_h": round(max_total_time_min / 60, 2),
            "reason": "all must-visit places fit in the time budget",
        }

    selected = _maximize_places_pack(G, start, required, max_total_time_min, weights)

    dropped = [p for p in required if p not in selected]
    final_order = _nn_order(G, start, selected) if selected else [start]
    est = estimate_route_time(G, final_order)

    return selected, {
        "fitted_all": False,
        "dropped": dropped,
        "selected": selected,
        "estimated_time_h": round(est / 60, 2) if est != float("inf") else None,
        "budget_h": round(max_total_time_min / 60, 2),
        "full_needed_h": round(full_time / 60, 2) if full_time != float("inf") else None,
        "reason": (
            f"Packed maximum {len(selected)} of {len(required)} places into "
            f"{round(max_total_time_min/60, 1)} h budget."
        ),
    }


def generate_multiple_subset_options(
    G, start, must_visit, max_total_time_min, weights=None, max_options=6
):
    """
    Build subsets that fit the time budget.

    PRIMARY goal: use as MANY places as possible (not minimal routes).
    Options are sorted by (#places desc, satisfaction desc).
    Single-place options are only kept if nothing larger fits.
    """
    weights = weights or {}
    required = list(dict.fromkeys([p for p in must_visit if p != start and p in G]))
    if not required:
        return []

    full_order = _nn_order(G, start, required)
    full_time = estimate_route_time(G, full_order)
    budget_h = round(max_total_time_min / 60, 2)

    if full_time != float("inf") and full_time <= max_total_time_min:
        sat = sum(float(G.nodes[p].get("satisfaction_score", 0)) for p in required)
        return [{
            "selected": required,
            "dropped": [],
            "estimated_time_h": round(full_time / 60, 2),
            "total_satisfaction": round(sat, 2),
            "label": f"All {len(required)} places (fits budget)",
            "fitted_all": True,
            "budget_h": budget_h,
            "full_needed_h": round(full_time / 60, 2),
        }]

    def visit_dur(p):
        return float(G.nodes[p].get("duration_time_min", 90))

    def near_start(p):
        return float((_edge(G, start, p) or {}).get("travel_time_min", 1e18))

    def sat_of(p):
        return float(G.nodes[p].get("satisfaction_score", 0))

    ranked_near = sorted(required, key=near_start)
    ranked_short = sorted(required, key=visit_dur)
    ranked_sat = sorted(required, key=sat_of, reverse=True)
    ranked_value = sorted(required, key=lambda p: _place_value(G, p, weights), reverse=True)
    ranked_cheap = sorted(required, key=lambda p: float(G.nodes[p].get("cost", 0)))
    ranked_density = sorted(
        required,
        key=lambda p: sat_of(p) / max(visit_dur(p), 1.0),
        reverse=True,
    )
    ranked_near_short = sorted(required, key=lambda p: (near_start(p), visit_dur(p)))

    strategies = [
        ("Max places (closest first)", ranked_near),
        ("Max places (shortest visits)", ranked_short),
        ("Max places (near + short)", ranked_near_short),
        ("Max places (satisfaction density)", ranked_density),
        ("Max places (your priorities)", ranked_value),
        ("Max places (satisfaction)", ranked_sat),
        ("Max places (lowest cost)", ranked_cheap),
    ]

    # Grow from each nearby place as seed (still maximising fill)
    for p in ranked_near[: min(5, len(ranked_near))]:
        others = [x for x in ranked_near if x != p]
        strategies.append((f"Max pack from {p}", [p] + others))

    options = []
    seen = set()

    def _add(selected, label):
        if not selected:
            return
        key = frozenset(selected)
        if key in seen:
            return
        # Always try to insert more before accepting
        selected = _try_insert_more(G, start, selected, required, max_total_time_min)
        key = frozenset(selected)
        if key in seen:
            return
        if not _subset_fits(G, start, selected, max_total_time_min):
            return
        seen.add(key)
        order = _nn_order(G, start, selected)
        est = estimate_route_time(G, order)
        sat = sum(sat_of(p) for p in selected)
        options.append({
            "selected": list(selected),
            "dropped": [p for p in required if p not in selected],
            "estimated_time_h": round(est / 60, 2) if est != float("inf") else None,
            "total_satisfaction": round(sat, 2),
            "label": label,
            "fitted_all": False,
            "budget_h": budget_h,
            "full_needed_h": (
                round(full_time / 60, 2) if full_time != float("inf") else None
            ),
            "n_places": len(selected),
        })

    for label, ranking in strategies:
        sel = _greedy_subset(G, start, ranking, max_total_time_min)
        sel = _try_insert_more(G, start, sel, required, max_total_time_min)
        _add(sel, label)

    # Dedicated maximiser
    best_pack = _maximize_places_pack(G, start, required, max_total_time_min, weights)
    _add(best_pack, "Maximum places pack")

    if not options:
        for p in ranked_near:
            if _subset_fits(G, start, [p], max_total_time_min):
                _add([p], f"Only place that fits: {p}")
                break

    # Sort: MOST places first, then highest satisfaction
    options.sort(
        key=lambda o: (len(o["selected"]), o["total_satisfaction"]),
        reverse=True,
    )

    # Drop tiny options if we already have larger packs (avoid "2-place only" as primary)
    if options:
        max_n = len(options[0]["selected"])
        # Keep options with at least max_n - 1 places when max_n >= 3
        if max_n >= 3:
            filtered = [o for o in options if len(o["selected"]) >= max_n - 1]
            if filtered:
                options = filtered

    return options[:max_options]




def optimize_required_route(G, start, required_places, max_total_time_min=None, exhaustive_limit=8):
    """Return the lowest-time feasible ordering of required places from start."""
    required = list(dict.fromkeys([place for place in required_places if place != start]))
    if not required:
        return route_metrics(G, [start])

    def feasible_metrics(route):
        metrics = route_metrics(G, route)
        if metrics is None:
            return None
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
            if test is None:
                continue
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
            if (
                route is not None
                and len(route["route"]) > 1
                and all(place in route["route"] for place in must_visit if place != start)
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
        if m is None:
            continue
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
