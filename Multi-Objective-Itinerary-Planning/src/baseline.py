"""
Baseline route builders for fair comparison with ACO.

1) Dijkstra shortest path on the time (or distance) graph between two nodes
2) Nearest-neighbour ordering of must-visit places (classic constructive baseline)
3) Optional 2-opt improvement on the NN order

These are single-objective / constructive baselines — not multi-objective like ACO.
"""

from __future__ import annotations

import heapq
from aco import route_metrics, _edge


def dijkstra_path(G, source, target, weight_key="travel_time_min"):
    """
    Shortest path from source to target using Dijkstra.
    weight_key: 'travel_time_min' or 'distance_km'
    Returns (path_list, total_weight) or (None, inf) if unreachable.
    """
    if source not in G or target not in G:
        return None, float("inf")
    if source == target:
        return [source], 0.0

    dist = {source: 0.0}
    prev = {source: None}
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == target:
            break
        for v in G.successors(u):
            edge = G[u][v]
            w = float(edge.get(weight_key, 0) or 0)
            if w <= 0:
                continue
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
        # undirected fallback
        for v in G.predecessors(u):
            if G.has_edge(u, v):
                continue
            edge = G[v][u]
            w = float(edge.get(weight_key, 0) or 0)
            if w <= 0:
                continue
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if target not in dist:
        return None, float("inf")

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path, dist[target]


def _pairwise_weight(G, a, b, weight_key="travel_time_min"):
    edge = _edge(G, a, b)
    if edge:
        return float(edge.get(weight_key, 0) or 0)
    # fallback: Dijkstra if no direct edge
    _, w = dijkstra_path(G, a, b, weight_key)
    return w


def nearest_neighbour_order(G, start, must_visit, weight_key="travel_time_min"):
    """
    Greedy nearest-neighbour: always go to the closest unvisited must-visit place.
    Returns ordered list: [start] + ordered must-visit.
    """
    required = [p for p in must_visit if p != start and p in G]
    if not required:
        return [start]

    route = [start]
    remaining = set(required)
    current = start

    while remaining:
        best, best_w = None, float("inf")
        for cand in remaining:
            w = _pairwise_weight(G, current, cand, weight_key)
            if w < best_w:
                best_w, best = w, cand
        if best is None:
            # unreachable leftovers — append in arbitrary order
            route.extend(sorted(remaining))
            break
        route.append(best)
        remaining.remove(best)
        current = best
    return route


def two_opt(G, route, weight_key="travel_time_min", max_passes=20):
    """
    Classic 2-opt local search on a fixed set of places (start fixed at index 0).
    Minimises sum of pairwise weights along the tour order.
    """
    if len(route) < 4:
        return list(route)

    best = list(route)
    n = len(best)

    def tour_cost(r):
        c = 0.0
        for i in range(len(r) - 1):
            c += _pairwise_weight(G, r[i], r[i + 1], weight_key)
        return c

    best_cost = tour_cost(best)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # reverse segment i..j
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                c = tour_cost(candidate)
                if c + 1e-9 < best_cost:
                    best = candidate
                    best_cost = c
                    improved = True
                    break
            if improved:
                break
    return best


def build_baseline_routes(G, start, must_visit, max_total_time_min=None):
    """
    Build several baseline routes for comparison with ACO.
    Returns list of route_metrics dicts labelled with baseline name.
    """
    results = []

    # 1) Nearest neighbour on time
    nn_time = nearest_neighbour_order(G, start, must_visit, "travel_time_min")
    m = route_metrics(G, nn_time)
    if m is not None and (not max_total_time_min or m["total_time_min"] <= max_total_time_min):
        m["baseline_name"] = "NearestNeighbour-Time"
        results.append(m)

    # 2) Nearest neighbour on distance
    nn_dist = nearest_neighbour_order(G, start, must_visit, "distance_km")
    m = route_metrics(G, nn_dist)
    if m is not None and (not max_total_time_min or m["total_time_min"] <= max_total_time_min):
        m["baseline_name"] = "NearestNeighbour-Distance"
        results.append(m)

    # 3) NN-Time + 2-opt
    opt = two_opt(G, nn_time, "travel_time_min")
    m = route_metrics(G, opt)
    if m is not None and (not max_total_time_min or m["total_time_min"] <= max_total_time_min):
        m["baseline_name"] = "NN-Time+2opt"
        results.append(m)

    # 4) NN-Distance + 2-opt
    opt_d = two_opt(G, nn_dist, "distance_km")
    m = route_metrics(G, opt_d)
    if m is not None and (not max_total_time_min or m["total_time_min"] <= max_total_time_min):
        m["baseline_name"] = "NN-Distance+2opt"
        results.append(m)

    return results
