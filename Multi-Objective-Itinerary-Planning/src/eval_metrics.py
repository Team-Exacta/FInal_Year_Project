"""
Paper-style evaluation metrics for the Sri Lanka ACO itinerary system.

Aligned with Saeki et al. (IEEE Access 2022) multi-objective trip planning:
  - POI satisfaction (higher better)
  - POI cost (lower better)
  - Past-trip similarity (higher better)
  - Travel time / distance
  - Preference / overall score
  - Comparison vs baselines (Nearest Neighbour + Dijkstra edges + 2-opt)
  - Comparison vs exhaustive true best (when n is small)
  - Optional multi-objective set indicators GD / IGD / MS (when many routes)

References (conceptually):
  Saeki et al., Multi-Objective Trip Planning Based on Ant Colony Optimization
  Utilizing Trip Records, IEEE Access, 2022.
"""

from __future__ import annotations

import itertools
import math
from evaluation import (
    max_historical_similarity,
    load_historical_routes,
    _score_routes,
)
from aco import route_metrics
from baseline import build_baseline_routes


def paper_style_metrics(route_result):
    """
    Extract the same style of metrics used in the IEEE paper tables.
    """
    return {
        "POI satisfaction": round(route_result.get("total_satisfaction", 0), 2),
        "POI cost": round(route_result.get("total_cost", 0), 0),
        "Similarity": round(route_result.get("historical_similarity", 0), 4),
        "Travel time (h)": round(route_result.get("total_time_min", 0) / 60, 2),
        "Distance (km)": round(route_result.get("total_distance_km", 0), 1),
        "#POIs": len(route_result.get("route", [])),
        "Preference score": round(route_result.get("preference_score", 0), 4),
    }


def _euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def generational_distance(approx, reference):
    """
    GD: average distance from each approximate solution to nearest reference point.
    Lower = better convergence. Points are (sat_norm, cost_norm, sim_norm).
    """
    if not approx or not reference:
        return None
    total = 0.0
    for p in approx:
        total += min(_euclidean(p, r) for r in reference) ** 2
    return math.sqrt(total / len(approx))


def inverted_generational_distance(approx, reference):
    """IGD: average distance from each reference point to nearest approximate. Lower better."""
    if not approx or not reference:
        return None
    total = 0.0
    for r in reference:
        total += min(_euclidean(r, p) for p in approx)
    return total / len(reference)


def maximum_spread(points):
    """MS: spread of the front. Higher = wider coverage of objectives."""
    if not points or len(points[0]) == 0:
        return None
    k = len(points[0])
    s = 0.0
    for i in range(k):
        vals = [p[i] for p in points]
        s += (max(vals) - min(vals)) ** 2
    return math.sqrt(s)


def _normalise_front(routes):
    """Turn routes into normalised (sat↑, cost↓→↑, sim↑) points in [0,1]."""
    if not routes:
        return []
    sats = [r["total_satisfaction"] for r in routes]
    costs = [r["total_cost"] for r in routes]
    sims = [r.get("historical_similarity", 0) for r in routes]

    def norm(val, lo, hi, higher_better=True):
        if hi == lo:
            return 1.0
        v = (val - lo) / (hi - lo)
        v = max(0.0, min(1.0, v))
        return v if higher_better else 1.0 - v

    return [
        (
            norm(r["total_satisfaction"], min(sats), max(sats), True),
            norm(r["total_cost"], min(costs), max(costs), False),
            norm(r.get("historical_similarity", 0), min(sims), max(sims), True),
        )
        for r in routes
    ]


def compute_front_indicators(approx_routes, reference_routes=None):
    """
    Compute GD, IGD, MS like the paper.
    If reference_routes is None, use approx as both (MS only is meaningful).
    """
    approx_pts = _normalise_front(approx_routes)
    if not approx_pts:
        return {"GD": None, "IGD": None, "MS": None}
    ms = maximum_spread(approx_pts)
    if reference_routes:
        ref_pts = _normalise_front(reference_routes)
        gd = generational_distance(approx_pts, ref_pts)
        igd = inverted_generational_distance(approx_pts, ref_pts)
    else:
        gd, igd = None, None
    return {
        "GD": round(gd, 4) if gd is not None else None,
        "IGD": round(igd, 4) if igd is not None else None,
        "MS": round(ms, 4) if ms is not None else None,
    }


def full_evaluation(
    G,
    start,
    must_visit,
    weights,
    aco_ranked,
    past_trips_file,
    max_total_time_min=None,
    max_exhaustive=7,
):
    """
    Full evaluation report for one user input:

    1) Paper-style metrics for ACO best
    2) Baseline metrics (NN + Dijkstra edges + 2-opt)
    3) Exhaustive true best (if small must-visit set)
    4) Score gap / Top-1 vs true best
    5) Improvement vs best baseline (same scoring pool)
    6) Optional GD/IGD/MS if enough routes
    """
    historical = load_historical_routes(past_trips_file)
    required = [p for p in must_visit if p != start]

    report = {
        "start": start,
        "must_visit": list(must_visit),
        "n_intermediate": len(required),
    }

    if not aco_ranked:
        report["error"] = "No ACO routes"
        return report

    # Joint pool so preference_score is comparable across methods
    pool = []
    for r in aco_ranked:
        rr = dict(r)
        if "historical_similarity" not in rr:
            rr["historical_similarity"] = max_historical_similarity(
                rr["route"], historical
            )
        rr["_source"] = "ACO"
        pool.append(rr)

    baselines_raw = build_baseline_routes(G, start, must_visit, max_total_time_min)
    for b in baselines_raw:
        b["historical_similarity"] = max_historical_similarity(b["route"], historical)
        b["_source"] = b.get("baseline_name", "Baseline")
        pool.append(b)

    _score_routes(pool, weights)
    pool.sort(key=lambda r: r["preference_score"], reverse=True)

    aco_only = [r for r in pool if r.get("_source") == "ACO"]
    bl_only = [r for r in pool if r.get("_source") != "ACO"]
    aco_best = aco_only[0] if aco_only else pool[0]

    report["ACO"] = paper_style_metrics(aco_best)
    report["ACO"]["route"] = " → ".join(aco_best["route"])

    if bl_only:
        by_name = {}
        for b in bl_only:
            name = b.get("_source", "Baseline")
            if name not in by_name or b["preference_score"] > by_name[name]["preference_score"]:
                by_name[name] = b
        report["baselines"] = []
        for name, b in sorted(by_name.items(), key=lambda x: -x[1]["preference_score"]):
            row = paper_style_metrics(b)
            row["name"] = name
            row["route"] = " → ".join(b["route"])
            report["baselines"].append(row)
        best_bl = report["baselines"][0]
        report["best_baseline"] = best_bl
        aco_s = aco_best["preference_score"]
        bl_s = best_bl["Preference score"]
        if bl_s and abs(bl_s) > 1e-12:
            report["improvement_vs_baseline_pct"] = round(
                (aco_s - bl_s) / abs(bl_s) * 100, 2
            )
        else:
            report["improvement_vs_baseline_pct"] = None
    else:
        report["baselines"] = []
        report["best_baseline"] = None
        report["improvement_vs_baseline_pct"] = None

    # --- Exhaustive true best (small n) ---
    report["exhaustive"] = None
    if 1 <= len(required) <= max_exhaustive:
        all_perms = []
        for order in itertools.permutations(required):
            route = [start] + list(order)
            m = route_metrics(G, route)
            if m is None:
                continue  # skip non-actual (missing road edge) routes
            if max_total_time_min and m["total_time_min"] > max_total_time_min:
                continue
            m["historical_similarity"] = max_historical_similarity(route, historical)
            all_perms.append(m)
        if all_perms:
            _score_routes(all_perms, weights)
            all_perms.sort(key=lambda r: r["preference_score"], reverse=True)
            true_best = all_perms[0]
            true_score = true_best["preference_score"]
            true_key = tuple(true_best["route"])
            best_keys = {
                tuple(r["route"])
                for r in all_perms
                if abs(r["preference_score"] - true_score) < 1e-9
            }

            score_of = {tuple(r["route"]): r["preference_score"] for r in all_perms}
            rank_of = {}
            for i, r in enumerate(all_perms):
                k = tuple(r["route"])
                if k not in rank_of:
                    rank_of[k] = i + 1

            aco_key = tuple(aco_best["route"])
            aco_fair = score_of.get(aco_key)
            aco_rank = rank_of.get(aco_key)
            if aco_fair is None:
                tmp = route_metrics(G, list(aco_best["route"]))
                tmp["historical_similarity"] = max_historical_similarity(
                    list(aco_best["route"]), historical
                )
                pool2 = all_perms + [tmp]
                _score_routes(pool2, weights)
                aco_fair = tmp["preference_score"]

            gap = max(0.0, (true_score - aco_fair) / true_score) if true_score else 0.0
            top1 = 1 if aco_key in best_keys else 0
            top3_keys = {tuple(r["route"]) for r in aco_only[:3]}
            top3 = 1 if (true_key in top3_keys or bool(best_keys & top3_keys)) else 0

            report["exhaustive"] = {
                **paper_style_metrics(true_best),
                "route": " → ".join(true_best["route"]),
                "n_feasible": len(all_perms),
                "n_total_orders": math.prod(range(1, len(required) + 1)),
                "ACO_fair_score": round(aco_fair, 4),
                "ACO_rank": aco_rank,
                "Top-1": top1,
                "Top-3 hit": top3,
                "Score gap %": round(gap * 100, 2),
            }

            ref = all_perms[: max(5, len(all_perms) // 5)]
            report["front_indicators"] = compute_front_indicators(aco_only, ref)
        else:
            report["exhaustive"] = {"error": "No feasible orders under time limit"}
    else:
        report["exhaustive"] = {
            "skipped": True,
            "reason": f"Must-visit size {len(required)} > {max_exhaustive}; exhaustive not run",
        }
        report["front_indicators"] = compute_front_indicators(aco_only, None)

    return report


def report_to_rows(report):
    """Flatten report into table-friendly rows for Streamlit / CSV."""
    rows = []
    if "ACO" in report:
        r = dict(report["ACO"])
        r["Method"] = "ACO (ours)"
        rows.append(r)
    for b in report.get("baselines") or []:
        r = dict(b)
        r["Method"] = b.get("name", "Baseline")
        rows.append(r)
    ex = report.get("exhaustive")
    if ex and "route" in ex:
        r = {k: ex[k] for k in (
            "POI satisfaction", "POI cost", "Similarity", "Travel time (h)",
            "Distance (km)", "#POIs", "Preference score", "route",
        ) if k in ex}
        r["Method"] = "True best (exhaustive)"
        rows.append(r)
    return rows
