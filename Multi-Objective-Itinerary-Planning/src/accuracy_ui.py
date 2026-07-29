"""Accuracy + evaluation for the UI: exhaustive Top-1/gap + baseline comparison."""
import itertools
from aco import route_metrics
from evaluation import (
    load_historical_routes,
    max_historical_similarity,
    _score_routes,
)

try:
    from eval_metrics import full_evaluation
except Exception:
    full_evaluation = None

MAX_INTERMEDIATE_FOR_EXHAUSTIVE = 7


def can_compute_accuracy(start, must_visit):
    required = [p for p in must_visit if p != start]
    return 1 <= len(required) <= MAX_INTERMEDIATE_FOR_EXHAUSTIVE


def permutation_count(start, must_visit):
    required = [p for p in must_visit if p != start]
    n = len(required)
    if n <= 0:
        return 0
    total = 1
    for i in range(2, n + 1):
        total *= i
    return total


def compute_accuracy_for_input(
    G,
    start,
    must_visit,
    weights,
    aco_ranked,
    past_trips_file,
    max_total_time_min=None,
):
    required = [p for p in must_visit if p != start]
    if not aco_ranked:
        return None

    result = {
        "applicable": True,
        "n_intermediate": len(required),
        "top1": None,
        "top3_hit": None,
        "score_gap": None,
        "aco_rank": None,
        "true_best_route": None,
        "comparison_table": None,
        "vs_baseline_pct": None,
    }

    # ---- Baselines + paper metrics via full_evaluation ----
    if full_evaluation is not None:
        try:
            report = full_evaluation(
                G, start, must_visit, weights, aco_ranked, past_trips_file,
                max_total_time_min=max_total_time_min,
            )
            from eval_metrics import report_to_rows
            import pandas as pd
            rows = report_to_rows(report)
            if rows:
                result["comparison_table"] = pd.DataFrame(rows)
            result["vs_baseline_pct"] = report.get("improvement_vs_baseline_pct")
            if report.get("exhaustive") and "route" in report["exhaustive"]:
                ex = report["exhaustive"]
                result["top1"] = ex.get("Top-1")
                result["top3_hit"] = ex.get("Top-3 hit")
                gap_pct = ex.get("Score gap %")
                result["score_gap"] = (gap_pct / 100.0) if gap_pct is not None else None
                result["aco_rank"] = ex.get("ACO_rank")
                result["true_best_route"] = ex["route"].split(" → ") if ex.get("route") else None
                result["true_best_score"] = ex.get("Preference score")
            return result
        except Exception:
            pass

    # ---- Fallback: exhaustive only ----
    if not required or len(required) > MAX_INTERMEDIATE_FOR_EXHAUSTIVE:
        return result if result.get("comparison_table") is not None else None

    historical = load_historical_routes(past_trips_file)
    all_perms = []
    for order in itertools.permutations(required):
        route = [start] + list(order)
        m = route_metrics(G, route)
        if m is None:
            continue
        if max_total_time_min and m["total_time_min"] > max_total_time_min:
            continue
        m["historical_similarity"] = max_historical_similarity(route, historical)
        all_perms.append(m)

    if not all_perms:
        result["error"] = "No feasible orders"
        return result

    _score_routes(all_perms, weights)
    all_perms.sort(key=lambda r: r["preference_score"], reverse=True)
    true_best = all_perms[0]
    true_score = true_best["preference_score"]
    true_route = true_best["route"]
    true_key = tuple(true_route)
    best_keys = {
        tuple(r["route"])
        for r in all_perms
        if abs(r["preference_score"] - true_score) < 1e-9
    }
    rank_of = {}
    score_of = {}
    for i, r in enumerate(all_perms):
        key = tuple(r["route"])
        if key not in rank_of:
            rank_of[key] = i + 1
            score_of[key] = r["preference_score"]

    aco_route = aco_ranked[0]["route"]
    aco_key = tuple(aco_route)
    if aco_key in score_of:
        aco_fair_score = score_of[aco_key]
        aco_rank = rank_of[aco_key]
    else:
        aco_fair_score = aco_ranked[0].get("preference_score", 0)
        aco_rank = None

    top1 = 1 if aco_key in best_keys else 0
    aco_top3_keys = {tuple(r["route"]) for r in aco_ranked[:3]}
    top3_hit = 1 if (true_key in aco_top3_keys or bool(best_keys & aco_top3_keys)) else 0
    gap = (true_score - aco_fair_score) / true_score if true_score > 0 else 0.0
    gap = max(0.0, gap)

    result.update({
        "true_best_score": round(true_score, 4),
        "true_best_route": true_route,
        "aco_fair_score": round(aco_fair_score, 4),
        "aco_rank": aco_rank,
        "top1": top1,
        "top3_hit": top3_hit,
        "score_gap": round(gap, 4),
        "score_gap_pct": round(gap * 100, 2),
    })
    return result
