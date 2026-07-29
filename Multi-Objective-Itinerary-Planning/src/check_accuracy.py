"""
Offline evaluation script (paper-style metrics + baselines + exhaustive accuracy).

Usage (from project root):
  python src/check_accuracy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from build_graph import build_graph
from aco import (
    run_aco,
    generate_diverse_required_routes,
    optimize_required_route,
    generate_route_variants,
)
from evaluation import evaluate_and_rank
from trips import initialize_pheromones_from_past_trips
from utils import normalize_weights
from eval_metrics import full_evaluation, report_to_rows
import pandas as pd

POI_FILE = ROOT / "data" / "pois.csv"
DISTANCE_FILE = ROOT / "data" / "route_distance_matrix.csv"
DURATION_FILE = ROOT / "data" / "route_duration_matrix.csv"
PAST_TRIPS_FILE = ROOT / "data" / "past_trips_dataset.csv"


def load_graph():
    G = build_graph(str(POI_FILE), str(DISTANCE_FILE), str(DURATION_FILE))
    G, matched, missing = initialize_pheromones_from_past_trips(
        G, str(PAST_TRIPS_FILE), boost=5.0
    )
    print(f"Graph: {G.number_of_nodes()} POIs, {G.number_of_edges()} edges")
    print(f"Pheromone boost: matched={matched}, missing={missing}")
    return G


def run_aco_pipeline(G, start, must_visit, preferences, max_total_time_min, max_places,
                     num_ants=40, iterations=15):
    weights = normalize_weights(preferences)
    candidate_places = list(dict.fromkeys([start] + must_visit))
    all_routes = []

    main_routes, _ = run_aco(
        G, start, candidate_places, preferences,
        num_ants, iterations, max_total_time_min, max_places, must_visit,
        seed=42, copy_pheromone=True,
    )
    all_routes.extend(main_routes)

    if must_visit:
        all_routes.extend(
            generate_diverse_required_routes(G, start, must_visit, max_total_time_min, 8)
        )
        exact = optimize_required_route(G, start, must_visit, max_total_time_min)
        if exact:
            all_routes.append(exact)

    specialised = [
        {"attraction": 10, "budget": 2, "time": 3, "popular": 3},
        {"attraction": 3, "budget": 2, "time": 10, "popular": 3},
        {"attraction": 3, "budget": 10, "time": 3, "popular": 3},
        {"attraction": 3, "budget": 2, "time": 3, "popular": 10},
        {"attraction": 5, "budget": 5, "time": 8, "popular": 5},
    ]
    for i, pref in enumerate(specialised):
        extra, _ = run_aco(
            G, start, candidate_places, pref,
            max(10, num_ants // 3), max(6, iterations // 3),
            max_total_time_min, max_places, must_visit,
            seed=100 + i * 17, copy_pheromone=True,
        )
        all_routes.extend(extra)

    unique, seen = [], set()
    for r in all_routes:
        key = tuple(r["route"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
        if len(unique) >= 8:
            break
    for r in unique:
        for v in generate_route_variants(G, r["route"], max_total_time_min, 4):
            if all(p in v["route"] for p in must_visit if p != start):
                all_routes.append(v)

    ranked = evaluate_and_rank(
        all_routes, weights, str(PAST_TRIPS_FILE), use_pareto=True, min_results=5
    )
    return ranked, weights


def main():
    G = load_graph()

    cases = [
        {
            "start": "Colombo National Museum",
            "must_visit": [
                "Colombo Lotus Tower",
                "Hikkaduwa Beach",
                "Coconut Tree Hill",
                "Aberdeen Falls",
                "Arugam Bay",
            ],
            "preferences": {"attraction": 10, "budget": 10, "time": 10, "popular": 10},
            "days": 5,
            "hours_per_day": 8.0,
        },
        {
            "start": "Colombo National Museum",
            "must_visit": [
                "Hikkaduwa Beach",
                "Coconut Tree Hill",
                "Colombo Lotus Tower",
            ],
            "preferences": {"attraction": 10, "budget": 10, "time": 10, "popular": 10},
            "days": 3,
            "hours_per_day": 8.0,
        },
    ]

    all_top1, all_gap, all_imp = [], [], []

    for case in cases:
        start = case["start"]
        must_visit = case["must_visit"]
        preferences = case["preferences"]
        days = case.get("days", 5)
        hours = case.get("hours_per_day", 8.0)
        max_t = days * hours * 60
        route_limit = max(8, len(set([start] + must_visit)))

        print("\n" + "=" * 70)
        print(f"Start: {start}")
        print(f"Must-visit: {must_visit}")
        print("=" * 70)

        ranked, weights = run_aco_pipeline(
            G, start, must_visit, preferences, max_t, route_limit
        )
        report = full_evaluation(
            G, start, must_visit, weights, ranked, str(PAST_TRIPS_FILE),
            max_total_time_min=max_t, max_exhaustive=7,
        )

        rows = report_to_rows(report)
        if rows:
            df = pd.DataFrame(rows)
            cols = [c for c in [
                "Method", "POI satisfaction", "POI cost", "Similarity",
                "Travel time (h)", "Distance (km)", "#POIs", "Preference score",
            ] if c in df.columns]
            print(df[cols].to_string(index=False))

        if report.get("improvement_vs_baseline_pct") is not None:
            print(f"ACO vs best baseline: {report['improvement_vs_baseline_pct']}%")
            all_imp.append(report["improvement_vs_baseline_pct"])

        ex = report.get("exhaustive") or {}
        if ex.get("route"):
            print(f"Top-1: {ex.get('Top-1')}  Top-3: {ex.get('Top-3 hit')}  "
                  f"Score gap: {ex.get('Score gap %')}%  ACO rank: {ex.get('ACO_rank')}")
            all_top1.append(ex.get("Top-1", 0))
            all_gap.append(ex.get("Score gap %", 0))

        fi = report.get("front_indicators") or {}
        if any(fi.get(k) is not None for k in ("GD", "IGD", "MS")):
            print(f"Front indicators: GD={fi.get('GD')}  IGD={fi.get('IGD')}  MS={fi.get('MS')}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if all_top1:
        print(f"Average Top-1 accuracy : {sum(all_top1)/len(all_top1)*100:.1f}%")
        print(f"Average score gap      : {sum(all_gap)/len(all_gap):.2f}%")
    if all_imp:
        print(f"Average ACO improvement vs baseline: {sum(all_imp)/len(all_imp):.2f}%")
    print("Metrics follow Saeki et al. style: satisfaction, cost, similarity, time.")
    print("Baselines: Nearest-Neighbour + Dijkstra edge weights + 2-opt.")


if __name__ == "__main__":
    main()
