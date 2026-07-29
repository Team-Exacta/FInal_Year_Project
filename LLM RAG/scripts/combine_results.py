"""
scripts/combine_results.py
==========================
Reads individual setup result folders and combines into:
    - FINAL_ablation_comparison.csv   (all setups × all questions)
    - FINAL_ablation_summary.csv      (mean scores per setup — for thesis table)
    - FINAL_ablation_report.txt       (human-readable comparison report)

Run standalone after all setups complete:
    cd "c:\\LLM RAG\\update promts"
    python scripts/combine_results.py
"""

import os
import sys
import io
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs" / "ragas_results"

SETUP_FOLDERS = [
    ("1_Baseline",                 "Vector Only"),
    ("2_Plus_BM25",               "Vector + BM25"),
    ("3_Plus_Graph",              "Vector + BM25 + Graph"),
    ("4_Plus_Dynamic_Thresholds", "Vector + BM25 + Graph + Dynamic"),
    ("5_Full_Architecture",       "All Features ON"),
]

METRIC_COLS = ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]

FINAL_CSV    = OUTPUT_DIR / "FINAL_ablation_comparison.csv"
SUMMARY_CSV  = OUTPUT_DIR / "FINAL_ablation_summary.csv"
REPORT_TXT   = OUTPUT_DIR / "FINAL_ablation_report.txt"

import pandas as pd

def load_setup(folder_name, label):
    """Load scores.csv from a setup folder."""
    csv_path = OUTPUT_DIR / folder_name / "scores.csv"
    if not csv_path.exists():
        print(f"  [WARN] {folder_name}/scores.csv not found — skipping")
        return None
    df = pd.read_csv(csv_path)
    df.insert(0, "setup_label", label)
    df.insert(0, "setup_name",  folder_name)
    print(f"  [OK] Loaded {folder_name}/scores.csv — {len(df)} rows")
    return df


def compute_intent_accuracy(df):
    """Fraction of questions where detected_intent == expected_intent."""
    valid = df.dropna(subset=["expected_intent", "detected_intent"])
    if len(valid) == 0:
        return float("nan")
    return (valid["expected_intent"] == valid["detected_intent"]).mean()


def main():
    print("=" * 70)
    print("  COMBINING ALL SETUP RESULTS")
    print("=" * 70)

    frames = []
    for folder, label in SETUP_FOLDERS:
        df = load_setup(folder, label)
        if df is not None:
            frames.append(df)

    if not frames:
        print("\n[FAIL] No results found. Run the setup scripts first.")
        sys.exit(1)

    # ── Combined full CSV ──────────────────────────────────────────────────────
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(FINAL_CSV, index=False)
    print(f"\n[Saved] Full comparison -> {FINAL_CSV}")

    # ── Summary CSV (one row per setup) ───────────────────────────────────────
    summary_rows = []
    for df in frames:
        setup  = df["setup_name"].iloc[0]
        label  = df["setup_label"].iloc[0]
        row    = {"setup_name": setup, "setup_label": label}

        for m in METRIC_COLS:
            if m in df.columns:
                col = df[m].dropna()
                row[f"{m}_mean"] = col.mean() if len(col) else float("nan")
                row[f"{m}_min"]  = col.min()  if len(col) else float("nan")
                row[f"{m}_max"]  = col.max()  if len(col) else float("nan")
            else:
                row[f"{m}_mean"] = float("nan")

        row["intent_accuracy"] = compute_intent_accuracy(df)
        total = len(df)
        answered = df["faithfulness"].notna().sum() if "faithfulness" in df.columns else 0
        row["questions_answered"] = answered
        row["questions_total"]    = total
        row["avg_elapsed_sec"]    = df["elapsed_seconds"].mean() if "elapsed_seconds" in df.columns else float("nan")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[Saved] Summary table    -> {SUMMARY_CSV}")

    # ── Human-readable report ─────────────────────────────────────────────────
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log()
    log("=" * 70)
    log("  ABLATION STUDY — FINAL COMPARISON REPORT")
    log("  Sri Lanka Tourism RAG v2 — Intent-Based Retrieval")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Setups combined: {len(frames)}")
    log("=" * 70)

    log("\n-- SETUP COMPARISON TABLE ----------------------------------------------")
    log(f"  {'Setup':<32} {'Faith':>7} {'CtxRec':>7} {'IntAcc':>7} {'Qs':>4}")
    log(f"  {'-'*60}")
    for row in summary_rows:
        faith  = f"{row['faithfulness_mean']:.4f}" if pd.notna(row.get('faithfulness_mean')) else "  N/A"
        recall = f"{row['context_recall_mean']:.4f}" if pd.notna(row.get('context_recall_mean')) else "  N/A"
        iacc   = f"{row['intent_accuracy']*100:.1f}%" if pd.notna(row.get('intent_accuracy')) else "  N/A"
        qs     = f"{row['questions_answered']}/{row['questions_total']}"
        log(f"  {row['setup_name']:<32} {faith:>7} {recall:>7} {iacc:>7} {qs:>6}")

    log("\n-- BEST SCORES ---------------------------------------------------------")
    best_faith  = max(summary_rows, key=lambda r: r.get("faithfulness_mean") or 0)
    best_recall = max(summary_rows, key=lambda r: r.get("context_recall_mean") or 0)
    log(f"  Best Faithfulness  : {best_faith['setup_name']} ({best_faith['faithfulness_mean']:.4f})")
    log(f"  Best Context Recall: {best_recall['setup_name']} ({best_recall['context_recall_mean']:.4f})")

    log("\n-- IMPROVEMENT OVER BASELINE -------------------------------------------")
    baseline = next((r for r in summary_rows if r["setup_name"] == "1_Baseline"), None)
    if baseline:
        for row in summary_rows[1:]:
            f_diff = (row.get("faithfulness_mean", 0) or 0) - (baseline.get("faithfulness_mean", 0) or 0)
            r_diff = (row.get("context_recall_mean", 0) or 0) - (baseline.get("context_recall_mean", 0) or 0)
            log(f"  {row['setup_name']:<32} Faith Δ={f_diff:+.4f}  Recall Δ={r_diff:+.4f}")

    log("\n-- PER-QUESTION BREAKDOWN (FAITHFULNESS) -------------------------------")
    for qid in combined["id"].dropna().unique():
        q_rows = combined[combined["id"] == qid]
        log(f"\n  Question: {qid}")
        for _, r in q_rows.iterrows():
            f = f"{r['faithfulness']:.3f}" if pd.notna(r.get("faithfulness")) else "  N/A"
            c = f"{r['context_recall']:.3f}" if pd.notna(r.get("context_recall")) else "  N/A"
            log(f"    {str(r.get('setup_name','')):<32} Faith={f}  Recall={c}")

    log("\n-- METRIC INTERPRETATION -----------------------------------------------")
    log("  Faithfulness    : Answer is grounded in retrieved context (no hallucination).")
    log("  Context Recall  : Retrieved context covers the expected ground-truth answer.")
    log("  Intent Accuracy : % of questions where intent was correctly classified.")
    log("\n  Score range: 0.0 (poor) --> 1.0 (perfect)")
    log("  Thresholds : Acceptable > 0.60 | Good > 0.70 | Excellent > 0.85")

    log("\n-- OUTPUT FILES --------------------------------------------------------")
    log(f"  Full CSV    : {FINAL_CSV}")
    log(f"  Summary CSV : {SUMMARY_CSV}")
    log(f"  Report TXT  : {REPORT_TXT}")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Saved] Final report     -> {REPORT_TXT}")
    print()
    print("=" * 70)
    print("  DONE! All results combined successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
