"""
scripts/evaluate_ragas.py
==========================
Full RAGAS evaluation pipeline for the v2 Intent-Based RAG system.

Steps:
    1. Load test questions from data/evaluation/test_questions.json
    2. Run the v2 RAG pipeline for each question
    3. Collect retrieved contexts and generated answers
    4. Save raw results to outputs/ragas_results/raw_results.json
    5. Add ground-truth/reference answers (already in test_questions.json)
    6. Run RAGAS metrics (faithfulness, answer_relevancy,
       context_precision, context_recall)
    7. Save per-question scores + generate analysis report

Usage:
    cd "c:\\LLM RAG\\LLMRag-LangChain"
    python scripts/evaluate_ragas.py

    # Evaluate only first N questions (fast mode):
    python scripts/evaluate_ragas.py --limit 5

    # Skip pipeline run and reuse saved raw results:
    python scripts/evaluate_ragas.py --skip-pipeline

Notes:
    - Requires RAGAS 0.4.x (installed)
    - Uses the Gemini LLM configured in .env for RAGAS judging
    - Each pipeline run takes ~30-60 seconds per question
    - Full 15-question eval takes approximately 20-30 minutes
"""

import sys
import os
import io
import json
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

# Force UTF-8 output so Windows cp1252 doesn't crash on special chars
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Suppress noisy deprecation warnings ──────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
from src.core.logger import get_logger

logger = get_logger("evaluate_ragas")

# Output paths
OUTPUT_DIR = ROOT / "outputs" / "ragas_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_RESULTS_PATH  = OUTPUT_DIR / "raw_results.json"
SCORES_CSV_PATH   = OUTPUT_DIR / "scores.csv"
REPORT_PATH       = OUTPUT_DIR / "evaluation_report.txt"
TEST_QUESTIONS_PATH = ROOT / "data" / "evaluation" / "test_questions.json"

SEP = "=" * 70


# =============================================================================
# STEP 1 — Load test questions
# =============================================================================

def load_test_questions(limit=None):
    """Loads test questions from the JSON dataset file."""
    with open(TEST_QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]
    _p(f"\n[Step 1] Loaded {len(questions)} test questions from {TEST_QUESTIONS_PATH.name}")
    return questions


# =============================================================================
# STEP 2 & 3 — Run pipeline and collect contexts + answers
# =============================================================================

def run_pipeline_for_questions(questions):
    """
    Runs the v2 RAG pipeline for each question.
    Collects: question, answer, contexts (retrieved), ground_truth, intent, metadata.
    """
    _p(f"\n[Step 2] Running v2 RAG pipeline for {len(questions)} questions ...")
    _p("  This may take several minutes. Each question ~30-60 seconds.\n")

    # Import here so pipeline initialises only when needed
    import importlib.util, types
    spec = importlib.util.spec_from_file_location("run_rag_v2", ROOT / "scripts" / "run_rag_v2.py")
    rag_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_mod)
    process_query = rag_mod.process_query

    raw_results = []
    
    # ── CACHE SETUP ──
    cached_results = {}
    if RAW_RESULTS_PATH.exists():
        try:
            with open(RAW_RESULTS_PATH, encoding="utf-8") as f:
                cached_data = json.load(f)
                cached_results = {r["id"]: r for r in cached_data}
            _p(f"  [Cache] Found {len(cached_results)} cached pipeline results in {RAW_RESULTS_PATH.name}.")
        except Exception as e:
            _p(f"  [Cache] Could not load raw_results.json: {e}")

    for i, item in enumerate(questions, start=1):
        qid          = item["id"]
        question     = item["question"]
        intent       = item["intent"]
        ground_truth = item["ground_truth"]

        _p(f"  [{i:02d}/{len(questions)}] {qid} ({intent}): {question[:60]}...")
        
        if qid in cached_results:
            _p("      [OK] Using cached pipeline result.")
            raw_results.append(cached_results[qid])
            continue
            
        start = time.time()

        try:
            result = process_query(question)

            answer           = result.get("text", "").strip()
            query_intent     = result.get("query_intent", {})
            detected_intent  = query_intent.get("intent_type", "unknown")
            confidence       = query_intent.get("confidence", 0.0)
            structured_facts = result.get("structured_facts", [])

            research_data = result.get("research_data", "")
            if research_data:
                # Split the merged research_data into distinct chunks for Ragas to grade
                contexts = [chunk.strip() for chunk in research_data.split("\n\n") if chunk.strip()]
            else:
                contexts = ["No context retrieved due to pipeline error."]

            elapsed = time.time() - start
            _p(f"      [OK] Done in {elapsed:.1f}s | intent={detected_intent}"
               f" (conf={confidence:.2f}) | contexts={len(contexts)}")

            raw_results.append({
                "id":                    qid,
                "question":              question,
                "expected_intent":       intent,
                "detected_intent":       detected_intent,
                "intent_confidence":     confidence,
                "answer":                answer,
                "contexts":              contexts,
                "ground_truth":          ground_truth,
                "structured_facts_count": len(structured_facts),
                "elapsed_seconds":       round(elapsed, 2),
                "error":                 None,
            })

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Pipeline failed for {qid}: {e}")
            _p(f"      [FAIL] ERROR: {e}")

            raw_results.append({
                "id":                    qid,
                "question":              question,
                "expected_intent":       intent,
                "detected_intent":       "error",
                "intent_confidence":     0.0,
                "answer":                "",
                "contexts":              ["No context retrieved due to pipeline error."],
                "ground_truth":          ground_truth,
                "structured_facts_count": 0,
                "elapsed_seconds":       round(elapsed, 2),
                "error":                 str(e),
            })
            
        # ── INCREMENTAL SAVE ──
        try:
            with open(RAW_RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save raw_results incrementally: {e}")

    return raw_results


def _extract_contexts(structured_facts, answer, query_intent):
    """
    Converts structured graph facts + intent entities into RAGAS context strings.
    RAGAS context_precision / context_recall need a list of text chunks.
    """
    contexts = []

    for fact in structured_facts[:10]:
        parts = []
        place    = fact.get("place") or fact.get("p.name") or fact.get("name") or ""
        activity = fact.get("activity") or fact.get("a.name")
        feature  = fact.get("feature")  or fact.get("f.name")
        sentiment  = fact.get("sentiment") or fact.get("r.sentiment")
        pct        = fact.get("percentage") or fact.get("r.percentage")
        district   = fact.get("district") or fact.get("d.name")
        category   = fact.get("category") or fact.get("c.name")

        if place:     parts.append(f"Place: {place}")
        if activity:  parts.append(f"Activity: {activity}")
        if feature:   parts.append(f"Feature: {feature}")
        if sentiment: parts.append(f"Sentiment: {sentiment}")
        if pct is not None:
            parts.append(f"Evidence%: {pct:.1f}%" if isinstance(pct, float) else f"Evidence%: {pct}%")
        if district:  parts.append(f"District: {district}")
        if category:  parts.append(f"Category: {category}")

        # Nested activity/feature lists (comparison_query)
        acts = fact.get("activities")
        if isinstance(acts, list):
            names = [(a.get("activity") or a.get("name") or "") if isinstance(a, dict) else str(a) for a in acts]
            names = [n for n in names if n][:5]
            if names:
                parts.append(f"Activities: {', '.join(names)}")

        feats = fact.get("features")
        if isinstance(feats, list):
            names = [(f_.get("feature") or f_.get("name") or "") if isinstance(f_, dict) else str(f_) for f_ in feats]
            names = [n for n in names if n][:5]
            if names:
                parts.append(f"Features: {', '.join(names)}")

        if parts:
            contexts.append(" | ".join(parts))

    # Fallback if no graph facts
    if not contexts:
        entities   = query_intent.get("entities", {})
        places     = entities.get("places", [])
        activities = entities.get("activities", [])
        if places:     contexts.append(f"Places mentioned: {', '.join(places)}")
        if activities: contexts.append(f"Activities mentioned: {', '.join(activities)}")
        if not contexts:
            contexts.append("No structured context retrieved from knowledge graph.")

    return contexts


# =============================================================================
# STEP 4 — Save / load raw results
# =============================================================================

def save_raw_results(raw_results):
    with open(RAW_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    _p(f"\n[Step 4] Raw results saved -> {RAW_RESULTS_PATH}")


def load_raw_results():
    with open(RAW_RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# STEP 6 — Run RAGAS metrics
# =============================================================================

def run_ragas_evaluation(raw_results):
    """
    Runs four RAGAS metrics:
        faithfulness, answer_relevancy, context_precision, context_recall
    Returns a combined DataFrame with metadata + scores.
    """
    _p(f"\n[Step 6] Running RAGAS evaluation on {len(raw_results)} results ...")
    _p("  Using Gemini LLM as judge. This takes a few minutes ...\n")

    valid  = [r for r in raw_results if r.get("answer") and not r.get("error")]
    failed = [r for r in raw_results if r.get("error")]

    if failed:
        _p(f"  [WARN] Skipping {len(failed)} failed pipeline result(s) from RAGAS scoring.\n")
    if not valid:
        _p("  [FAIL] No valid results to evaluate.")
        return pd.DataFrame()

    # RAGAS 0.4.x imports
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from src.core.config import settings

    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_retries=3, # Survive rate limit bursts
        api_key=os.environ.get("OPENAI_API_KEY", "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")
    )
    ragas_llm = LangchainLLMWrapper(judge_llm)
    
    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        max_retries=3,
        api_key=os.environ.get("OPENAI_API_KEY", "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")
    )
    from ragas.embeddings import LangchainEmbeddingsWrapper
    ragas_embeddings = LangchainEmbeddingsWrapper(openai_embeddings)
    ragas_llm = LangchainLLMWrapper(judge_llm)

    # Run only Faithfulness as requested
    metrics = [
        Faithfulness(llm=ragas_llm)
    ]
    METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # ── CACHE SETUP ──
    cached_scores = {}
    if SCORES_CSV_PATH.exists():
        try:
            import math
            df = pd.read_csv(SCORES_CSV_PATH)
            for _, row in df.iterrows():
                if pd.notna(row.get("id")):
                    cached_scores[row["id"]] = row.to_dict()
            _p(f"  [Cache] Found {len(cached_scores)} cached RAGAS evaluations in {SCORES_CSV_PATH.name}.")
        except Exception as e:
            _p(f"  [Cache] Could not load scores.csv: {e}")

    individual_scores = []
    
    for idx, r in enumerate(valid):
        qid = r["id"]
        _p(f"    Evaluating Q{idx+1}/{len(valid)}: {qid} ...")
        
        if qid in cached_scores:
            _p("      [OK] Using cached RAGAS score.")
            cached_metrics = {m: cached_scores[qid].get(m) for m in METRIC_COLS}
            individual_scores.append(pd.DataFrame([cached_metrics]))
            continue
            
        try:
            sample = SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
            dataset = EvaluationDataset(samples=[sample])
            
            from ragas.run_config import RunConfig
            run_config = RunConfig(max_workers=1, max_wait=120)
            
            ragas_result = evaluate(
                dataset=dataset, 
                metrics=metrics, 
                llm=ragas_llm, 
                embeddings=ragas_embeddings, 
                raise_exceptions=True, 
                show_progress=False,
                run_config=run_config
            )
            q_scores_df = ragas_result.to_pandas()
            for col in METRIC_COLS:
                if col not in q_scores_df.columns:
                    q_scores_df[col] = None
            individual_scores.append(q_scores_df)
            
            # ── INCREMENTAL SAVE ──
            try:
                temp_scores = pd.concat(individual_scores, ignore_index=True)
                temp_meta = pd.DataFrame([{
                    "id": vr["id"], "question": vr["question"], "expected_intent": vr["expected_intent"],
                    "detected_intent": vr["detected_intent"], "intent_confidence": vr["intent_confidence"],
                    "structured_facts_count": vr["structured_facts_count"], "elapsed_seconds": vr["elapsed_seconds"]
                } for vr in valid[:len(individual_scores)]])
                temp_combined = pd.concat([temp_meta.reset_index(drop=True), temp_scores.reset_index(drop=True)], axis=1)
                temp_combined.to_csv(SCORES_CSV_PATH, index=False)
            except Exception as e:
                logger.error(f"Incremental CSV save failed: {e}")
            
            if idx < len(valid) - 1:
                _p(f"      Sleeping 20s to avoid Gemini API rate limits...")
                time.sleep(20)
                
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            logger.error(f"Ragas evaluation failed for {r['id']}: {e}\n{err_trace}")
            _p(f"      [FAIL] Ragas evaluation failed for {r['id']}: {type(e).__name__} - {e}")
            _p(f"      [TRACEBACK]: {err_trace}")
            empty_row = pd.DataFrame([{m: None for m in METRIC_COLS}])
            individual_scores.append(empty_row)

    if individual_scores:
        scores_df = pd.concat(individual_scores, ignore_index=True)
    else:
        scores_df = pd.DataFrame(columns=METRIC_COLS)

    # Add metadata
    meta_df = pd.DataFrame([{
        "id":                     r["id"],
        "question":               r["question"],
        "expected_intent":        r["expected_intent"],
        "detected_intent":        r["detected_intent"],
        "intent_confidence":      r["intent_confidence"],
        "structured_facts_count": r["structured_facts_count"],
        "elapsed_seconds":        r["elapsed_seconds"],
    } for r in valid])

    combined = pd.concat([meta_df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

    # Append failed rows
    if failed:
        failed_rows = pd.DataFrame([{
            "id": r["id"], "question": r["question"],
            "expected_intent": r["expected_intent"],
            "detected_intent": "error", "intent_confidence": 0.0,
            "structured_facts_count": 0, "elapsed_seconds": r["elapsed_seconds"],
            **{m: None for m in METRIC_COLS},
        } for r in failed])
        combined = pd.concat([combined, failed_rows], ignore_index=True)

    return combined


# =============================================================================
# STEP 7 — Analyse and report
# =============================================================================

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def analyse_and_report(scores_df, raw_results):
    lines = []

    def log(line=""):
        _p(line)
        lines.append(line)

    log(f"\n{'='*70}")
    log("  RAGAS EVALUATION REPORT -- Sri Lanka Tourism RAG v2")
    log("  System: Intent-Based Retrieval with Query Classification (v2)")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Questions evaluated: {len(raw_results)}")
    log('='*70)

    # -- Overall scores -------------------------------------------------------
    log("\n-- OVERALL METRIC SCORES -----------------------------------------------")
    log(f"  {'Metric':<22} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    log(f"  {'-'*54}")
    for metric in METRIC_NAMES:
        if metric in scores_df.columns:
            col = scores_df[metric].dropna()
            if len(col) > 0:
                log(f"  {metric:<22} {col.mean():>8.4f} {col.min():>8.4f} {col.max():>8.4f} {col.std():>8.4f}")
            else:
                log(f"  {metric:<22} {'N/A':>8}")

    # -- Intent detection accuracy --------------------------------------------
    log("\n-- INTENT DETECTION ACCURACY -------------------------------------------")
    correct = sum(1 for r in raw_results
                  if r.get("expected_intent") == r.get("detected_intent"))
    total = len([r for r in raw_results if not r.get("error")])
    if total > 0:
        log(f"  Correct intent detections: {correct}/{total} = {correct/total*100:.1f}%")
    else:
        log("  N/A")

    # -- Per-question breakdown -----------------------------------------------
    log("\n-- PER-QUESTION SCORES -------------------------------------------------")
    log(f"  {'ID':<5} {'Intent':<20} {'Faith':>7} {'AnsRel':>7} {'CtxPre':>7} {'CtxRec':>7} {'IMatch':>7}")
    log(f"  {'-'*62}")

    for _, row in scores_df.iterrows():
        qid    = str(row.get("id", "?"))
        intent = str(row.get("expected_intent", "?"))[:18]
        faith  = f"{row['faithfulness']:.3f}"    if pd.notna(row.get("faithfulness"))     else "  N/A"
        ar     = f"{row['answer_relevancy']:.3f}"if pd.notna(row.get("answer_relevancy")) else "  N/A"
        cp     = f"{row['context_precision']:.3f}"if pd.notna(row.get("context_precision"))else "  N/A"
        cr     = f"{row['context_recall']:.3f}"  if pd.notna(row.get("context_recall"))   else "  N/A"
        imatch = "YES" if row.get("expected_intent") == row.get("detected_intent") else "NO"
        log(f"  {qid:<5} {intent:<20} {faith:>7} {ar:>7} {cp:>7} {cr:>7} {imatch:>7}")

    # -- Per-intent breakdown -------------------------------------------------
    log("\n-- SCORES BY INTENT TYPE -----------------------------------------------")
    log(f"  {'Intent':<22} {'N':>3} {'Faith':>7} {'AnsRel':>7} {'CtxPre':>7} {'CtxRec':>7}")
    log(f"  {'-'*54}")
    for intent in scores_df["expected_intent"].dropna().unique():
        subset = scores_df[scores_df["expected_intent"] == intent]
        n = len(subset)
        def mfmt(col):
            v = subset[col].mean() if col in subset else float("nan")
            return f"{v:.3f}" if pd.notna(v) else "  N/A"
        log(f"  {intent:<22} {n:>3} {mfmt('faithfulness'):>7} {mfmt('answer_relevancy'):>7}"
            f" {mfmt('context_precision'):>7} {mfmt('context_recall'):>7}")

    # -- Retrieval stats ------------------------------------------------------
    log("\n-- RETRIEVAL STATISTICS ------------------------------------------------")
    if "structured_facts_count" in scores_df.columns:
        log(f"  Avg graph facts / question : {scores_df['structured_facts_count'].mean():.1f}")
        log(f"  Min graph facts            : {scores_df['structured_facts_count'].min()}")
        log(f"  Max graph facts            : {scores_df['structured_facts_count'].max()}")
    if "elapsed_seconds" in scores_df.columns:
        log(f"  Avg pipeline time / question: {scores_df['elapsed_seconds'].mean():.1f}s")

    # -- Guide ----------------------------------------------------------------
    log("\n-- METRIC INTERPRETATION -----------------------------------------------")
    log("  Faithfulness     : Answer is grounded in retrieved context (anti-hallucination).")
    log("  Answer Relevancy : Answer directly addresses the user question.")
    log("  Context Precision: Retrieved context is focused on what was asked.")
    log("  Context Recall   : Context covers the expected ground-truth answer.")
    log("\n  Score range: 0.0 (poor) --> 1.0 (perfect)")
    log("  Thresholds : Acceptable > 0.60 | Good > 0.70 | Excellent > 0.85")

    # -- Save -----------------------------------------------------------------
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log(f"\n[Step 7] Report saved  -> {REPORT_PATH}")
    log(f"         Scores CSV    -> {SCORES_CSV_PATH}")
    log(f"         Raw results   -> {RAW_RESULTS_PATH}")
    log(f"\n{'='*70}\n")


# =============================================================================
# Utility
# =============================================================================

def _p(text=""):
    """Print with flush; safe for redirected streams."""
    print(text, flush=True)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for Sri Lanka RAG v2")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N questions (quick test)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Load cached raw_results.json instead of re-running the pipeline")
    args = parser.parse_args()

    _p(SEP)
    _p("  Sri Lanka Tourism RAG v2 -- RAGAS Evaluation Pipeline")
    _p(SEP)

    # Step 1
    questions = load_test_questions(limit=args.limit)

    # Steps 2-4
    if args.skip_pipeline:
        if not RAW_RESULTS_PATH.exists():
            _p(f"[FAIL] No saved raw_results.json found at {RAW_RESULTS_PATH}.")
            _p("       Run without --skip-pipeline first.")
            sys.exit(1)
        raw_results = load_raw_results()
        if args.limit:
            raw_results = raw_results[:args.limit]
        _p(f"\n[Step 2-3] Loaded {len(raw_results)} cached results from {RAW_RESULTS_PATH.name}")
    else:
        raw_results = run_pipeline_for_questions(questions)
        save_raw_results(raw_results)

    # Step 6
    scores_df = run_ragas_evaluation(raw_results)
    if scores_df.empty:
        _p("\n[FAIL] Evaluation produced no scores. Check errors above.")
        sys.exit(1)

    scores_df.to_csv(SCORES_CSV_PATH, index=False)
    _p(f"\n[Step 6 complete] Scores saved -> {SCORES_CSV_PATH}")

    # Step 7
    analyse_and_report(scores_df, raw_results)


if __name__ == "__main__":
    main()
