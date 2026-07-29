"""
scripts/setup4_dynamic.py
=========================
ABLATION SETUP 4 - Baseline + BM25 + Graph + Dynamic Thresholds
Features: Vector search + BM25 + Neo4j graph + dynamic score weighting
          (no Intent Classifier, no Fact Checker)

Run:
    cd "c:/LLM RAG/update promts"
    python scripts/setup4_dynamic.py
"""

import os, sys, json, time, io, warnings
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load .env so OPENAI_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

# ── Feature flags for THIS setup ──────────────────────────────────────────────
os.environ["USE_BM25"]                = "true"
os.environ["USE_GRAPH_RETRIEVAL"]     = "true"
os.environ["USE_DYNAMIC_THRESHOLDS"]  = "true"
os.environ["USE_FACT_CHECKER"]        = "false"
os.environ["USE_INTENT_CLASSIFIER"]   = "false"

SETUP_NAME = "4_Plus_Dynamic_Thresholds"

ROOT       = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

OUTPUT_DIR = ROOT / "outputs" / "ragas_results" / SETUP_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_JSON   = OUTPUT_DIR / "raw_results.json"
SCORES_CSV = OUTPUT_DIR / "scores.csv"
REPORT_TXT = OUTPUT_DIR / "evaluation_report.txt"
TEST_Q     = ROOT / "data" / "evaluation" / "test_questions.json"

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

print("=" * 70)
print(f"  ABLATION SETUP: {SETUP_NAME}")
print(f"  USE_BM25=true | USE_GRAPH_RETRIEVAL=true")
print(f"  USE_DYNAMIC_THRESHOLDS=true | USE_INTENT_CLASSIFIER=false")
print("=" * 70)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

with open(TEST_Q, encoding="utf-8") as f:
    questions = json.load(f)
if args.limit:
    questions = questions[:args.limit]
print(f"\n[Step 1] Loaded {len(questions)} questions")

import importlib.util
spec = importlib.util.spec_from_file_location("run_rag_v2", ROOT / "scripts" / "run_rag_v2.py")
rag_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_mod)
process_query = rag_mod.process_query

raw_results = []
print(f"\n[Step 2] Running pipeline for {len(questions)} questions...\n")

# ── CACHE SETUP ──
cached_results = {}
if RAW_JSON.exists():
    try:
        with open(RAW_JSON, encoding="utf-8") as f:
            cached_data = json.load(f)
            cached_results = {r["id"]: r for r in cached_data}
        print(f"  [Cache] Found {len(cached_results)} cached pipeline results.")
    except Exception as e:
        print(f"  [Cache] Could not load raw_results.json: {e}")

for i, item in enumerate(questions, 1):
    qid, question, intent, ground_truth = item["id"], item["question"], item["intent"], item["ground_truth"]
    
    if qid in cached_results:
        print(f"  [{i:02d}/{len(questions)}] {qid}: {question[:55]}...")
        print("      [OK] Using cached pipeline result.")
        raw_results.append(cached_results[qid])
        continue
    print(f"  [{i:02d}/{len(questions)}] {qid}: {question[:55]}...")
    start = time.time()
    try:
        result          = process_query(question)
        answer          = result.get("text", "").strip()
        query_intent    = result.get("query_intent", {})
        detected_intent = query_intent.get("intent_type", "unknown")
        confidence      = query_intent.get("confidence", 0.0)
        research_data   = result.get("research_data", "")
        raw_blocks = research_data.split("\n\n") if research_data else []
        contexts = []
        for block in raw_blocks:
            block = block.strip()
            if not block: continue
            if block.startswith("QUERY INTENT:") or block.startswith("RETRIEVAL QUALITY") or block.startswith("VERIFIED ROAD DISTANCE FACTS") or block.startswith("RETRIEVAL NOTES:"):
                continue
            if block.startswith("GRAPH FACTS"):
                lines = block.split("\n")
                for line in lines[1:]:
                    if line.strip():
                        contexts.append(f"Knowledge Graph Fact {line.strip()}")
            else:
                contexts.append(block)
        if not contexts:
            contexts = ["No context retrieved."]
        elapsed         = round(time.time() - start, 2)
        print(f"      [OK] {elapsed}s | intent={detected_intent} | contexts={len(contexts)}")
        raw_results.append({"id": qid, "question": question, "expected_intent": intent,
                             "detected_intent": detected_intent, "intent_confidence": confidence,
                             "answer": answer, "contexts": contexts, "ground_truth": ground_truth,
                             "elapsed_seconds": elapsed, "error": None})
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"      [FAIL] {e}")
        raw_results.append({"id": qid, "question": question, "expected_intent": intent,
                             "detected_intent": "error", "intent_confidence": 0.0,
                             "answer": "", "contexts": ["Pipeline error."], "ground_truth": ground_truth,
                             "elapsed_seconds": elapsed, "error": str(e)})
                             
    # ── INCREMENTAL SAVE ──
    try:
        with open(RAW_JSON, "w", encoding="utf-8") as fj:
            json.dump(raw_results, fj, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save raw_results incrementally: {e}")

with open(RAW_JSON, "w", encoding="utf-8") as f:
    json.dump(raw_results, f, indent=2, ensure_ascii=False)
print(f"\n[Step 3] Raw results saved -> {RAW_JSON}")

import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, ContextRecall, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

from ragas.run_config import RunConfig

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3, api_key=OPENAI_KEY)
ragas_llm = LangchainLLMWrapper(judge_llm)
ragas_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small", max_retries=3, api_key=OPENAI_KEY))
metrics   = [Faithfulness(llm=ragas_llm)]
run_config = RunConfig(max_workers=1, max_wait=120)

valid = [r for r in raw_results if r.get("answer") and not r.get("error")]
print(f"\n[Step 4] RAGAS scoring {len(valid)} valid results...")

# ── CACHE SETUP ──
cached_scores = {}
if SCORES_CSV.exists():
    try:
        df = pd.read_csv(SCORES_CSV)
        for _, row in df.iterrows():
            if pd.notna(row.get("id")):
                cached_scores[row["id"]] = row.to_dict()
        print(f"  [Cache] Found {len(cached_scores)} cached RAGAS evaluations.")
    except Exception as e:
        print(f"  [Cache] Could not load scores.csv: {e}")

individual = []
for idx, r in enumerate(valid):
    print(f"  Scoring {r['id']} ({idx+1}/{len(valid)})...")
    
    if r["id"] in cached_scores:
        print("      [OK] Using cached RAGAS score.")
        cached_metrics = {m: cached_scores[r["id"]].get(m) for m in METRIC_COLS}
        individual.append(pd.DataFrame([cached_metrics]))
        continue
    try:
        sample  = SingleTurnSample(user_input=r["question"], response=r["answer"],
                                   retrieved_contexts=r["contexts"], reference=r["ground_truth"])
        dataset = EvaluationDataset(samples=[sample])
        res     = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm,
                           embeddings=ragas_emb, raise_exceptions=True, show_progress=False, run_config=run_config)
        df_res  = res.to_pandas()
        for col in METRIC_COLS:
            if col not in df_res.columns:
                df_res[col] = None
        individual.append(df_res)
        
        # ── INCREMENTAL SAVE ──
        try:
            temp_scores = pd.concat(individual, ignore_index=True)
            temp_meta = pd.DataFrame([{
                "id": vr["id"], "question": vr["question"], "expected_intent": vr["expected_intent"],
                "detected_intent": vr["detected_intent"], "intent_confidence": vr["intent_confidence"],
                "elapsed_seconds": vr.get("elapsed_seconds", 0.0)
            } for vr in valid[:len(individual)]])
            temp_combined = pd.concat([temp_meta.reset_index(drop=True), temp_scores.reset_index(drop=True)], axis=1)
            temp_combined.to_csv(SCORES_CSV, index=False)
        except Exception as e:
            print(f"Incremental CSV save failed: {e}")
            
        if idx < len(valid) - 1:
            print("    Sleeping 3s...")
            time.sleep(3)
    except Exception as e:
        print(f"    [FAIL] {e}")
        individual.append(pd.DataFrame([{m: None for m in METRIC_COLS}]))

scores_df = pd.concat(individual, ignore_index=True) if individual else pd.DataFrame(columns=METRIC_COLS)
meta_df   = pd.DataFrame([{"id": r["id"], "question": r["question"],
                            "expected_intent": r["expected_intent"], "detected_intent": r["detected_intent"],
                            "intent_confidence": r["intent_confidence"], "elapsed_seconds": r["elapsed_seconds"]}
                           for r in valid])
combined  = pd.concat([meta_df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

valid_map = {r["id"]: r for r in valid}
combined["user_input"]         = combined["id"].map(lambda x: valid_map[x]["question"] if x in valid_map else "")
combined["retrieved_contexts"] = combined["id"].map(lambda x: str(valid_map[x]["contexts"]) if x in valid_map else "")
combined["response"]           = combined["id"].map(lambda x: valid_map[x]["answer"] if x in valid_map else "")
combined["reference"]          = combined["id"].map(lambda x: valid_map[x]["ground_truth"] if x in valid_map else "")
combined.to_csv(SCORES_CSV, index=False)
print(f"\n[Step 5] Scores saved -> {SCORES_CSV}")

lines = []
def log(s=""):
    print(s); lines.append(s)

log(f"\n{'='*70}")
log(f"  EVALUATION REPORT — Setup: {SETUP_NAME}")
log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Questions evaluated: {len(raw_results)}")
log("=" * 70)
log("\n  FEATURE FLAGS:")
log("  USE_BM25=true | USE_GRAPH_RETRIEVAL=true")
log("  USE_DYNAMIC_THRESHOLDS=true | USE_FACT_CHECKER=false | USE_INTENT_CLASSIFIER=false")
log("\n  OVERALL METRIC SCORES:")
log(f"  {'Metric':<22} {'Mean':>8} {'Min':>8} {'Max':>8}")
log(f"  {'-'*50}")
for m in METRIC_COLS:
    if m in combined.columns:
        col = combined[m].dropna()
        if len(col):
            log(f"  {m:<22} {col.mean():>8.4f} {col.min():>8.4f} {col.max():>8.4f}")
        else:
            log(f"  {m:<22} {'N/A':>8}")

correct = sum(1 for r in raw_results if r.get("expected_intent") == r.get("detected_intent") and not r.get("error"))
total   = len([r for r in raw_results if not r.get("error")])
log(f"\n  INTENT ACCURACY: {correct}/{total} = {correct/total*100:.1f}%" if total else "\n  INTENT ACCURACY: N/A")
log(f"\n  Avg pipeline time: {sum(r['elapsed_seconds'] for r in raw_results)/len(raw_results):.1f}s")

with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[Done] Report -> {REPORT_TXT}")
print(f"       CSV    -> {SCORES_CSV}")
print(f"       JSON   -> {RAW_JSON}")
