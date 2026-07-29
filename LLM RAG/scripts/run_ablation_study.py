import os
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs" / "ragas_results"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_ragas.py"
SCORES_PATH = OUTPUT_DIR / "scores.csv"
MASTER_CSV = OUTPUT_DIR / "ablation_study_results.csv"

# Define the 5 incremental setups
setups = [
    {
        "name": "1_Baseline",
        "env": {
            "USE_BM25": "false",
            "USE_GRAPH_RETRIEVAL": "false",
            "USE_DYNAMIC_THRESHOLDS": "false",
            "USE_FACT_CHECKER": "false",
            "USE_INTENT_CLASSIFIER": "false"
        }
    },
    {
        "name": "2_Plus_BM25",
        "env": {
            "USE_BM25": "true",
            "USE_GRAPH_RETRIEVAL": "false",
            "USE_DYNAMIC_THRESHOLDS": "false",
            "USE_FACT_CHECKER": "false",
            "USE_INTENT_CLASSIFIER": "false"
        }
    },
    {
        "name": "3_Plus_Graph",
        "env": {
            "USE_BM25": "true",
            "USE_GRAPH_RETRIEVAL": "true",
            "USE_DYNAMIC_THRESHOLDS": "false",
            "USE_FACT_CHECKER": "false",
            "USE_INTENT_CLASSIFIER": "false"
        }
    },
    {
        "name": "4_Plus_Dynamic_Thresholds",
        "env": {
            "USE_BM25": "true",
            "USE_GRAPH_RETRIEVAL": "true",
            "USE_DYNAMIC_THRESHOLDS": "true",
            "USE_FACT_CHECKER": "false",
            "USE_INTENT_CLASSIFIER": "false"
        }
    },
    {
        "name": "5_Full_Architecture",
        "env": {
            "USE_BM25": "true",
            "USE_GRAPH_RETRIEVAL": "true",
            "USE_DYNAMIC_THRESHOLDS": "true",
            "USE_FACT_CHECKER": "true",
            "USE_INTENT_CLASSIFIER": "true"
        }
    }
]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to test")
    args = parser.parse_args()

    all_scores = []

    for setup in setups:
        print(f"\n{'='*70}")
        print(f"RUNNING ABLATION SETUP: {setup['name']}")
        print(f"{'='*70}")
        
        # Prepare environment with overrides
        env = os.environ.copy()
        for k, v in setup["env"].items():
            env[k] = v
            print(f"  {k}={v}")
            
        # Command to run evaluate_ragas.py
        cmd = [sys.executable, str(EVAL_SCRIPT)]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
            
        print("\nStarting evaluation subprocess...")
        start_time = time.time()
        
        # Execute subprocess
        result = subprocess.run(cmd, env=env)
        
        if result.returncode != 0:
            print(f"\n[ERROR] Setup {setup['name']} failed! See logs above.")
            continue
            
        # Read the generated scores.csv from evaluate_ragas.py
        if SCORES_PATH.exists():
            df = pd.read_csv(SCORES_PATH)
            # Inject Setup name column at the front
            df.insert(0, "Setup", setup["name"])
            all_scores.append(df)
            print(f"\n[OK] Setup {setup['name']} completed in {time.time() - start_time:.1f}s")
            
            # Progressively save to master CSV so results are live
            master_df = pd.concat(all_scores, ignore_index=True)
            master_df.to_csv(MASTER_CSV, index=False)
        else:
            print(f"\n[ERROR] Output scores.csv not found for setup {setup['name']}")

    # Combine and save to master CSV
    if all_scores:
        master_df = pd.concat(all_scores, ignore_index=True)
        master_df.to_csv(MASTER_CSV, index=False)
        print(f"\n{'='*70}")
        print(f"ABLATION STUDY COMPLETE!")
        print(f"Master results saved to: {MASTER_CSV}")
        
        # Print summary of mean scores
        print("\nSummary of Mean Scores by Setup:")
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        summary = master_df.groupby("Setup")[metrics].mean()
        print(summary.to_string())
    else:
        print("\n[FAIL] No successful runs to combine.")

if __name__ == "__main__":
    main()
