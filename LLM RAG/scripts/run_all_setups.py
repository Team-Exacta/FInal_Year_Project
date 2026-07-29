"""
scripts/run_all_setups.py
=========================
Master runner — runs all 5 ablation setups ONE BY ONE.
Each setup saves its own output to a separate folder:

    outputs/ragas_results/
    ├── 1_Baseline/
    │   ├── raw_results.json
    │   ├── scores.csv
    │   └── evaluation_report.txt
    ├── 2_Plus_BM25/       ...
    ├── 3_Plus_Graph/      ...
    ├── 4_Plus_Dynamic_Thresholds/  ...
    ├── 5_Full_Architecture/        ...
    └── FINAL_ablation_comparison.csv  ← combined at the end

Usage:
    cd "c:\\LLM RAG\\update promts"
    python scripts/run_all_setups.py

    # Run only first N questions (fast test):
    python scripts/run_all_setups.py --limit 3
"""

import sys
import os
import time
import subprocess
import argparse
from pathlib import Path

ROOT       = Path(__file__).parent.parent
SCRIPTS    = ROOT / "scripts"
OUTPUT_DIR = ROOT / "outputs" / "ragas_results"
COMBINE    = SCRIPTS / "combine_results.py"

SETUPS = [
    "setup1_baseline.py",
    "setup2_bm25.py",
    "setup3_graph.py",
    "setup4_dynamic.py",
    "setup5_full.py",
]

SEP = "=" * 70

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate first N questions (quick test)")
    parser.add_argument("--skip", type=str, default=None,
                        help="Comma-separated setup numbers to skip e.g. 1,2")
    args = parser.parse_args()

    skip_set = set()
    if args.skip:
        skip_set = {f"setup{n.strip()}" for n in args.skip.split(",")}

    print(SEP)
    print("  ABLATION STUDY — ALL 5 SETUPS")
    print(f"  Output folder: {OUTPUT_DIR}")
    if args.limit:
        print(f"  Running only first {args.limit} questions per setup (fast mode)")
    print(SEP)

    results = {}
    total_start = time.time()

    for i, script_name in enumerate(SETUPS, 1):
        # Check skip
        skip_key = f"setup{i}"
        if skip_key in skip_set:
            print(f"\n[SKIP] Setup {i} skipped by user request")
            continue

        print(f"\n{SEP}")
        print(f"  RUNNING SETUP {i}/5: {script_name}")
        print(SEP)

        cmd = [sys.executable, str(SCRIPTS / script_name)]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

        start = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"\n  ✅ Setup {i} COMPLETED in {elapsed:.0f}s")
            results[script_name] = "OK"
        else:
            print(f"\n  ❌ Setup {i} FAILED (exit code {result.returncode})")
            results[script_name] = "FAILED"

    # Final summary
    print(f"\n{SEP}")
    print("  ALL SETUPS DONE — SUMMARY")
    print(SEP)
    for name, status in results.items():
        icon = "✅" if status == "OK" else "❌"
        print(f"  {icon}  {name}: {status}")
    print(f"\n  Total time: {(time.time() - total_start)/60:.1f} minutes")

    # Run combine script to merge all results
    ok_count = sum(1 for s in results.values() if s == "OK")
    if ok_count >= 1:
        print(f"\n{SEP}")
        print("  COMBINING ALL RESULTS...")
        print(SEP)
        subprocess.run([sys.executable, str(COMBINE)], cwd=str(ROOT))
    else:
        print("\n[WARN] No successful setups to combine.")

if __name__ == "__main__":
    main()
