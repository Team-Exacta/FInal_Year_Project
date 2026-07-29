"""Step 2 of evaluation: score the human-labeled CSV.

Usage:
  python run_eval_score.py
  python run_eval_score.py --csv output/evaluation/gold_label_sample.csv
  python run_eval_score.py --no-errors      # suppress error examples
"""

import argparse
import os
from evaluation.scorer import score

DEFAULT_CSV = os.path.join("output", "evaluation", "gold_label_sample.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default=DEFAULT_CSV)
    parser.add_argument("--no-errors", action="store_true",
                        help="Skip printing false positive / false negative examples")
    args = parser.parse_args()

    score(args.csv, print_errors=not args.no_errors)


if __name__ == "__main__":
    main()
