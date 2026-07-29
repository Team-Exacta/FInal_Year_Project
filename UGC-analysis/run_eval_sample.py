"""Step 1 of evaluation: generate a sentence sample for manual labeling.

Usage:
  python run_eval_sample.py              # 300 sentences (default)
  python run_eval_sample.py --n 500      # 500 sentences
  python run_eval_sample.py --n 150 --seed 99
"""

import argparse
from evaluation.sampler import build_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=300, help="Total sentences to sample")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    build_sample(n_total=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
