"""Score the normalization step against the value gold layer.

Reads output/evaluation/norm_gold_sample.csv (after it has been value-labeled via
labeling_ui_norm.py) and prints per-field accuracy / macro-F1 / confusion matrices,
crowd ordinal MAE + quadratic-weighted kappa, and coverage. Saves
output/evaluation/norm_eval_results.json.

Usage:
  python run_eval_norm_score.py
"""

from evaluation.norm_scorer import score

if __name__ == "__main__":
    score()
