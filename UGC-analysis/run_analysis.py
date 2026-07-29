"""Entry point for the review analysis tagging pipeline.

Detects best_time / crowd_level / cost_level signals in cleaned reviews
using zero-shot NLI classification.

Usage:
  python run_analysis.py                          # process all places (resume by default)
  python run_analysis.py --no-resume             # reprocess everything from scratch
  python run_analysis.py --place Aberdeen_Falls  # single place (debug/test)
  python run_analysis.py --threshold 0.6         # override confidence threshold
"""

import argparse

from analysis.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Tag travel reviews with best_time / crowd_level / cost_level"
    )
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore previous progress and reprocess all places")
    parser.add_argument("--place", type=str, default=None,
                        help="Process only this place key (filename stem, for debugging)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override ANALYSIS_SCORE_THRESHOLD from settings")
    args = parser.parse_args()

    run_pipeline(
        resume=not args.no_resume,
        places_filter=[args.place] if args.place else None,
        threshold_override=args.threshold,
    )


if __name__ == "__main__":
    main()
