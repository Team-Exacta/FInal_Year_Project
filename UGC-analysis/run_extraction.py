"""CLI entry point for Step 4+5: Span Extraction + Normalization.

Usage:
  python run_extraction.py                     # process all places (resume-safe)
  python run_extraction.py --no-resume         # restart from scratch
  python run_extraction.py --place Sigiriya_Lion_Rock
"""

import argparse
import sys

from extraction.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Span extraction + normalization pipeline")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore previous progress and reprocess everything")
    parser.add_argument("--place", metavar="PLACE_KEY",
                        help="Process only this one place (by file key, e.g. Sigiriya_Lion_Rock)")
    args = parser.parse_args()

    places = [args.place] if args.place else None

    run_pipeline(
        resume=not args.no_resume,
        places_filter=places,
        verbose=True,
    )


if __name__ == "__main__":
    main()
