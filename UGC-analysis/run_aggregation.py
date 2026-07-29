"""CLI entry point for Step 6: POI Aggregation.

Usage:
  python run_aggregation.py
"""

from aggregation.aggregator import run_aggregation

if __name__ == "__main__":
    run_aggregation(verbose=True)
