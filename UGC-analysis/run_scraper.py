"""Entry point for TripAdvisor review scraping via Apify.

Usage:
  python run_scraper.py                 # scrape all pending places
  python run_scraper.py --limit 5       # only scrape the first N pending places
  python run_scraper.py --input FILE    # use a custom places list (JSON list of {name,url,id})
"""

import argparse

from scraper.apify_collector import collect_with_apify


def main():
    parser = argparse.ArgumentParser(description="Scrape TripAdvisor reviews via Apify")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only scrape the first N pending places this run")
    parser.add_argument("--input", type=str, default=None,
                        help="Custom places JSON file (defaults to config/places_to_scrape.json)")
    args = parser.parse_args()

    collect_with_apify(limit=args.limit, places_file=args.input)


if __name__ == "__main__":
    main()
