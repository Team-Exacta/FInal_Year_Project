# How to measure accuracy

This project includes `src/check_accuracy.py`.

## What it does

For small itineraries (few must-visit places) it:

1. Generates **every possible order** of the places
2. Scores each order with the **same ranking formula** as the app
3. Treats the highest score as the **true best**
4. Runs the **ACO pipeline** (same logic as the Streamlit app)
5. Compares ACO vs true best and reports:
   - **Top-1 accuracy** – ACO best score equals true best
   - **Top-3 hit rate** – true best appears in ACO’s top 3
   - **Score gap** – how much worse ACO is than the true optimum (0% = perfect)

## Run

From the project root:

```bash
pip install -r requirements.txt
python src/check_accuracy.py
```

or:

```bash
python -m src.check_accuracy
```

## Example output

```
True best score : 0.9611
ACO best score  : 0.9611
Top-1 (same score as true best): 1
Score gap                      : 0.00%

SUMMARY
Average Top-1 accuracy   : 100.0%
Average Top-3 hit rate   : 100.0%
Average score gap        : 0.00%
```

## Notes for report / viva

- This method only works for **small** sets (about ≤ 6 intermediate places), because all orders are checked.
- For larger problems, use relative quality (scores + ranking) instead of exhaustive accuracy.
- The script uses the same `run_aco`, `evaluate_and_rank`, and `_score_routes` functions as the UI, so the comparison is fair.

## Change test cases

Edit the `cases` list at the bottom of `src/check_accuracy.py`.
Use exact POI names from `data/pois.csv`.
