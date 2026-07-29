# Evaluation & accuracy

## Metrics used (sufficient for report / viva)

| Metric | Meaning | Better when |
|--------|---------|-------------|
| **Top-1** | ACO’s best route is a true optimal order | Yes |
| **Top-3 hit** | True best appears in ACO’s top 3 | Yes |
| **Score gap %** | (true best − ACO) / true best × 100 | **Lower** (0% = perfect) |
| **ACO rank** | Position of ACO’s route among all orders | **1** = optimal |

These are computed by **exhaustive enumeration**: every order of the must-visit places is scored with the same preference formula as the app; the highest score is the true best; ACO is compared to it.

## When it runs

- Must-visit list with **≤ 7** intermediate places (so all orders are computable)
- Example: 5 intermediate places → 120 orders

## Run offline

```bash
python src/check_accuracy.py
```

## Note

Method comparison tables (ACO vs Nearest-Neighbour baselines) are **not** required for evaluation. Top-1, Top-3, score gap, and ACO rank alone are enough to report accuracy against the true best order.
