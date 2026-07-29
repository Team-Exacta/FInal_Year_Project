# Second-Annotator Guide — Inter-Annotator Agreement (IAA)

**Purpose:** a second person independently labels a subset of the same sentences, so we can
measure agreement (Cohen's κ) and prove the gold labels are reliable — not one person's opinion.

**Do NOT look at the original labelled sheets** (`gold_label_sample.csv`, `norm_gold_sample.csv`).
The `_ann2` sheets have all labels and model hints blanked on purpose — label from the sentence alone.

Subset size: **149 sentences** (binary), of which **79** also need normalization values.

---

## Step 1 — Binary detection labels (all 149 sentences)

For each sentence, decide **yes/no** for each of the three categories:

| Category | Mark YES when the sentence gives... |
|---|---|
| **Best Time** | advice about *when* to go — time of day, month, season, weekday vs weekend, "avoid holidays" |
| **Crowd Level** | how busy/empty it is — "packed", "quiet", "no crowds", "long queues" |
| **Cost Level** | money/fee info — an amount, "free entry", "expensive", "ticket price" |

A sentence can be YES for several categories, or NO for all three.

Run:
```
LABEL_CSV=output/evaluation/gold_label_sample_ann2.csv streamlit run labeling_ui.py
```

## Step 2 — Normalized values (the 79 positive sentences)

For sentences that ARE about a category, pick the correct **canonical value**:
- **Best time →** time of day (EARLY_MORNING … NIGHT), season (DRY/MONSOON/SHOULDER), day type
- **Crowd →** a level **1–5** (1 = empty, 3 = moderate, 5 = packed)
- **Cost →** a band (FREE / LOW / MODERATE / HIGH / VERY_HIGH) + amount if a number is given

Use **NOT_STATED** when the sentence is about the category but gives no concrete value.

Run:
```
LABEL_CSV=output/evaluation/norm_gold_sample_ann2.csv streamlit run labeling_ui_norm.py
```

---

## Step 3 — Compute agreement (you, after they finish)

```
python run_eval_iaa.py
```

Reads κ guide: <0.20 slight · 0.21–0.40 fair · 0.41–0.60 moderate · 0.61–0.80 substantial · 0.81+ almost perfect.
Aim for **substantial (≥0.6)** — that is what makes the gold sample defensible in the write-up.
