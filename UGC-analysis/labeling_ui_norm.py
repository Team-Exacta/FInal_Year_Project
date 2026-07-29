"""Manual labeling UI for the NORMALIZATION gold layer (Part B, step 2).

Complements labeling_ui.py (which collects the binary category labels). Here the
annotator supplies the CORRECT canonical value for each sentence that is a
positive for a category, so we can later measure whether the regex normalizer
maps text to the right value.

Only positives are asked about:
  best_time positive -> time_of_day, season, day_type
  crowd positive     -> crowd_ordinal (1-5)
  cost positive      -> cost_band (+ optional amount_lkr)

The system's own prediction is shown for reference (label independently — don't
just copy it). NONE / NOT_STATED is a valid answer when the sentence is about the
category but gives no concrete value (e.g. "best time to visit" with no time).

Run:
  python run_eval_norm_sample.py      # (re)build the sheet first
  streamlit run labeling_ui_norm.py

To label a different sheet (e.g. the 2nd-annotator IAA copy) set LABEL_CSV:
  LABEL_CSV=output/evaluation/norm_gold_sample_ann2.csv streamlit run labeling_ui_norm.py
"""

import os
import pandas as pd
import streamlit as st

CSV_PATH = os.environ.get(
    "LABEL_CSV", os.path.join("output", "evaluation", "norm_gold_sample.csv"))

# Allowed canonical values (must match extraction/normalizer.py).
TIME_OF_DAY = ["EARLY_MORNING", "MID_MORNING", "AFTERNOON", "EVENING", "NIGHT", "NOT_STATED"]
SEASON      = ["DRY_SEASON", "MONSOON", "SHOULDER", "NOT_STATED"]
DAY_TYPE    = ["WEEKDAY", "WEEKEND", "PUBLIC_HOLIDAY", "NOT_STATED"]
CROWD       = ["1 (EMPTY)", "2 (QUIET)", "3 (MODERATE)", "4 (BUSY)", "5 (PACKED)", "NOT_STATED"]
COST_BAND   = ["FREE", "LOW", "MODERATE", "HIGH", "VERY_HIGH", "NOT_STATED"]

# Which true_* value columns are required for a positive of each category.
REQUIRED = {
    "best_time":   ["true_time_of_day", "true_season", "true_day_type"],
    "crowd_level": ["true_crowd_ordinal"],
    "cost_level":  ["true_cost_band"],
}

st.set_page_config(page_title="Normalized-Value Labeler", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data
def load_data(path):
    return pd.read_csv(path, dtype=str).fillna("")


def _radio(label, options, current, key):
    """Radio pre-selected to the saved value if present, else nothing."""
    idx = options.index(current) if current in options else None
    return st.radio(label, options, index=idx, key=key, horizontal=True)


with st.sidebar:
    st.title("Value Labeling Guide")
    st.markdown(
        "For each **positive** sentence, choose the canonical value the sentence "
        "actually conveys. Pick **NOT_STATED** if the category applies but no "
        "concrete value is given.")
    st.markdown("---")
    st.markdown("**Crowd scale**")
    st.markdown("1 EMPTY · 2 QUIET · 3 MODERATE · 4 BUSY · 5 PACKED\n\n"
                "*'not crowded' → 2, 'a bit busy' → 3-4, 'packed' → 5.*")
    st.markdown("**Cost band**")
    st.markdown("FREE · LOW (<500) · MODERATE (500-2000) · HIGH (2000-5000) · VERY_HIGH (>5000).\n\n"
                "Judge by the *meaning*; put the LKR number in amount if one is stated.")
    st.markdown("**Season**: DRY (Dec-Apr) · MONSOON (May-Oct) · SHOULDER (Nov).")

st.title("Normalized-Value Labeler")

if not os.path.exists(CSV_PATH):
    st.error(f"Not found: {CSV_PATH}\nRun: python run_eval_norm_sample.py")
    st.stop()

df = load_data(CSV_PATH)


def _is_positive(row, cat):
    return str(row.get(f"true_{cat}", "")).strip() == "1"


def _row_needs_label(row):
    """A row needs work if it is a positive for a category whose required
    value columns are not all filled in."""
    for cat, cols in REQUIRED.items():
        if _is_positive(row, cat) and any(str(row.get(c, "")).strip() == "" for c in cols):
            return True
    return False


todo = [i for i in df.index if _row_needs_label(df.loc[i])]
n_pos_rows = sum(1 for i in df.index if any(_is_positive(df.loc[i], c) for c in REQUIRED))
st.progress(1 - len(todo) / max(n_pos_rows, 1),
            text=f"{n_pos_rows - len(todo)} / {n_pos_rows} positive rows fully labeled")

if not todo:
    st.success("All positive sentences labeled! Run: python run_eval_norm_score.py")
    st.stop()

if "idx" not in st.session_state or st.session_state.idx not in todo:
    st.session_state.idx = todo[0]
idx = st.session_state.idx
row = df.loc[idx]

st.markdown(f"**Place:** `{row['place'].replace('_', ' ')}`  &nbsp;&nbsp; (row {idx + 1})")
st.markdown(
    f"<div style='background:#1e1e2e;padding:18px;border-radius:8px;"
    f"font-size:1.15rem;color:#cdd6f4;'>{row['sentence']}</div>",
    unsafe_allow_html=True)
st.markdown("---")

pending = {}   # column -> chosen value

if _is_positive(row, "best_time"):
    st.markdown("### ⏰ Best time")
    st.caption(f"system: time={row['system_time_of_day'] or '—'}, "
               f"season={row['system_season'] or '—'}, day={row['system_day_type'] or '—'}")
    pending["true_time_of_day"] = _radio("Time of day", TIME_OF_DAY, row["true_time_of_day"], "tod")
    pending["true_season"] = _radio("Season", SEASON, row["true_season"], "sea")
    pending["true_day_type"] = _radio("Day type", DAY_TYPE, row["true_day_type"], "day")
    st.markdown("---")

if _is_positive(row, "crowd_level"):
    st.markdown("### 👥 Crowd level")
    st.caption(f"system: {row['system_crowd_ordinal'] or '—'} ({row['system_crowd_label'] or '—'})")
    _cur = row["true_crowd_ordinal"].strip()
    _sel = next((c for c in CROWD if c == _cur or (_cur and c.startswith(_cur))), "")
    choice = _radio("Crowd", CROWD, _sel, "cro")
    # store "NOT_STATED" verbatim, otherwise just the digit (1-5)
    pending["true_crowd_ordinal"] = ("NOT_STATED" if choice == "NOT_STATED"
                                     else (choice[0] if choice else ""))
    st.markdown("---")

if _is_positive(row, "cost_level"):
    st.markdown("### 💰 Cost")
    st.caption(f"system: band={row['system_cost_band'] or '—'}, amount={row['system_amount_lkr'] or '—'}")
    pending["true_cost_band"] = _radio("Cost band", COST_BAND, row["true_cost_band"], "cost")
    pending["true_amount_lkr"] = st.text_input("Amount in LKR (optional, number only)",
                                               value=row["true_amount_lkr"], key="amt")
    st.markdown("---")

pending["notes"] = st.text_input("Notes (optional)", value=row.get("notes", ""), key="notes")

col_save, col_skip = st.columns(2)


def _save_and_advance():
    fresh = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    for col, val in pending.items():
        if val is None:
            val = ""
        fresh.at[idx, col] = str(val)
    fresh.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()
    remaining = [i for i in fresh.index if _row_needs_label(fresh.loc[i])]
    if remaining:
        st.session_state.idx = remaining[0]


with col_save:
    if st.button("💾  Save & next", type="primary", use_container_width=True):
        # require every shown required field to have a value (NOT_STATED counts)
        missing = [c for cat, cols in REQUIRED.items() if _is_positive(row, cat)
                   for c in cols if not str(pending.get(c, "")).strip()]
        if missing:
            st.warning("Please answer every question (use NOT_STATED if unsure).")
        else:
            _save_and_advance()
            st.rerun()
with col_skip:
    if st.button("Skip →", use_container_width=True):
        nxt = [i for i in todo if i > idx]
        st.session_state.idx = nxt[0] if nxt else todo[0]
        st.rerun()
