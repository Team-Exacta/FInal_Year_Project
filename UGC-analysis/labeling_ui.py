"""Manual labeling UI for gold-label evaluation.

Run with:
  streamlit run labeling_ui.py

To label a different sheet (e.g. the 2nd-annotator IAA copy) set LABEL_CSV:
  LABEL_CSV=output/evaluation/gold_label_sample_ann2.csv streamlit run labeling_ui.py
"""

import os
import pandas as pd
import streamlit as st

CSV_PATH = os.environ.get(
    "LABEL_CSV", os.path.join("output", "evaluation", "gold_label_sample.csv"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Review Sentence Labeler",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load data (cached so edits don't reload)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    return df

def save_data(df, path):
    df.to_csv(path, index=False)

# ---------------------------------------------------------------------------
# Sidebar — labeling guide
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Labeling Guide")

    st.markdown("---")
    st.markdown("### How to label")
    st.markdown(
        "Read the sentence. Decide **Yes** or **No** for each of the three categories. "
        "A sentence can belong to multiple categories at once."
    )

    st.markdown("---")

    st.markdown("#### ⏰ Best Time to Visit")
    st.markdown(
        "**Yes** if the sentence tells you **when** to visit — a specific time recommendation."
    )
    st.markdown("**Consider YES for:**")
    st.markdown(
        "- Time of day: *early morning, before 9am, at sunset, at night*\n"
        "- Season: *dry season, monsoon, December to April*\n"
        "- Day type: *weekdays, avoid weekends, public holidays*\n"
        "- Advice verb: *go early, visit in the morning, come before 8*\n"
        "- Avoid phrasing: *avoid peak hours, don't go on weekends*"
    )
    st.markdown("**Say NO if:**")
    st.markdown(
        "- Just says 'beautiful at any time' (no specific timing)\n"
        "- Mentions a date the reviewer visited, not advice for future visitors\n"
        "- Only mentions the weather without timing advice"
    )

    st.markdown("---")

    st.markdown("#### 👥 Crowd Level")
    st.markdown(
        "**Yes** if the sentence describes **how crowded** the place is or was."
    )
    st.markdown("**Consider YES for:**")
    st.markdown(
        "- Crowd words: *crowded, packed, busy, overrun, swarming*\n"
        "- Quiet words: *quiet, empty, deserted, peaceful, no one there*\n"
        "- Queue/wait: *long queue, had to wait, no queue*\n"
        "- Relative: *lots of tourists, few visitors, had the place to ourselves*"
    )
    st.markdown("**Say NO if:**")
    st.markdown(
        "- Just says 'popular' without describing actual crowd experience\n"
        "- Mentions 'many things to see' (about content, not people)\n"
        "- Generic: 'I went with my family'"
    )

    st.markdown("---")

    st.markdown("#### 💰 Cost Level")
    st.markdown(
        "**Yes** if the sentence mentions **money, fees, or cost**."
    )
    st.markdown("**Consider YES for:**")
    st.markdown(
        "- Amounts: *Rs 1500, LKR 5000, $10*\n"
        "- Fee words: *entry fee, ticket, admission, conservation fee*\n"
        "- Free: *free entry, no charge, free of charge*\n"
        "- Opinion: *expensive, cheap, affordable, overpriced, value for money, worth it*"
    )
    st.markdown("**Say NO if:**")
    st.markdown(
        "- 'Worth visiting' means worth the trip, not the price\n"
        "- Only mentions transport cost, not the attraction fee\n"
        "- 'Priceless view' is figurative, not cost information"
    )

    st.markdown("---")
    st.markdown("#### Tricky cases")
    st.markdown(
        "- *'Go early to avoid crowds'* → **best_time ✓** + **crowd_level ✓**\n"
        "- *'Expensive but worth it'* → **cost_level ✓** only\n"
        "- *'Not crowded at all'* → **crowd_level ✓** (negation counts)\n"
        "- *'Beautiful and peaceful'* → crowd_level ✓ if *peaceful* means few people; NO if just atmosphere"
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Travel Review Sentence Labeler")

if not os.path.exists(CSV_PATH):
    st.error(f"CSV not found: {CSV_PATH}\nRun: python run_eval_sample.py")
    st.stop()

df = load_data(CSV_PATH)

# Count progress
labeled_mask = (
    (df["true_best_time"].isin(["0", "1"])) &
    (df["true_crowd_level"].isin(["0", "1"])) &
    (df["true_cost_level"].isin(["0", "1"]))
)
n_labeled = labeled_mask.sum()
n_total   = len(df)

st.progress(n_labeled / n_total, text=f"Progress: {n_labeled} / {n_total} labeled")

if n_labeled == n_total:
    st.success("All sentences labeled! Run: python run_eval_score.py")
    st.stop()

# Find first unlabeled row
unlabeled_idx = df[~labeled_mask].index.tolist()

# Session state for navigation
if "current_idx" not in st.session_state or st.session_state.current_idx not in unlabeled_idx:
    st.session_state.current_idx = unlabeled_idx[0]

idx = st.session_state.current_idx
row = df.loc[idx]

# ---------------------------------------------------------------------------
# Sentence display
# ---------------------------------------------------------------------------

col_main, col_jump = st.columns([4, 1])

with col_main:
    st.markdown(f"**Place:** `{row['place'].replace('_', ' ')}`")

with col_jump:
    jump_to = st.selectbox(
        "Jump to row",
        options=unlabeled_idx,
        index=unlabeled_idx.index(idx),
        format_func=lambda i: f"Row {i + 1}",
        key="jump_select",
    )
    if jump_to != idx:
        st.session_state.current_idx = jump_to
        st.rerun()

st.markdown("---")
st.markdown(
    f"<div style='background:#1e1e2e; padding:20px; border-radius:8px; "
    f"font-size:1.2rem; line-height:1.8; color:#cdd6f4;'>"
    f"{row['sentence']}"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# System prediction display
# ---------------------------------------------------------------------------

sys_bt = row.get("system_best_time", "0") == "1"
sys_cl = row.get("system_crowd_level", "0") == "1"
sys_co = row.get("system_cost_level", "0") == "1"

st.markdown("**System predicted:** "
    + (":blue[⏰ Best Time]  " if sys_bt else "")
    + (":orange[👥 Crowd]  "  if sys_cl else "")
    + (":green[💰 Cost]  "    if sys_co else "")
    + ("_(none)_" if not any([sys_bt, sys_cl, sys_co]) else "")
)

st.markdown("---")
st.markdown("### Your labels")

# ---------------------------------------------------------------------------
# Label buttons — 3 columns, one per category
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

def _label_col(col, label_key: str, display_name: str, emoji: str, color: str):
    current = row.get(label_key, "")
    with col:
        st.markdown(f"**{emoji} {display_name}**")
        yes_btn = st.button("✅  Yes", key=f"yes_{label_key}", use_container_width=True,
                            type="primary" if current == "1" else "secondary")
        no_btn  = st.button("❌  No",  key=f"no_{label_key}",  use_container_width=True,
                            type="primary" if current == "0" else "secondary")
        if current == "1":
            st.success("Labeled: YES")
        elif current == "0":
            st.error("Labeled: NO")
        else:
            st.info("Not labeled yet")
        return yes_btn, no_btn

yes_bt, no_bt = _label_col(col1, "true_best_time",   "Best Time",   "⏰", "blue")
yes_cl, no_cl = _label_col(col2, "true_crowd_level", "Crowd Level", "👥", "orange")
yes_co, no_co = _label_col(col3, "true_cost_level",  "Cost Level",  "💰", "green")

# ---------------------------------------------------------------------------
# Handle button clicks and auto-advance
# ---------------------------------------------------------------------------

def apply_labels(bt=None, cl=None, co=None):
    """Write labels, save CSV, advance to next row."""
    # Reload fresh (bypass cache) to avoid overwriting concurrent edits
    fresh = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    if bt is not None:
        fresh.at[idx, "true_best_time"]   = str(bt)
    if cl is not None:
        fresh.at[idx, "true_crowd_level"] = str(cl)
    if co is not None:
        fresh.at[idx, "true_cost_level"]  = str(co)
    save_data(fresh, CSV_PATH)
    st.cache_data.clear()

    # Advance to next unlabeled row
    remaining = fresh[
        ~(
            fresh["true_best_time"].isin(["0","1"]) &
            fresh["true_crowd_level"].isin(["0","1"]) &
            fresh["true_cost_level"].isin(["0","1"])
        )
    ].index.tolist()
    if remaining:
        st.session_state.current_idx = remaining[0]

# Apply individual category clicks
if yes_bt: apply_labels(bt=1); st.rerun()
if no_bt:  apply_labels(bt=0); st.rerun()
if yes_cl: apply_labels(cl=1); st.rerun()
if no_cl:  apply_labels(cl=0); st.rerun()
if yes_co: apply_labels(co=1); st.rerun()
if no_co:  apply_labels(co=0); st.rerun()

st.markdown("---")

# Quick-submit: all three at once then advance
st.markdown("**Quick submit all three and go to next:**")

qcols = st.columns(8)
combos = [
    ("None of these", 0, 0, 0),
    ("Best Time only", 1, 0, 0),
    ("Crowd only",     0, 1, 0),
    ("Cost only",      0, 0, 1),
    ("BT + Crowd",     1, 1, 0),
    ("BT + Cost",      1, 0, 1),
    ("Crowd + Cost",   0, 1, 1),
    ("All three",      1, 1, 1),
]
for col, (label, bt, cl, co) in zip(qcols, combos):
    with col:
        if st.button(label, use_container_width=True, key=f"quick_{label}"):
            apply_labels(bt=bt, cl=cl, co=co)
            st.rerun()
