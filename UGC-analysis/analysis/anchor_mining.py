"""Data-driven anchor mining — derive detection anchors from the real corpus.

Motivation
----------
The detector's positive anchors (analysis/detector.py :: CATEGORY_ANCHORS) were
hand-written from intuition. That is hard to defend ("how did you get these?").
This module replaces intuition with a **reproducible recipe** that reads the
anchors out of the corpus itself:

  1. Encode every review sentence in output/cleaned_data with SBERT (once).
  2. For each category, keep the sentences whose contrastive net-score is within
     a recall-generous band ( >= threshold - MARGIN ). This captures both the
     confident phrasings and the false-negative tail the current anchors *almost*
     catch — so mined anchors can cover phrasings the hand-written ones miss.
  3. Drop every sentence that appears in the gold evaluation set (LEAKAGE GUARD):
     anchors are mined only from non-test text, so any later F1 gain is honest.
  4. Cluster each category's pool with KMeans. Each cluster = one recurring way
     visitors phrase the constraint.
  5. Take each cluster's MEDOID (the real sentence nearest the cluster centre) as
     an anchor. The cluster size = how common that phrasing is ("the most used
     one"). Larger clusters first.

Output
------
analysis/mined_anchors.json — for every category, a list of
  {anchor, cluster_size, score, examples[]}
ordered by cluster_size (most common phrasing first). This file is the auditable
evidence of *where each anchor came from*.

Run:
  python -m analysis.anchor_mining
  python -m analysis.anchor_mining --k 14 --margin 0.10 --max-anchors 12
"""

import argparse
import csv
import json
import os

import numpy as np
from sklearn.cluster import KMeans

from analysis.detector import ReviewAnalysisDetector, CATEGORIES, NEGATIVE_ALPHA
from config.settings import (
    ANALYSIS_MODEL, ANALYSIS_SCORE_THRESHOLD, ANALYSIS_SCORE_THRESHOLDS,
    ANALYSIS_BATCH_SIZE, ANALYSIS_OUTPUT_DIR, CLEANED_DATA_DIR, OUTPUT_DIR,
)

GOLD_CSV = os.path.join(OUTPUT_DIR, "evaluation", "gold_label_sample.csv")
MINED_ANCHORS_FILE = os.path.join(os.path.dirname(__file__), "mined_anchors.json")

# Cache of the (expensive) whole-corpus encode, so clustering params can be
# re-tuned in seconds instead of re-encoding ~169k sentences each run.
CORPUS_EMB_CACHE  = os.path.join(ANALYSIS_OUTPUT_DIR, "_corpus_embeddings.npz")
CORPUS_SENT_CACHE = os.path.join(ANALYSIS_OUTPUT_DIR, "_corpus_sentences.json")

# Defaults (overridable on the CLI)
DEFAULT_MARGIN        = 0.05   # pool = sentences whose BEST-category net_score >= threshold - margin
DEFAULT_K             = 60     # KMeans clusters per category (many → tight, specific medoids)
DEFAULT_MAX_ANCHORS   = 12     # anchors kept per category (largest clusters)
DEFAULT_MIN_CLUSTER   = 4      # ignore clusters smaller than this (noise)
DEFAULT_CONF_MARGIN   = 0.05   # sentence's best-cat score must beat 2nd-best by this
NEG_SIM_MAX           = 0.60   # drop a medoid too similar to a negative anchor
SEED                  = 42


# ---------------------------------------------------------------------------
# Corpus + gold loading
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Canonical form for matching sentences across files."""
    return " ".join(s.strip().lower().split())


def load_gold_sentences() -> set:
    """Return the set of normalized sentences used in the gold evaluation set."""
    if not os.path.exists(GOLD_CSV):
        print(f"WARNING: gold set not found at {GOLD_CSV} — leakage guard is a no-op.")
        return set()
    gold = set()
    with open(GOLD_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("sentence"):
                gold.add(_norm(row["sentence"]))
    return gold


def collect_corpus_sentences(detector: ReviewAnalysisDetector) -> list:
    """Tokenise every cleaned review into unique candidate sentences."""
    seen = set()
    sentences = []
    files = sorted(f for f in os.listdir(CLEANED_DATA_DIR)
                   if f.endswith(".json") and not f.startswith("_"))
    for fname in files:
        with open(os.path.join(CLEANED_DATA_DIR, fname), encoding="utf-8") as f:
            reviews = json.load(f)
        for review in reviews:
            text = (review.get("text_display") or review.get("text") or "")
            for sent in detector._tokenize_sentences(text):
                key = _norm(sent)
                if key and key not in seen:
                    seen.add(key)
                    sentences.append(sent)
    return sentences


def get_corpus_embeddings(detector: ReviewAnalysisDetector, refresh: bool = False):
    """Return (sentences, embeddings), using an on-disk cache when available.

    Encoding ~169k sentences on CPU is the only slow step; caching it means
    re-tuning the clustering (k / margin) is near-instant.
    """
    if (not refresh and os.path.exists(CORPUS_EMB_CACHE)
            and os.path.exists(CORPUS_SENT_CACHE)):
        print("Loading cached corpus embeddings...")
        with open(CORPUS_SENT_CACHE, encoding="utf-8") as f:
            sentences = json.load(f)
        embs = np.load(CORPUS_EMB_CACHE)["embs"].astype(np.float32)
        print(f"  {len(sentences)} sentences (cached).")
        return sentences, embs

    print("Collecting corpus sentences...")
    sentences = collect_corpus_sentences(detector)
    print(f"  {len(sentences)} unique sentences. Encoding (once)...")
    embs = detector._model.encode(
        sentences, batch_size=ANALYSIS_BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=True,
    )
    embs = np.asarray(embs, dtype=np.float32)
    # Cache as float16 to halve disk; precision loss is negligible for clustering.
    np.savez_compressed(CORPUS_EMB_CACHE, embs=embs.astype(np.float16))
    with open(CORPUS_SENT_CACHE, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False)
    print(f"  cached -> {CORPUS_EMB_CACHE}")
    return sentences, embs


# ---------------------------------------------------------------------------
# Scoring + clustering
# ---------------------------------------------------------------------------

def _net_scores_all(detector: ReviewAnalysisDetector, embs: np.ndarray, cat: str) -> np.ndarray:
    """Vectorised contrastive net score for every embedding, for one category.

    Mirrors ReviewAnalysisDetector._net_score but over the whole matrix at once.
    """
    pos = (detector._anchor_emb[cat] @ embs.T).max(axis=0)
    neg = (detector._negative_emb @ embs.T).max(axis=0)
    cat_neg = detector._cat_negative_emb.get(cat)
    if cat_neg is not None and len(cat_neg) > 0:
        neg = np.maximum(neg, (cat_neg @ embs.T).max(axis=0))
    return pos - NEGATIVE_ALPHA * neg


def _max_negative_sim(detector: ReviewAnalysisDetector, emb: np.ndarray, cat: str) -> float:
    """Max cosine similarity of one embedding to any negative anchor (global+cat)."""
    sim = float((detector._negative_emb @ emb).max())
    cat_neg = detector._cat_negative_emb.get(cat)
    if cat_neg is not None and len(cat_neg) > 0:
        sim = max(sim, float((cat_neg @ emb).max()))
    return sim


def mine_category(detector, sentences, embs, scores, cat,
                  k, max_anchors, min_cluster, min_score=0.0, examples_per=3) -> list:
    """Cluster one category's pool and return medoid anchors, largest cluster first.

    min_score : drop a cluster whose medoid scores below this (weak / off-topic
                anchors that would only add false positives).
    """
    n = len(sentences)
    if n == 0:
        return []
    k = max(2, min(k, n))
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(embs)

    anchors = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) < min_cluster:
            continue
        centroid = km.cluster_centers_[c]
        # Medoid = real sentence whose embedding is closest to the centroid.
        sims = embs[idx] @ centroid
        order = idx[np.argsort(-sims)]
        medoid_i = int(order[0])
        # Skip medoids that are really just generic travel praise.
        if _max_negative_sim(detector, embs[medoid_i], cat) > NEG_SIM_MAX:
            continue
        # Skip weak/off-topic medoids (would only add false positives).
        if float(scores[medoid_i]) < min_score:
            continue
        examples = [sentences[int(j)] for j in order[:examples_per]]
        anchors.append({
            "anchor":       sentences[medoid_i],
            "cluster_size": int(len(idx)),
            "score":        round(float(scores[medoid_i]), 4),
            "examples":     examples,
        })

    anchors.sort(key=lambda a: a["cluster_size"], reverse=True)
    return anchors[:max_anchors]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def mine_all(margin=DEFAULT_MARGIN, k=DEFAULT_K, max_anchors=DEFAULT_MAX_ANCHORS,
             min_cluster=DEFAULT_MIN_CLUSTER, conf_margin=DEFAULT_CONF_MARGIN,
             write=True, refresh=False) -> dict:
    """Run the full mining recipe and (optionally) write mined_anchors.json.

    Returns the detailed dict {cat: [ {anchor, cluster_size, score, examples} ]}.
    """
    detector = ReviewAnalysisDetector(ANALYSIS_MODEL, ANALYSIS_SCORE_THRESHOLDS,
                                      ANALYSIS_BATCH_SIZE)
    detector._ensure_model()

    sentences, embs = get_corpus_embeddings(detector, refresh=refresh)

    # Score every sentence for every category, then assign each sentence to the
    # category it fits BEST (argmax). This keeps pools disjoint and on-topic —
    # e.g. a "not crowded but clean beach" sentence goes to crowd, not cost.
    net = {cat: _net_scores_all(detector, embs, cat) for cat in CATEGORIES}
    net_stack = np.vstack([net[c] for c in CATEGORIES])   # (n_cats, N)
    best_cat = net_stack.argmax(axis=0)                    # index into CATEGORIES
    # Runner-up score per sentence → confidence margin of the winning category.
    second_best = np.sort(net_stack, axis=0)[-2]           # (N,)

    gold = load_gold_sentences()
    print(f"Leakage guard: {len(gold)} gold sentences will be excluded from every pool.")

    result = {}
    for ci, cat in enumerate(CATEGORIES):
        thr = ANALYSIS_SCORE_THRESHOLDS.get(cat, ANALYSIS_SCORE_THRESHOLD)
        # Pool = sentences that (a) fit THIS category best, (b) clear the
        # recall-generous band, and (c) favour it *confidently* over the runner-up
        # (drops ambiguous beach/restaurant sentences that score ~equally for two
        # categories). Medoids must additionally score >= the production threshold.
        cand = np.where((best_cat == ci)
                        & (net[cat] >= thr - margin)
                        & ((net[cat] - second_best) >= conf_margin))[0]

        removed = 0
        pool_idx = []
        for i in cand:
            if _norm(sentences[int(i)]) in gold:
                removed += 1
                continue
            pool_idx.append(int(i))
        pool_idx = np.asarray(pool_idx, dtype=int)

        pool_sents  = [sentences[i] for i in pool_idx]
        pool_embs   = embs[pool_idx] if len(pool_idx) else np.empty((0, embs.shape[1]), np.float32)
        pool_scores = net[cat][pool_idx] if len(pool_idx) else np.empty((0,), np.float32)

        anchors = mine_category(detector, pool_sents, pool_embs, pool_scores, cat,
                                k=k, max_anchors=max_anchors, min_cluster=min_cluster,
                                min_score=thr)
        result[cat] = anchors
        print(f"\n=== {cat} ===")
        print(f"  pool: {len(pool_sents)} sentences (argmax==this cat & net >= "
              f"{thr - margin:.2f} & conf>= {conf_margin}); {removed} gold excluded (leakage guard)")
        print(f"  mined {len(anchors)} anchors (largest cluster first):")
        for a in anchors:
            print(f"    [{a['cluster_size']:>4}  s={a['score']:.2f}]  {a['anchor'][:85]}")

    if write:
        with open(MINED_ANCHORS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved -> {MINED_ANCHORS_FILE}")

    return result


def load_mined_anchor_bank(path: str = MINED_ANCHORS_FILE) -> dict:
    """Load mined_anchors.json → {cat: [anchor sentences]} for use as an anchor bank."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {cat: [a["anchor"] for a in items] for cat, items in data.items()}


def main():
    ap = argparse.ArgumentParser(description="Mine data-driven detection anchors from the corpus.")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help="pool = sentences with net_score >= threshold - margin")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="KMeans clusters per category")
    ap.add_argument("--max-anchors", type=int, default=DEFAULT_MAX_ANCHORS,
                    help="anchors kept per category (largest clusters)")
    ap.add_argument("--min-cluster", type=int, default=DEFAULT_MIN_CLUSTER,
                    help="ignore clusters smaller than this")
    ap.add_argument("--conf-margin", type=float, default=DEFAULT_CONF_MARGIN,
                    help="best-category score must beat 2nd-best by this")
    ap.add_argument("--refresh", action="store_true",
                    help="re-encode the corpus even if a cache exists")
    args = ap.parse_args()
    mine_all(margin=args.margin, k=args.k, max_anchors=args.max_anchors,
             min_cluster=args.min_cluster, conf_margin=args.conf_margin, refresh=args.refresh)


if __name__ == "__main__":
    main()
