"""Normalization vocabulary evidence — how the rule words were chosen.

The regex rules in extraction/normalizer.py map text to canonical values using
word/phrase patterns (packed, crowded, quiet, expensive, early morning ...).
Historically those words were picked by intuition. This script produces the
*corpus evidence* that justifies them, answering the examiner's question
"how did you choose these words?" with data instead of assertion:

  1. Frequency table — the most common unigrams & bigrams in the real sentences
     the pipeline tagged for each category. The rule vocabulary should track
     this list. ("I built the patterns from the most frequent expressions.")
  2. Rule coverage — the % of tagged sentences on which at least one existing
     rule fires (vs. abstains). Quantifies how much of real usage the rules
     already capture.
  3. Gaps — frequent expressions that NO current rule matches. This is the
     data-driven to-do list for improving the rules (defensible, not guessed).

Input : output/analysis/_{category}.csv  (already produced by run_analysis.py)
Output: output/evaluation/norm_vocab_report.json  + printed tables
Run   : python analyze_norm_vocab.py
"""

import csv
import json
import os
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from extraction.normalizer import (
    _TIME_OF_DAY, _SEASON, _DAY_TYPE, _CROWD_LEVELS, _COST_EVAL, _AMOUNT_PATTERN,
)
from config.settings import ANALYSIS_OUTPUT_DIR, OUTPUT_DIR

CATEGORIES = ["best_time", "crowd_level", "cost_level"]
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")
TOP_N = 25

_WORD = re.compile(r"[a-z]+")
# Keep a few short function-ish words that are meaningful for these categories.
_KEEP = {"am", "pm", "no", "not"}
_STOP = (ENGLISH_STOP_WORDS - _KEEP)


def _load_tagged_sentences(cat: str) -> list:
    """Read the matched sentences the detector tagged for a category."""
    path = os.path.join(ANALYSIS_OUTPUT_DIR, f"_{cat}.csv")
    sents = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get(f"{cat}_sentences", "")
            if not raw:
                continue
            try:
                for s in json.loads(raw):
                    if s and s.strip():
                        sents.append(s.strip())
            except (json.JSONDecodeError, TypeError):
                continue
    return sents


def _tokens(text: str) -> list:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1 or w in _KEEP]


def _freq_tables(sentences: list):
    uni = Counter()
    bi = Counter()
    for s in sentences:
        toks = _tokens(s)
        uni.update(toks)
        bi.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    return uni, bi


def _rule_fires(cat: str, sentence: str) -> bool:
    """True if any existing normalizer rule matches this sentence."""
    text = sentence.lower()
    if cat == "best_time":
        banks = [_TIME_OF_DAY, _SEASON, _DAY_TYPE]
        for bank in banks:
            for _label, patterns in bank:
                if any(re.search(p, text) for p in patterns):
                    return True
        # month names also count as a best_time signal
        if re.search(r"\b(january|february|march|april|may|june|july|august|"
                     r"september|october|november|december)\b", text):
            return True
        return False
    if cat == "crowd_level":
        for _lvl, _lbl, patterns in _CROWD_LEVELS:
            if any(re.search(p, text) for p in patterns):
                return True
        return False
    if cat == "cost_level":
        if _AMOUNT_PATTERN.search(text):
            return True
        for _label, patterns in _COST_EVAL:
            if any(re.search(p, text) for p in patterns):
                return True
        return False
    return False


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    report = {}

    for cat in CATEGORIES:
        sentences = _load_tagged_sentences(cat)
        uni, bi = _freq_tables(sentences)

        n = len(sentences)
        covered_sents = [s for s in sentences if _rule_fires(cat, s)]
        uncovered_sents = [s for s in sentences if not _rule_fires(cat, s)]
        coverage = round(len(covered_sents) / n, 4) if n else 0.0

        # Real gaps = the vocabulary of the sentences the rules actually MISS.
        # (Computed on whole sentences, not bare words, so context is respected.)
        gap_uni, gap_bi = _freq_tables(uncovered_sents)
        gaps = gap_uni.most_common(15)

        report[cat] = {
            "n_tagged_sentences": n,
            "rule_coverage": coverage,
            "abstention": round(1 - coverage, 4),
            "n_uncovered_sentences": len(uncovered_sents),
            "top_unigrams": uni.most_common(TOP_N),
            "top_bigrams": bi.most_common(TOP_N),
            "top_terms_in_missed_sentences": gaps,
            "top_bigrams_in_missed_sentences": gap_bi.most_common(15),
        }

        print(f"\n{'='*66}\n{cat}   ({n} tagged sentences)\n{'='*66}")
        print(f"Rule coverage: {coverage:.1%}   (abstention {1-coverage:.1%})")
        print(f"\nTop {TOP_N} unigrams:")
        print("  " + ", ".join(f"{w}({c})" for w, c in uni.most_common(TOP_N)))
        print(f"\nTop {TOP_N} bigrams:")
        print("  " + ", ".join(f"{w}({c})" for w, c in bi.most_common(TOP_N)))
        if gaps:
            print(f"\nTop terms in the {len(uncovered_sents)} sentences the rules MISS "
                  f"(data-driven candidates to add):")
            print("  " + ", ".join(f"{w}({c})" for w, c in gaps))

    out = os.path.join(EVAL_DIR, "norm_vocab_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
