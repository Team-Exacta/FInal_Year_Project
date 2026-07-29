"""Detect and filter non-English reviews."""

from langdetect import detect_langs, DetectorFactory

# Set seed for reproducibility
DetectorFactory.seed = 0

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import LANGUAGE_CONFIDENCE_THRESHOLD, MIN_TEXT_LENGTH_FOR_LANG_DETECT


def detect_language(text):
    """Detect the language of a text string.

    Returns (language_code, confidence) or ('unknown', 0.0) on failure.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH_FOR_LANG_DETECT:
        return "en", 1.0  # Too short to detect, assume English

    try:
        results = detect_langs(text)
        # Results are sorted by probability descending
        top = results[0]
        return str(top.lang), top.prob
    except Exception:
        return "unknown", 0.0


def is_english(text, threshold=None):
    """Check if text is English with sufficient confidence."""
    if threshold is None:
        threshold = LANGUAGE_CONFIDENCE_THRESHOLD

    lang, confidence = detect_language(text)
    return lang == "en" and confidence >= threshold


def filter_english_reviews(reviews):
    """Filter a list of reviews to keep only English ones.

    Returns (english_reviews, filtered_out) tuple.
    """
    english = []
    filtered_out = []

    for review in reviews:
        # Combine title and text for better detection
        combined = f"{review.get('title', '')} {review.get('text', '')}"

        if is_english(combined):
            english.append(review)
        else:
            lang, conf = detect_language(combined)
            filtered_out.append({
                "title": review.get("title", ""),
                "detected_lang": lang,
                "confidence": conf,
                "url": review.get("url", "")
            })

    return english, filtered_out
