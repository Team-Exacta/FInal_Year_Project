"""Detect and remove duplicate reviews."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DUPLICATE_SIMILARITY_THRESHOLD


def deduplicate_by_url(reviews):
    """Remove reviews with duplicate URLs.

    Returns (deduplicated_reviews, removed_count).
    """
    seen_urls = set()
    unique = []
    removed = 0

    for review in reviews:
        url = review.get("url", "")
        if url and url in seen_urls:
            removed += 1
            continue
        if url:
            seen_urls.add(url)
        unique.append(review)

    return unique, removed


def find_fuzzy_duplicates(reviews, threshold=None):
    """Find near-duplicate reviews using TF-IDF cosine similarity.

    Returns list of (index_i, index_j, similarity) tuples for duplicates.
    """
    if threshold is None:
        threshold = DUPLICATE_SIMILARITY_THRESHOLD

    texts = [r.get("text_clean", r.get("text", "")) for r in reviews]

    if len(texts) < 2:
        return []

    # Build TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    # Compute pairwise similarity
    duplicates = []
    sim_matrix = cosine_similarity(tfidf_matrix)

    for i in range(len(reviews)):
        for j in range(i + 1, len(reviews)):
            if sim_matrix[i][j] >= threshold:
                duplicates.append((i, j, float(sim_matrix[i][j])))

    return duplicates


def deduplicate_fuzzy(reviews, threshold=None):
    """Remove fuzzy duplicates, keeping the review with more complete data.

    Returns (deduplicated_reviews, removed_count).
    """
    duplicates = find_fuzzy_duplicates(reviews, threshold)

    if not duplicates:
        return reviews, 0

    # Determine which indices to remove
    indices_to_remove = set()
    for i, j, sim in duplicates:
        if i in indices_to_remove or j in indices_to_remove:
            continue
        # Keep the one with more data (longer text, non-null user, etc.)
        score_i = len(reviews[i].get("text", "")) + (1000 if reviews[i].get("user") else 0)
        score_j = len(reviews[j].get("text", "")) + (1000 if reviews[j].get("user") else 0)
        indices_to_remove.add(j if score_i >= score_j else i)

    result = [r for idx, r in enumerate(reviews) if idx not in indices_to_remove]
    return result, len(indices_to_remove)


def deduplicate_reviews(reviews):
    """Full deduplication pipeline: URL-based then fuzzy.

    Returns (deduplicated_reviews, stats_dict).
    """
    # Step 1: Exact URL dedup
    reviews, url_removed = deduplicate_by_url(reviews)

    # Step 2: Fuzzy dedup
    reviews, fuzzy_removed = deduplicate_fuzzy(reviews)

    stats = {
        "url_duplicates_removed": url_removed,
        "fuzzy_duplicates_removed": fuzzy_removed,
        "total_removed": url_removed + fuzzy_removed,
    }

    return reviews, stats
