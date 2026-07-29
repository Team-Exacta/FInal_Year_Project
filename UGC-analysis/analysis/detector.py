"""Sentence-BERT semantic similarity classifier for review tagging.

Approach (Reimers & Gurevych, 2019 — Sentence-BERT + contrastive scoring):
  1. Encode each review sentence with all-MiniLM-L6-v2.
  2. Compute positive similarity against category-specific anchor sentences.
  3. Compute negative similarity against generic travel description sentences.
  4. Net score = positive_sim - alpha * negative_sim (contrastive).
  5. Tag sentences whose net score exceeds threshold.

The contrastive step prevents generic phrases like "highly recommend visiting"
or "beautiful and amazing place" from triggering category tags.

Model: sentence-transformers/all-MiniLM-L6-v2  (~22 MB, ~14k sentences/sec on CPU)
"""

import nltk
import numpy as np
nltk.download("punkt_tab", quiet=True)

CATEGORIES = ["best_time", "crowd_level", "cost_level"]
NEGATIVE_ALPHA = 0.5   # weight for penalising generic travel language

# Positive anchors — specific, fact-dense sentences for each category.
# Deliberately avoid generic "visit" + "recommend" language.
CATEGORY_ANCHORS = {
    "best_time": [
        "Monsoon season from May to October brings heavy rainfall and high water levels.",
        "The dry season from December to April has clear skies and is ideal for trekking.",
        "Go before 9 am to enjoy the falls before the afternoon heat builds up.",
        "January through March offers minimal rainfall and the most pleasant temperatures.",
        "Public holidays see many more visitors; weekdays are far quieter.",
        "Rainy months July and August make waterfalls most spectacular but swimming is unsafe.",
        "Weekday mornings have fewer tourist groups than busy weekends.",
        "The water flow is highest after heavy monsoon rains in June and July.",
        "Avoid peak season in December and Easter when prices and crowds are at their highest.",
        "Early morning arrival lets you beat the crowds and get the best light for photos.",
    ],
    "crowd_level": [
        "The site was packed with tourists and extremely overcrowded when we arrived.",
        "Long queues of visitors stretched back from the entrance on weekends.",
        "We had the entire waterfall to ourselves as there were very few other tourists.",
        "The area is overrun by tourist groups and it is hard to find a quiet moment.",
        "We visited midweek and found the place calm and peaceful with almost no crowds.",
        "The rush of tourist buses on holidays makes enjoyment very difficult.",
        "Despite its popularity this site was surprisingly uncrowded during our visit.",
        "Crowds of visitors made it nearly impossible to take a decent photograph.",
        "The solitude and tranquility here was a welcome break from busy tourist spots.",
        "There were long waits at the entrance gate due to the volume of visitors.",
        # Additions from 600-sample FN analysis — patterns the model was missing
        "It is advisable to arrive before the crowd gets there in the late morning.",
        "Better to avoid weekends and public holidays as the place gets very crowded.",
        "There were noticeably fewer tourists at this site than at other nearby attractions.",
        "Tour groups can be very large with sixty to seventy people arriving together.",
    ],
    "cost_level": [
        "We paid five hundred rupees each at the entrance gate as the entry fee.",
        "The conservation charge at the ticket booth was two hundred rupees per person.",
        "Admission is completely free with no entrance ticket required.",
        "The entry fee of one thousand LKR per foreigner seemed expensive.",
        "There is a small gate fee of three hundred rupees to enter the park.",
        "No ticket price is charged and entry to this attraction is free of charge.",
        "The admission cost was very affordable at only one hundred rupees.",
        "We were surprised that the entrance fee was waived during the holiday weekend.",
        "The ticket costs were worth every rupee given the quality of the experience.",
        "A conservation fee is collected at the checkpoint before the trail begins.",
        # Additions from 600-sample FN analysis — indirect cost patterns
        "Restaurant and food prices around this beach were noticeably higher than other areas.",
        "He told us the rate per person is one thousand rupees including the guide.",
    ],
}

# Negative anchors — generic travel praise that should NOT trigger a category tag.
NEGATIVE_ANCHORS = [
    "This is a beautiful and amazing natural tourist attraction.",
    "I highly recommend visiting this wonderful place with family.",
    "The scenery was breathtaking and the experience was absolutely unforgettable.",
    "Great spot to relax and enjoy the natural surroundings.",
    "The views were spectacular and the atmosphere was peaceful and serene.",
    "One of the best places I have ever visited in Sri Lanka.",
    "Definitely worth seeing if you are in the area.",
    "An incredible natural wonder that should not be missed.",
]

# Category-specific negative anchors.
# These penalise sentences that share vocabulary with a category but carry
# no useful signal — the specific false-positive patterns found in evaluation.
CATEGORY_NEGATIVE_ANCHORS = {
    "best_time": [
        # Past-visit descriptions — not advice for future visitors
        "We visited this beautiful place during our Sri Lanka holiday.",
        "We went there last year and had an amazing time.",
        "I was there today and thoroughly enjoyed the whole experience.",
        "We spent the entire day exploring the historical site together.",
        "Our travel agent recommended this as a must-see destination.",
        "We have been to this waterfall and it was spectacular.",
    ],
    "crowd_level": [
        # Mentions tourists/groups/entrance in a non-crowd-density context
        "We were part of an organised group tour and had a local guide.",
        "Tourist attractions in Sri Lanka are well maintained by authorities.",
        "Our tour guide arranged all the transport and accommodation for us.",
        "Tourists are required to follow the dress code when visiting temples.",
        "The management does not allow photography inside the premises.",
        "I do not know the entrance price as our guide made all the payments.",
        "Tourist facilities at this location could be improved significantly.",
        "We did a hike up to the waterfall as part of our tour group.",
    ],
    "cost_level": [
        # 'Entrance' as physical location, not fee info. Anchors must NOT
        # mention price / fee / ticket / admission terms — those words are
        # the very signal we want to keep.
        "The entrance to the temple is through a small narrow stone gate.",
        "Walk through the main entrance arch to reach the inner courtyard.",
        "The access road leading to the national park is in poor condition.",
        "The market entrance is located on the main street near the bus stop.",
        "The main gate opens at sunrise and closes well before sunset.",
    ],
}


class ReviewAnalysisDetector:
    """Sentence-BERT contrastive tagger for best_time / crowd_level / cost_level."""

    def __init__(self, model_name: str, score_threshold, batch_size: int = 64,
                 category_anchors: dict = None,
                 negative_anchors: list = None,
                 category_negative_anchors: dict = None):
        """
        Parameters
        ----------
        score_threshold : float | dict
            Either a single scalar applied to every category, or a
            {category: threshold} mapping. Categories missing from the dict
            fall back to the smallest threshold provided.
        category_anchors, negative_anchors, category_negative_anchors : optional
            Anchor banks to use for scoring. Default to the module-level
            hand-written banks (CATEGORY_ANCHORS / NEGATIVE_ANCHORS /
            CATEGORY_NEGATIVE_ANCHORS). Passing these in lets the anchor-ablation
            harness swap in data-driven (corpus-mined) banks without editing
            globals, so the same detector code evaluates every configuration.
        """
        self.model_name          = model_name
        # Anchor banks (injectable — default to the hand-written module constants)
        self._category_anchors          = category_anchors if category_anchors is not None else CATEGORY_ANCHORS
        self._negative_anchors          = negative_anchors if negative_anchors is not None else NEGATIVE_ANCHORS
        self._category_negative_anchors = (category_negative_anchors
                                           if category_negative_anchors is not None
                                           else CATEGORY_NEGATIVE_ANCHORS)
        # Normalise threshold to a dict; keep a representative scalar for logs.
        if isinstance(score_threshold, dict):
            self.score_thresholds = {c: float(score_threshold.get(c, min(score_threshold.values())))
                                     for c in CATEGORIES}
            self.score_threshold = min(self.score_thresholds.values())
        else:
            self.score_thresholds = {c: float(score_threshold) for c in CATEGORIES}
            self.score_threshold = float(score_threshold)
        self.batch_size          = batch_size
        self._model              = None
        self._anchor_emb         = {}   # cat -> (n_anchors, dim) normalised
        self._negative_emb       = None # (n_global_neg, dim) normalised
        self._cat_negative_emb   = {}   # cat -> (n_cat_neg, dim) normalised

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model '{self.model_name}' on {device.upper()}...")
        self._model = SentenceTransformer(self.model_name, device=device)

        enc = lambda texts: self._model.encode(
            texts, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )
        for cat, anchors in self._category_anchors.items():
            self._anchor_emb[cat] = enc(anchors)
        self._negative_emb = enc(self._negative_anchors)
        for cat, neg_anchors in self._category_negative_anchors.items():
            self._cat_negative_emb[cat] = enc(neg_anchors)
        print("Model loaded.")

    def _ensure_model(self):
        if self._model is None:
            self._load_model()

    def _tokenize_sentences(self, text: str) -> list:
        if not text or not text.strip():
            return []
        return [s.strip() for s in nltk.sent_tokenize(text) if len(s.strip()) >= 15]

    def _net_score(self, emb: np.ndarray, cat: str) -> float:
        """Contrastive score: max positive sim − alpha × max negative sim.

        The negative term takes the max over BOTH the global negative anchors
        and the category-specific negative anchors, so per-category
        false-positive patterns (e.g. the crowd "popular spot" descriptors)
        are also penalised. Previously _cat_negative_emb was computed but
        never applied in scoring.
        """
        pos = float(np.dot(self._anchor_emb[cat], emb).max())
        neg = float(np.dot(self._negative_emb,    emb).max())
        cat_neg_emb = self._cat_negative_emb.get(cat)
        if cat_neg_emb is not None and len(cat_neg_emb) > 0:
            cat_neg = float(np.dot(cat_neg_emb, emb).max())
            neg = max(neg, cat_neg)
        return pos - NEGATIVE_ALPHA * neg

    def tag_reviews_batch(self, reviews: list) -> list:
        self._ensure_model()

        # Tokenise all reviews → flat sentence list
        all_sents = []
        sent_map  = []
        for idx, review in enumerate(reviews):
            text  = (review.get("text_display") or review.get("text") or "")
            title = (review.get("title") or "").strip()
            if title:
                text = title + ". " + text
            for s in self._tokenize_sentences(text):
                all_sents.append(s)
                sent_map.append(idx)

        n = len(reviews)
        empty_result = [
            {**r,
             "analysis_tags":      {c: False for c in CATEGORIES},
             "analysis_scores":    {c: 0.0   for c in CATEGORIES},
             "analysis_sentences": {c: []    for c in CATEGORIES}}
            for r in reviews
        ]
        if not all_sents:
            return empty_result

        # Encode all sentences in one call
        embs = self._model.encode(
            all_sents, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )  # (n_sents, dim)

        # Contrastive score per sentence per category
        rev_scores = [{c: [] for c in CATEGORIES} for _ in range(n)]
        rev_sents  = [{c: [] for c in CATEGORIES} for _ in range(n)]

        for i, rev_idx in enumerate(sent_map):
            emb  = embs[i]
            sent = all_sents[i]
            for cat in CATEGORIES:
                score = self._net_score(emb, cat)
                rev_scores[rev_idx][cat].append(score)
                if score >= self.score_thresholds[cat]:
                    rev_sents[rev_idx][cat].append(sent)

        # Build enriched output
        enriched = []
        for idx, review in enumerate(reviews):
            r = dict(review)
            tags = {}; scores = {}; sents = {}
            for cat in CATEGORIES:
                cat_scores = rev_scores[idx][cat]
                max_score  = round(max(cat_scores), 4) if cat_scores else 0.0
                matched    = rev_sents[idx][cat]
                tags[cat]   = len(matched) > 0
                scores[cat] = max_score
                sents[cat]  = matched
            r["analysis_tags"]      = tags
            r["analysis_scores"]    = scores
            r["analysis_sentences"] = sents
            enriched.append(r)

        return enriched
