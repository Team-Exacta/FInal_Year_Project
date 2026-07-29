"""SpaCy-based span extractor for best_time / crowd_level / cost_level sentences.

For each tagged sentence, extracts the key actionable phrase using:
  - SpaCy NER   : TIME and MONEY entities
  - SpaCy Matcher: custom phrase patterns
  - Dependency   : imperative/advisory verb subtrees
"""

import re
import spacy
from spacy.matcher import PhraseMatcher, Matcher

_TIME_PHRASES = [
    "early morning", "early in the morning", "early in morning",
    "in the morning", "morning hours",
    "mid morning", "late morning",
    "in the afternoon", "mid day", "midday", "noon",
    "in the evening", "late afternoon", "at sunset", "at sunrise",
    "at night", "after dark",
    "rainy season", "dry season", "monsoon season", "peak season",
    "off season", "off-season", "shoulder season",
    "dry months", "wet months",
    "weekday", "weekdays", "weekend", "weekends",
    "public holiday", "public holidays", "poya day",
]

_CROWD_PHRASES = [
    "very crowded", "extremely crowded", "too crowded", "quite crowded",
    "not crowded", "less crowded", "not too crowded",
    "very busy", "extremely busy", "not busy", "less busy",
    "packed with tourists", "packed with people",
    "full of tourists", "full of people",
    "lots of tourists", "lot of tourists", "many tourists",
    "few visitors", "few people", "no one", "no other people",
    "had the place to ourselves", "had it to ourselves",
    "long queue", "long queues", "long wait",
    "overrun with tourists",
]

_COST_PHRASES = [
    "entry fee", "entrance fee", "admission fee", "gate fee",
    "conservation fee", "park fee", "ticket price",
    "free entry", "free admission", "free of charge", "no entry fee",
    "no charge", "no admission fee",
    "value for money", "worth it", "worth every rupee",
]

_AVOID_VERBS = {"avoid", "skip", "miss"}
_ACTION_VERBS = {"go", "visit", "come", "arrive", "start", "get", "head",
                 "leave", "enter", "climb", "hike", "trek", "walk"}


class SpanExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self._build_matchers()

    def _build_matchers(self):
        self._phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._phrase_matcher.add("TIME_PHRASE",
            [self.nlp.make_doc(p) for p in _TIME_PHRASES])
        self._phrase_matcher.add("CROWD_PHRASE",
            [self.nlp.make_doc(p) for p in _CROWD_PHRASES])
        self._phrase_matcher.add("COST_PHRASE",
            [self.nlp.make_doc(p) for p in _COST_PHRASES])

        # Money pattern: [optional currency] + number + [optional currency]
        self._money_matcher = Matcher(self.nlp.vocab)
        self._money_matcher.add("MONEY_LKR", [
            [{"LOWER": {"IN": ["rs", "rs.", "lkr", "rupees", "rupee"]}},
             {"IS_DIGIT": True}],
            [{"IS_DIGIT": True},
             {"LOWER": {"IN": ["rs", "rs.", "lkr", "rupees", "rupee"]}}],
            [{"LOWER": {"IN": ["rs", "rs.", "lkr", "rupees", "rupee"]}},
             {"IS_PUNCT": True, "OP": "?"},
             {"IS_DIGIT": True}],
        ])

    def _phrase_matches(self, doc, match_id_str: str) -> list:
        out = []
        for match_id, start, end in self._phrase_matcher(doc):
            if self.nlp.vocab.strings[match_id] == match_id_str:
                out.append(doc[start:end].text)
        return out

    def _ner_spans(self, doc, label: str) -> list:
        return [ent.text for ent in doc.ents if ent.label_ == label]

    def _money_spans(self, doc) -> list:
        spans = []
        for _, start, end in self._money_matcher(doc):
            spans.append(doc[start:end].text)
        return spans

    def _imperative_phrases(self, doc) -> list:
        """Extract ROOT verb + its right subtree for imperative sentences."""
        phrases = []
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Check for no subject (imperative) or modal construction
                has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children)
                is_modal = any(c.dep_ == "aux" and c.lower_ in
                               ("should", "must", "need", "recommend", "suggest", "advise")
                               for c in token.children)
                if not has_subj or is_modal:
                    subtree = [t.text for t in token.subtree]
                    phrase = " ".join(subtree).strip()
                    if len(phrase.split()) <= 12:
                        phrases.append(phrase)
        return phrases

    def _avoid_objects(self, doc) -> list:
        """Extract what is being avoided: 'avoid crowds', 'avoid weekends'."""
        objects = []
        for token in doc:
            if token.lower_ in _AVOID_VERBS:
                for child in token.children:
                    if child.dep_ in ("dobj", "pobj", "nmod"):
                        obj_text = " ".join(t.text for t in child.subtree).strip()
                        objects.append(obj_text)
        return objects

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    def extract_best_time(self, sentences: list) -> dict:
        time_spans = []
        avoid = []
        action = None

        for sent in sentences:
            doc = self.nlp(sent)

            # Phrase matches
            time_spans += self._phrase_matches(doc, "TIME_PHRASE")
            # NER TIME entities
            time_spans += [e for e in self._ner_spans(doc, "TIME")
                           if len(e.split()) <= 5]
            # Avoid objects
            avoid += self._avoid_objects(doc)
            # Root action verb
            for token in doc:
                if token.dep_ == "ROOT" and token.lower_ in _ACTION_VERBS:
                    action = token.lower_

        return {
            "time_spans": list(dict.fromkeys(time_spans)),  # deduplicate, preserve order
            "action":     action,
            "avoid":      list(dict.fromkeys(avoid)),
        }

    def extract_crowd(self, sentences: list) -> dict:
        crowd_spans = []
        crowd_when  = []

        for sent in sentences:
            doc = self.nlp(sent)
            crowd_spans += self._phrase_matches(doc, "CROWD_PHRASE")

            # Adjective + intensifier: "very crowded", "not busy"
            for token in doc:
                if token.lower_ in {"crowded", "busy", "packed", "overrun",
                                    "quiet", "empty", "deserted", "peaceful",
                                    "calm", "serene"}:
                    # grab left modifiers (intensifier / negation)
                    mods = [c.text for c in token.lefts
                            if c.dep_ in ("advmod", "neg", "amod")]
                    phrase = " ".join(mods + [token.text]).strip()
                    crowd_spans.append(phrase)

            # When context
            crowd_when += self._phrase_matches(doc, "TIME_PHRASE")
            crowd_when += [e for e in self._ner_spans(doc, "TIME")
                           if len(e.split()) <= 4]

        return {
            "crowd_spans": list(dict.fromkeys(crowd_spans)),
            "crowd_when":  list(dict.fromkeys(crowd_when)),
        }

    def extract_cost(self, sentences: list) -> dict:
        price_spans = []
        evaluation  = None
        fee_type    = None

        _eval_words = {
            "free": "free", "cheap": "cheap", "inexpensive": "cheap",
            "affordable": "affordable", "reasonable": "reasonable",
            "expensive": "expensive", "pricey": "expensive",
            "overpriced": "overpriced", "worth": "worth_it",
        }

        for sent in sentences:
            doc = self.nlp(sent)

            # Money entities + custom LKR pattern
            price_spans += self._ner_spans(doc, "MONEY")
            price_spans += self._money_spans(doc)
            # Cost phrases (entry fee, free entry, etc.)
            price_spans += self._phrase_matches(doc, "COST_PHRASE")

            # Evaluation adjective
            for token in doc:
                if token.lower_ in _eval_words and evaluation is None:
                    evaluation = _eval_words[token.lower_]

            # Fee type
            if fee_type is None:
                text_lower = sent.lower()
                if any(p in text_lower for p in
                       ["conservation fee", "park fee", "wildlife"]):
                    fee_type = "CONSERVATION_FEE"
                elif any(p in text_lower for p in
                         ["entry fee", "entrance fee", "admission", "gate fee",
                          "ticket price", "ticket cost"]):
                    fee_type = "ENTRY_FEE"

        return {
            "price_spans": list(dict.fromkeys(price_spans)),
            "evaluation":  evaluation,
            "fee_type":    fee_type,
        }

    def extract_review(self, review: dict) -> dict:
        """Extract spans from all 3 categories for a single review."""
        sents  = review.get("analysis_sentences", {})
        tags   = review.get("analysis_tags", {})

        spans = {}
        for cat, method in [
            ("best_time",   self.extract_best_time),
            ("crowd_level", self.extract_crowd),
            ("cost_level",  self.extract_cost),
        ]:
            if tags.get(cat):
                spans[cat] = method(sents.get(cat, []))
            else:
                spans[cat] = {}

        return spans
