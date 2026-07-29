"""Maps extracted spans to structured normalized values.

TIME_OF_DAY  : EARLY_MORNING | MID_MORNING | AFTERNOON | EVENING | NIGHT
SEASON       : DRY_SEASON | MONSOON | SHOULDER
DAY_TYPE     : WEEKDAY | WEEKEND | PUBLIC_HOLIDAY
CROWD_LEVEL  : 1 (EMPTY) … 5 (PACKED)
COST_LEVEL   : FREE | LOW | MODERATE | HIGH | VERY_HIGH
"""

import re


def _clean(parts) -> str:
    """Join, lowercase, and normalise smart quotes so regexes with plain
    apostrophes (don't / wasn't) still match text containing '’'."""
    return (" ".join(parts).lower()
            .replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"'))


# ---------------------------------------------------------------------------
# TIME normalization
# ---------------------------------------------------------------------------

# Time-of-day patterns, ordered by SPECIFICITY (not by clock time). "late
# afternoon" must be listed under EVENING and checked before plain "afternoon",
# otherwise the bare "afternoon" keyword grabs it first (first-match-wins).
_TIME_PATTERNS = [
    # NB: bare "late afternoon" is intentionally NOT here — it is ambiguous
    # ("late afternoon stroll" = afternoon, "late afternoon for sunset" = evening).
    # The sunset/evening/dusk cue decides, so it resolves from context.
    ("EVENING", [
        r"\bevening\b", r"\bdusk\b", r"\bsunset\b", r"\bnightfall\b",
        r"after \d{1,2}\s*(?:pm|o'?clock)", r"\b(?:5|6|7)\s*pm\b",
    ]),
    ("EARLY_MORNING", [
        r"early (?:in the )?mornings?", r"\bsunrise\b", r"\bsun rise\b",
        r"\bdawn\b", r"\bdaybreak\b", r"\bbreak of day\b", r"first thing",
        r"before (?:9|8|7|6|nine|eight|seven|six)\b",
        r"\b(?:1|2|3|4|5|6|7|8)\s*am\b", r"\b(?:4|5|6)\s*o'?clock\b",
        r"early part of (?:the )?day", r"earlier in the day", r"\bearly hours?\b",
        r"(?:go|climb|visit|arrive|start|leave|set off|come|walk|went|hike|reach|get there)"
        r"\s+\w{0,8}?\s*earl(?:y|ier)",
        r"\bearlier\b",
    ]),
    ("MID_MORNING", [
        r"mid[- ]?morning", r"\bmorning\b", r"\b(?:9|10|11)\s*am\b",
    ]),
    ("AFTERNOON", [
        r"\bafternoon\b", r"\bmidday\b", r"\bmid[- ]?day\b", r"\bnoon\b",
        r"\b(?:12|1|2|3)\s*pm\b",
    ]),
    ("NIGHT", [
        r"\bnight\b", r"after dark", r"night ?time", r"\b(?:8|9|10|11)\s*pm\b",
    ]),
]
_TIME_ORDER = [label for label, _ in _TIME_PATTERNS]
# Tie-break when a sentence mentions several times (e.g. "morning or evening"):
# annotators consistently record the earlier/primary slot, so prefer morning,
# then evening over afternoon (keeps "late afternoon ... sunset" = EVENING).
_TIME_PREFERENCE = ["EARLY_MORNING", "MID_MORNING", "EVENING", "AFTERNOON", "NIGHT"]

_SEASON = [
    ("DRY_SEASON", [
        r"\bdry season\b", r"\bdry months?\b",
        r"\b(december|january|february|march|april)\b",
        r"\bnorth[- ]?east monsoon\b",
    ]),
    ("MONSOON", [
        r"\b(monsoon|rainy season|wet season)\b",
        r"\b(may|june|july|august|september|october)\b",
        r"\bsouth[- ]?west monsoon\b",
    ]),
    ("SHOULDER", [
        r"\b(november|shoulder season)\b",
    ]),
]

_DAY_TYPE = [
    ("WEEKEND",        [r"\bweekend(s)?\b"]),
    ("WEEKDAY",        [r"\bweekday(s)?\b", r"\bweek day(s)?\b"]),
    # Bare "holiday" removed: in this corpus it overwhelmingly means the
    # tourist's OWN trip ("our holiday", "new year holiday week"), not a public
    # holiday — it was the main source of day_type false positives. Require an
    # explicit public-holiday term instead.
    ("PUBLIC_HOLIDAY", [r"\b(public holiday|poya day|poya|full moon poya|"
                        r"national holiday|bank holiday)\b"]),
]

# Negation / avoidance markers. A recommendation like "not suitable in the
# afternoon" or "avoid weekends" is the OPPOSITE of a positive mention, so we
# must detect it and resolve to the complement instead of the literal value.
_NEG = re.compile(
    r"\b(?:not|never|avoid|avoiding|don'?t|do not|rather than|apart from|"
    r"other than|except|skip|stay away|steer clear|no need)\b")
# Explicit avoidance cues used for day-type polarity. We flip to the complement
# only on an explicit "avoid/not/skip" — NOT on a mere description like "busy at
# the weekend" (which states a fact about the weekend rather than recommending
# against it). Human labels treat those two cases differently.
# Only EXPLICIT avoidance flips the day-type recommendation. The descriptive
# crowd words (busy / crowded / packed …) were removed: they contradicted this
# function's own documented policy ("NOT on a mere description like 'busy at the
# weekend'") and caused spurious WEEKDAY predictions.
_DAY_AVOID = re.compile(
    r"\b(?:avoid|avoiding|not|never|don'?t|do not|skip|stay away|steer clear|"
    r"rather than|worst)\w*\b")


def _negated_before(text: str, pos: int, window: int = 16) -> bool:
    """True if a negation/avoidance marker appears just before position `pos`."""
    return _NEG.search(text[max(0, pos - window):pos]) is not None


def _resolve_time_of_day(text: str):
    """Negation-aware time-of-day resolution.

    Picks the highest-specificity NON-negated time mentioned. If every time
    mentioned is negated (e.g. "do not go in the afternoon"), infers the
    complement conservatively rather than returning the avoided slot.
    """
    matched = []   # (label, position, negated)
    for label, patterns in _TIME_PATTERNS:
        for p in patterns:
            m = re.search(p, text)
            if m:
                matched.append((label, m.start(), _negated_before(text, m.start())))
                break
    if not matched:
        return None
    positive = [lab for lab, _pos, neg in matched if not neg]
    if positive:
        return sorted(positive, key=_TIME_PREFERENCE.index)[0]
    # Everything mentioned is negated → infer the complement.
    neg_labels = {lab for lab, _pos, _neg in matched}
    hot = re.search(r"\b(?:hot|heat|humid|sun|sunny|warm)\b", text) is not None
    if hot and ({"AFTERNOON", "MID_MORNING", "NIGHT"} & neg_labels):
        return "EARLY_MORNING"
    if "AFTERNOON" in neg_labels:
        return "EVENING"
    if "EVENING" in neg_labels or "NIGHT" in neg_labels:
        return "EARLY_MORNING"
    return None


def _resolve_day_type(text: str):
    """Return (recommend_day, avoid_list), applying complement logic.

    'avoid weekends' / 'weekends get crowded' → recommend WEEKDAY (+ avoid WEEKEND).
    """
    bad = _DAY_AVOID.search(text) is not None
    recommend, avoid = None, []
    if re.search(r"\bweekend", text):
        if bad:
            recommend, avoid = "WEEKDAY", avoid + ["WEEKEND"]
        else:
            recommend = "WEEKEND"
    elif re.search(r"\bweek\s?day", text):
        if bad:
            recommend, avoid = "WEEKEND", avoid + ["WEEKDAY"]
        else:
            recommend = "WEEKDAY"
    if re.search(r"\b(?:public holiday|poya|full moon|national holiday|bank holiday)", text):
        if bad:
            avoid.append("PUBLIC_HOLIDAY")
            recommend = recommend or "WEEKDAY"
        else:
            recommend = recommend or "PUBLIC_HOLIDAY"
    return recommend, sorted(set(avoid))


def normalize_best_time(spans: dict, raw_sentences: list = None) -> dict:
    parts = spans.get("time_spans", []) + spans.get("avoid", [])
    if raw_sentences:
        parts = list(raw_sentences) + parts
    text = _clean(parts)

    time_of_day = _resolve_time_of_day(text)

    season = None
    for label, patterns in _SEASON:
        if any(re.search(p, text) for p in patterns):
            season = label
            break

    months = []
    month_names = ["january","february","march","april","may","june",
                   "july","august","september","october","november","december"]
    for m in month_names:
        if re.search(rf"\b{m}\b", text):
            months.append(m.upper())

    recommend_day, avoid = _resolve_day_type(text)

    return {
        "time_of_day":   time_of_day,
        "season":        season,
        "months":        months,
        "recommend_day": recommend_day,
        "avoid":         avoid,
    }


# ---------------------------------------------------------------------------
# CROWD normalization
# ---------------------------------------------------------------------------

_CROWD_LEVELS = [
    (5, "PACKED", [
        r"\b(very |extremely |super |incredibly |absolutely )?(over ?crowded|crowded|packed|jam[- ]?packed|overrun|swarming)\b",
        r"\bimpossible to (enjoy|move|visit)\b",
        r"\boverwhelming (crowd|number of tourist|tourist)\b",
        r"\b(too much|so much|way too much) (of a )?crowd",
    ]),
    (4, "BUSY", [
        r"\b(crowded|busy|full of (tourist|people|visitor)|lot of (tourist|people))\b",
        r"\bmany (tourist|people|visitor|crowd)\b",
        r"\b(lots?|loads|plenty|a lot) of (crowd|people|tourist|visitor)",
        r"\bquite (some|a few|a lot of) (people|crowd|tourist)",
        r"\blong queue\b", r"\blong wait\b",
    ]),
    (3, "MODERATE", [
        r"\b(moderate|manageable|some tourist|somewhat busy|a (few|bit) crowd)\b",
        r"\bnot too (crowded|busy)\b",
    ]),
    (2, "QUIET", [
        r"\b(quiet|calm|peaceful|not (very |too )?crowd|not (very |too )?busy|less crowd|fewer (tourist|people))\b",
        r"\bnot (many|much) (tourist|people|crowd|visitor)\b",
    ]),
    (1, "EMPTY", [
        r"\b(empty|deserted|no one|nobody|to ourselves|had (the place|it) (all )?to (our|my)self)\b",
        r"\bno other (people|visitor|tourist)\b",
        r"\bno (queue|wait|crowd)\b",
    ]),
]

_LEVEL_LABEL = {1: "EMPTY", 2: "QUIET", 3: "MODERATE", 4: "BUSY", 5: "PACKED"}

# Negation of a crowd word ("not overly crowded", "wasn't busy", "not too packed").
# Allows up to 3 words between the negator and the crowd term (e.g. "not overly").
_CROWD_NEG = re.compile(
    r"\b(?:not|never|hardly|barely|isn'?t|wasn'?t|weren'?t|aren'?t|no)\b"
    r"(?:\s+\w+){0,3}\s+(?:crowd|crowded|busy|packed|overrun|overcrowded|touristy|full)")


def normalize_crowd(spans: dict, raw_sentences: list = None) -> dict:
    # Read the RAW sentence, not just the extracted span — the span often drops
    # the negation ("not overly crowded" gets extracted as just "crowded").
    parts = list(raw_sentences or []) + spans.get("crowd_spans", [])
    if not parts:
        return {"crowd_level": None, "crowd_label": None, "crowd_when": []}

    text = _clean(parts)
    level = label = None
    for lvl, lbl, patterns in _CROWD_LEVELS:
        if any(re.search(p, text) for p in patterns):
            level, label = lvl, lbl
            break

    # Negation: "not (overly) crowded / busy / packed" is NOT a high level.
    if level and level >= 3 and _CROWD_NEG.search(text):
        level = 2   # QUIET
        label = _LEVEL_LABEL[level]

    when_text = _clean(list(raw_sentences or []) + spans.get("crowd_when", []))
    crowd_when = []
    for lbl, patterns in _DAY_TYPE:
        if any(re.search(p, when_text) for p in patterns):
            crowd_when.append(lbl)

    return {
        "crowd_level": level,
        "crowd_label": label,
        "crowd_when":  crowd_when,
    }


# ---------------------------------------------------------------------------
# COST normalization
# ---------------------------------------------------------------------------

USD_TO_LKR = 300   # rough conversion for USD amounts

_AMOUNT_PATTERN = re.compile(
    r"(?:rs\.?\s*|lkr\s*|rupees?\s*)(\d[\d,]*)"          # g1: LKR before  (Rs 200)
    r"|(\d[\d,]*)\s*(?:rs\.?|lkr|rupees?)"               # g2: LKR after   (200 rupees)
    r"|(?:us\$\s*|\$\s*)(\d[\d,.]*)"                     # g3: USD symbol  ($100, US$ 50)
    r"|(\d[\d,.]*)\s*(?:usd|us dollars?|dollars?)",      # g4: USD word    (25 USD)
    re.IGNORECASE,
)

_COST_EVAL = [
    ("FREE",      [r"\bfree\b", r"\bno (entry |admission |entrance )?fee\b",
                   r"\bno charge\b", r"\bfree of charge\b"]),
    ("VERY_HIGH", [r"\bvery expensive\b", r"\boverpriced\b", r"\bway too (high|much|expensive)\b"]),
    ("HIGH",      [r"\bexpensive\b", r"\bpricey\b", r"\btoo (high|much|expensive)\b",
                   r"\bnot (cheap|affordable|worth)\b"]),
    ("MODERATE",  [r"\breasonable\b", r"\baffordable\b", r"\bworth (it|the|every)\b",
                   r"\bnot (expensive|pricey|too much)\b"]),
    ("LOW",       [r"\bcheap\b", r"\binexpensive\b", r"\bvery affordable\b",
                   r"\bgreat value\b", r"\bvalue for money\b",
                   r"\bnot much\b", r"\bnot a lot\b", r"\bnominal\b", r"\bminimal\b",
                   r"\b(small|little|tiny) (fee|charge|amount|entrance fee|entry fee)\b"]),
]

_FEE_TYPE = [
    ("ENTRY_FEE",        [r"\bentry fee\b", r"\bentrance fee\b", r"\badmission fee\b",
                           r"\bgate fee\b", r"\bticket (price|cost|fee)\b"]),
    ("CONSERVATION_FEE", [r"\bconservation fee\b", r"\bpark fee\b", r"\bwildlife fee\b"]),
    ("GUIDE_FEE",        [r"\bguide fee\b", r"\bguiding fee\b"]),
]


def _match_value(m):
    """Convert one _AMOUNT_PATTERN match to an integer LKR amount."""
    rs_val = m.group(1) or m.group(2)
    usd    = m.group(3) or m.group(4)
    if rs_val:
        try:
            return int(rs_val.replace(",", ""))
        except ValueError:
            return None
    if usd:
        try:
            return int(float(usd.replace(",", "")) * USD_TO_LKR)
        except ValueError:
            return None
    return None


def _parse_amount_lkr(text: str):
    """Return the relevant numeric amount in LKR.

    When a sentence lists separate local and foreigner prices ("200 rupees for
    locals and 16000 rupees for foreigners"), the foreigner price is the one that
    matters for a tourist-facing planner, so it is preferred over the local price.
    """
    amounts = []   # (value, is_foreign, is_local)
    for m in _AMOUNT_PATTERN.finditer(text):
        v = _match_value(m)
        if v is None:
            continue
        ctx = text[max(0, m.start() - 18):m.end() + 22].lower()
        is_foreign = bool(re.search(r"foreign|tourist|non[- ]?local", ctx))
        is_local   = bool(re.search(r"\blocal|resident|sri[- ]?lankan|citizen", ctx))
        amounts.append((v, is_foreign, is_local))

    if not amounts:
        return None
    foreign = [v for v, f, l in amounts if f]
    if foreign:
        return foreign[0]
    non_local = [v for v, f, l in amounts if not l]
    if non_local:
        return non_local[0]
    return amounts[0][0]


# Cost-band cutoffs (LKR). Data-driven: derived from the distribution of all
# 708 extracted amount_lkr values (>0) in output/extraction/_all_spans.csv.
# Each boundary sits on a distributional percentile rather than being guessed —
# see output/evaluation/cost_threshold_analysis.json.
#   500  -> 33rd percentile (33.3% of fees fall below)
#   1500 -> 67th percentile (67.4% below)   [was 2000; moved onto p67]
#   5000 -> 92nd percentile (92.4% below, ~p95)
# Resulting band shares: LOW 33% · MODERATE 34% · HIGH 25% · VERY_HIGH 8%.
_COST_CUTOFFS_LKR = (500, 1500, 5000)


def _amount_to_level(amount: int) -> str:
    lo, mid, hi = _COST_CUTOFFS_LKR
    if amount == 0:
        return "FREE"
    if amount < lo:
        return "LOW"
    if amount < mid:
        return "MODERATE"
    if amount < hi:
        return "HIGH"
    return "VERY_HIGH"


def normalize_cost(spans: dict, raw_sentences: list = None) -> dict:
    # Read the RAW sentence so negation ("not cheap") and local/foreigner context
    # ("16000 for foreigners") survive — the extracted span often drops them.
    raw_text      = _clean(raw_sentences or [])
    price_text    = _clean(spans.get("price_spans", []))
    eval_text     = _clean([spans.get("evaluation") or ""])
    combined_text = " ".join([raw_text, price_text, eval_text]).strip()

    # Amount-based level: from numeric LKR/USD values (raw first — it has the
    # foreigner/local context needed to pick the right price).
    amount       = _parse_amount_lkr(raw_text or price_text)
    amount_level = _amount_to_level(amount) if amount is not None else None

    # Sentiment-based level: from opinion adjectives only
    sentiment_level = None
    for level_label, patterns in _COST_EVAL:
        if any(re.search(p, combined_text) for p in patterns):
            sentiment_level = level_label
            break

    # Primary: prefer amount-based (objective) over sentiment (subjective)
    cost_level = amount_level if amount_level is not None else sentiment_level

    return {
        "cost_level":      cost_level,
        "amount_level":    amount_level,
        "sentiment_level": sentiment_level,
        "amount_lkr":      amount,
        "evaluation":      spans.get("evaluation"),
        "fee_type":        spans.get("fee_type"),
    }
