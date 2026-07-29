import csv, json, re, html, os
from collections import defaultdict, Counter

INPUT = '/mnt/data/aggregated_cleaned_reviews(5).csv'
OUT_JSONL = '/mnt/data/review_activity_triples.jsonl'
OUT_JSON = '/mnt/data/review_activity_triples.json'
OUT_CSV = '/mnt/data/review_activity_triples.csv'
OUT_SUMMARY = '/mnt/data/activity_summary_by_place.csv'
OUT_EMPTY = '/mnt/data/reviews_with_no_activity.csv'
OUT_STATS = '/mnt/data/activity_extraction_stats.json'

ACTIVITY_SPECS = [
    ("White water rafting", ["white water", "white-water"], [r"\bwhite[-\s]?water\s+rafting\b"]),
    ("Rafting", ["raft"], [r"\braft(?:ing|ed)?\b"]),
    ("Scuba diving", ["scuba"], [r"\bscuba\s+div(?:e|ing|ed)?\b"]),
    ("Diving", ["dive", "diving", "dived"], [r"\bdiv(?:e|ing|ed|es)\b"]),
    ("Sea bathing", ["sea bath", "bathing"], [r"\bsea\s+bath(?:ing)?\b", r"\bbath(?:e|ing|ed)?\s+in\s+the\s+sea\b"]),
    ("Swimming", ["swim", "swam", "dip"], [r"\bswim(?:ming|s|med)?\b", r"\bswam\b", r"\btake\s+a\s+dip\b"]),
    ("Kite surfing", ["kite", "kitesurf"], [r"\bkite[-\s]?surf(?:ing|ed)?\b", r"\bkitesurf(?:ing|ed)?\b"]),
    ("Wind surfing", ["wind surf", "windsurf"], [r"\bwind[-\s]?surf(?:ing|ed)?\b", r"\bwindsurf(?:ing|ed)?\b"]),
    ("Surfing", ["surf"], [r"\bsurf(?:ing|ed)?\b", r"\bsurf\s+lessons?\b"]),
    ("Body boarding", ["body board", "bodyboard"], [r"\bbody[-\s]?boarding\b", r"\bbody\s+board(?:ing)?\b"]),
    ("Paddle boarding", ["paddle", "sup"], [r"\bpaddle[-\s]?boarding\b", r"\bstand[-\s]?up\s+paddle(?:\s+board(?:ing)?)?\b", r"\bSUP\b"]),
    ("Snorkelling", ["snorkel"], [r"\bsnorkel(?:ling|ing|ed)?\b", r"\bsnorkels?\b"]),
    ("Sunbathing", ["sunbath", "tan", "lie in the sun"], [r"\bsunbath(?:e|ing|ed)?\b", r"\btan(?:ning|ned)?\b", r"\blie\s+in\s+the\s+sun\b"]),
    ("Beach walking", ["beach", "stroll"], [r"\bwalk(?:ed|ing)?\s+(?:along|on|by|down|around)\s+(?:the\s+)?beach\b", r"\bbeach\s+walk(?:ing)?\b", r"\bstroll(?:ed|ing)?\s+(?:along|on)\s+(?:the\s+)?beach\b"]),
    ("Beach relaxing", ["beach", "relax", "chill"], [r"\brelax(?:ed|ing)?\s+(?:on|at|by)\s+(?:the\s+)?beach\b", r"\bbeach\s+relax(?:ing|ation)?\b", r"\bchill(?:ed|ing)?\s+(?:on|at|by)\s+(?:the\s+)?beach\b"]),
    ("Lagoon visit", ["lagoon"], [r"\bvisit(?:ed|ing)?\s+(?:the\s+)?lagoon\b", r"\blagoon\s+(?:visit|tour|trip)\b", r"\b(?:in|around)\s+(?:the\s+)?lagoon\b"]),
    ("Boat safari", ["boat safari", "safari boat"], [r"\bboat\s+safari\b", r"\bsafari\s+boat\b"]),
    ("Boat ride", ["boat", "cruise"], [r"\bboat\s+(?:ride|trip|tour|cruise|journey)\b", r"\btook\s+(?:a\s+)?boat\b", r"\bby\s+boat\b", r"\bboating\b", r"\bcruise\b"]),
    ("Canoeing", ["canoe"], [r"\bcanoe(?:ing|d)?\b"]),
    ("Kayaking", ["kayak"], [r"\bkayak(?:ing|ed)?\b"]),
    ("Waterfall bathing", ["waterfall", "falls"], [r"\bbath(?:e|ing|ed)?\s+(?:at|in|under)\s+(?:the\s+)?waterfalls?\b", r"\bswim(?:ming)?\s+(?:at|in|under)\s+(?:the\s+)?waterfalls?\b"]),
    ("Waterfall visit", ["waterfall", "falls"], [r"\bvisit(?:ed|ing)?\s+(?:the\s+)?(?:[\w'’.-]+\s+)?waterfalls?\b", r"\bwaterfalls?\s+(?:visit|trip|tour)\b", r"\bwent\s+to\s+(?:the\s+)?waterfalls?\b", r"\bsee\s+(?:the\s+)?waterfalls?\b"]),
    ("Fishing", ["fishing", "fished", "fish trip"], [r"\bfishing\b", r"\bfished\b", r"\bgo(?:ing)?\s+fishing\b", r"\bfish(?:ing)?\s+(?:trip|tour)\b"]),
    ("Mountain biking", ["mountain bik"], [r"\bmountain\s+bik(?:e|ing|ed)\b"]),
    ("Cycling", ["cycl", "bike", "biking"], [r"\bcycl(?:e|ing|ed)\b", r"\bbike\s+ride\b", r"\bbiking\b"]),
    ("Rock climbing", ["rock climb"], [r"\brock\s+climb(?:ing|ed)?\b"]),
    ("Mountain climbing", ["mountain", "climb"], [r"\bmountain\s+climb(?:ing|ed)?\b", r"\bclimb(?:ed|ing)?\s+(?:up\s+)?(?:the\s+)?mountain\b"]),
    ("Hiking", ["hike", "hiking", "hiked"], [r"\bhik(?:e|ing|ed|es)\b"]),
    ("Trekking", ["trek"], [r"\btrek(?:king|ked|s)?\b"]),
    ("Nature walk", ["nature walk"], [r"\bnature\s+walk(?:ing)?\b"]),
    ("Walking trail", ["trail"], [r"\bwalking\s+trail\b", r"\btrail\s+walk(?:ing)?\b", r"\bwalk(?:ed|ing)?\s+(?:the\s+)?trail\b"]),
    ("Ziplining", ["zip"], [r"\bzip[-\s]?lin(?:e|ing|ed)\b"]),
    ("Abseiling", ["abseil", "rappel"], [r"\babseil(?:ing|ed)?\b", r"\brappel(?:ling|ing|ed)?\b"]),
    ("Camping", ["camp"], [r"\bcamp(?:ing|ed)?\b", r"\bcamp\s+site\b"]),
    ("Caving", ["cave", "caving"], [r"\bcaving\b", r"\bcave\s+explor(?:e|ing|ation)\b"]),
    ("Forest walk", ["forest", "rainforest"], [r"\bforest\s+walk(?:ing)?\b", r"\bwalk(?:ed|ing)?\s+(?:through|in|inside)\s+(?:the\s+)?(?:rain\s*)?forest\b"]),
    ("Jeep safari", ["jeep", "4x4"], [r"\bjeep\s+safari\b", r"\bsafari\s+jeep\b", r"\b4x4\s+safari\b"]),
    ("Safari", ["safari", "game drive"], [r"\bsafari(?:s)?\b", r"\bgame\s+drive\b"]),
    ("Whale watching", ["whale"], [r"\bwhale\s*watch(?:ing)?\b", r"\bwatch(?:ed|ing)?\s+whales\b", r"\bsee\s+whales\b", r"\bsaw\s+whales\b"]),
    ("Dolphin watching", ["dolphin"], [r"\bdolphin\s*watch(?:ing)?\b", r"\bwatch(?:ed|ing)?\s+dolphins\b", r"\bsee\s+dolphins\b", r"\bsaw\s+dolphins\b"]),
    ("Elephant watching", ["elephant"], [r"\belephant\s*watch(?:ing)?\b", r"\bwatch(?:ed|ing)?\s+elephants\b", r"\bsee\s+elephants\b", r"\bsaw\s+elephants\b", r"\belephants?\s+(?:watching|seen|spotting)\b"]),
    ("Leopard watching", ["leopard"], [r"\bleopard\s*watch(?:ing)?\b", r"\bwatch(?:ed|ing)?\s+leopards\b", r"\bsee\s+leopards\b", r"\bsaw\s+leopards\b", r"\bleopards?\s+(?:watching|seen|spotting)\b"]),
    ("Turtle watching", ["turtle"], [r"\bturtle\s*watch(?:ing)?\b", r"\bwatch(?:ed|ing)?\s+turtles\b", r"\bsee\s+turtles\b", r"\bsaw\s+turtles\b", r"\bturtles?\s+(?:watching|seen|spotting)\b"]),
    ("Bird watching", ["bird"], [r"\bbird\s*watch(?:ing)?\b", r"\bwatch(?:ed|ing)?\s+birds\b", r"\bbird\s+spotting\b", r"\bsaw\s+birds\b"]),
    ("Wildlife watching", ["wildlife", "animal", "spotting"], [r"\bwildlife\s*watch(?:ing)?\b", r"\bwatch(?:ed|ing)?\s+wildlife\b", r"\bsee\s+wildlife\b", r"\bsaw\s+wildlife\b", r"\banimal\s+spotting\b", r"\bspot(?:ted|ting)?\s+(?:wildlife|animals)\b"]),
    ("Scenic train ride", ["train", "scenic"], [r"\bscenic\s+train\s+(?:ride|journey|trip)\b", r"\btrain\s+(?:ride|journey|trip).{0,50}\bscenic\b", r"\bscenic.{0,50}\btrain\b"]),
    ("Train ride", ["train"], [r"\btrain\s+(?:ride|journey|trip)\b", r"\btook\s+(?:the\s+)?train\b", r"\bby\s+train\b"]),
    ("Scenic drive", ["drive", "scenic"], [r"\bscenic\s+drive\b", r"\bdrive\s+(?:was\s+)?scenic\b", r"\bdrive\s+(?:through|around|along)\b"]),
    ("Sunset watching", ["sunset"], [r"\bsunset(?:s)?\b", r"\bwatch(?:ed|ing)?\s+(?:the\s+)?sunset\b"]),
    ("Sunrise watching", ["sunrise"], [r"\bsunrise(?:s)?\b", r"\bwatch(?:ed|ing)?\s+(?:the\s+)?sunrise\b"]),
    ("Viewpoint visit", ["viewpoint", "view point", "lookout", "observation"], [r"\bviewpoint\b", r"\bview\s+point\b", r"\blookout\b", r"\bobservation\s+(?:deck|point)\b"]),
    ("Photography", ["photo", "picture", "photograph"], [r"\bphotograph(?:y|s|ed|ing)?\b", r"\bphotos?\b", r"\bpictures?\b", r"\btake\s+(?:some\s+)?(?:photos|pictures)\b", r"\bphoto\s+op(?:portunity)?\b"]),
    ("Sightseeing", ["sightseeing", "sight-seeing", "sights"], [r"\bsightseeing\b", r"\bsight[-\s]?seeing\b", r"\bsee\s+(?:the\s+)?sights\b"]),
    ("Temple visit", ["temple", "vihara", "viharaya", "kovil"], [r"\bvisit(?:ed|ing)?\s+(?:the\s+)?(?:[\w'’.-]+\s+)?(?:temple|vihara|viharaya|kovil)\b", r"\btemple\s+visit\b", r"\bwent\s+to\s+(?:the\s+)?(?:[\w'’.-]+\s+)?temple\b"]),
    ("Pilgrimage", ["pilgrim"], [r"\bpilgrimage\b", r"\bpilgrims?\b"]),
    ("Religious visit", ["church", "mosque", "shrine", "dagoba", "stupa", "buddha"], [r"\breligious\s+visit\b", r"\bvisit(?:ed|ing)?\s+(?:the\s+)?(?:church|mosque|kovil|shrine|dagoba|stupa|buddha\s+statue)\b"]),
    ("Ancient city visit", ["ancient city"], [r"\bancient\s+city\s+(?:visit|tour)\b", r"\bvisit(?:ed|ing)?\s+(?:the\s+)?ancient\s+city\b"]),
    ("Ruins exploration", ["ruin", "ruins"], [r"\bruin(?:s)?\s+(?:exploration|tour|visit)\b", r"\bexplor(?:e|ed|ing)\s+(?:the\s+)?ruins\b", r"\bwalk(?:ed|ing)?\s+(?:through|around)\s+(?:the\s+)?ruins\b"]),
    ("Heritage exploration", ["heritage", "historical", "history"], [r"\bheritage\s+(?:exploration|tour|visit|site)\b", r"\bexplor(?:e|ed|ing)\s+(?:the\s+)?heritage\b", r"\bhistorical\s+(?:tour|visit|site)\b"]),
    ("Fort visit", ["fort"], [r"\bvisit(?:ed|ing)?\s+(?:the\s+)?(?:[\w'’.-]+\s+)?fort\b", r"\bfort\s+visit\b", r"\bwent\s+to\s+(?:the\s+)?(?:[\w'’.-]+\s+)?fort\b"]),
    ("Museum visit", ["museum"], [r"\bvisit(?:ed|ing)?\s+(?:the\s+)?(?:[\w'’.-]+\s+)?museum\b", r"\bmuseum\s+visit\b"]),
    ("Cultural show", ["cultural show", "dance show", "traditional show"], [r"\bcultural\s+show\b", r"\bdance\s+show\b", r"\btraditional\s+show\b"]),
    ("Village tour", ["village"], [r"\bvillage\s+(?:tour|walk|visit)\b", r"\bvisit(?:ed|ing)?\s+(?:a\s+)?village\b"]),
    ("Tea tasting", ["tea", "taste"], [r"\btea\s+tast(?:e|ing)\b", r"\btast(?:e|ed|ing)\s+(?:the\s+)?tea\b"]),
    ("Tea factory visit", ["tea factory"], [r"\btea\s+factory\s+(?:visit|tour)\b", r"\bvisit(?:ed|ing)?\s+(?:the\s+)?tea\s+factory\b"]),
    ("Tea plantation visit", ["tea plantation", "tea estate"], [r"\btea\s+plantation\s+(?:visit|tour)\b", r"\bvisit(?:ed|ing)?\s+(?:the\s+)?tea\s+(?:plantation|estate)\b", r"\bwalk(?:ed|ing)?\s+(?:through|around)\s+(?:the\s+)?tea\s+(?:plantation|estate)\b"]),
    ("Street food tasting", ["street food"], [r"\bstreet\s+food\b"]),
    ("Seafood tasting", ["seafood", "sea food"], [r"\bseafood\b", r"\bsea\s+food\b"]),
    ("Local food tasting", ["local food", "food tasting", "try food", "taste food"], [r"\blocal\s+food\b", r"\btast(?:e|ed|ing)\s+(?:the\s+)?(?:local\s+)?food\b", r"\btry\s+(?:the\s+)?(?:local\s+)?food\b", r"\bfood\s+tasting\b"]),
    ("Market visit", ["market"], [r"\bvisit(?:ed|ing)?\s+(?:the\s+)?(?:local\s+)?market\b", r"\bmarket\s+(?:visit|tour)\b", r"\bwalk(?:ed|ing)?\s+(?:through|around)\s+(?:the\s+)?market\b"]),
    ("Souvenir shopping", ["souvenir"], [r"\bsouvenir\s+shopping\b", r"\bbuy(?:ing|\s+some)?\s+souvenirs?\b", r"\bsouvenirs?\b"]),
    ("Shopping", ["shopping", "shop", "shops"], [r"\bshopping\b", r"\bshop(?:ped|ping)?\b", r"\bshops?\b"]),
    ("Cooking class", ["cooking class", "cookery"], [r"\bcooking\s+class\b", r"\bcookery\s+class\b"]),
    ("Ayurveda wellness", ["ayurveda", "ayurvedic"], [r"\bayurved(?:a|ic)\b"]),
    ("Spa treatment", ["spa", "massage"], [r"\bspa\b", r"\bmassage\b"]),
    ("Meditation", ["meditat"], [r"\bmeditat(?:e|ion|ing|ed)\b"]),
    ("Yoga", ["yoga"], [r"\byoga\b"]),
    ("Picnic", ["picnic"], [r"\bpicnic\b"]),
    ("Relaxing", ["relax", "chill", "unwind"], [r"\brelax(?:ing|ed|ation)?\b", r"\bchill(?:ed|ing)?\b", r"\bunwind\b"]),
]

COMPILED = [(a, kws, [re.compile(p, re.I) for p in pats]) for a,kws,pats in ACTIVITY_SPECS]

VISIT_ACTION_RE = re.compile(r"\b(visit(?:ed|ing)?|went|go(?:ing)?|tour(?:ed|ing)?|walk(?:ed|ing)?|explor(?:e|ed|ing)|see|saw|seen)\b", re.I)
PLACE_IMPLIED = [
    ("Temple visit", re.compile(r"\b(temple|vihara|viharaya|kovil)\b", re.I)),
    ("Religious visit", re.compile(r"\b(church|mosque|shrine|dagoba|stupa|buddha|tooth\s+relic)\b", re.I)),
    ("Fort visit", re.compile(r"\bfort\b", re.I)),
    ("Museum visit", re.compile(r"\bmuseum\b", re.I)),
    ("Waterfall visit", re.compile(r"\bfalls?|waterfall\b", re.I)),
    ("Ancient city visit", re.compile(r"\bancient\s+city|polonnaruwa|anuradhapura\b", re.I)),
    ("Ruins exploration", re.compile(r"\bruins?|archaeological\b", re.I)),
    ("Tea factory visit", re.compile(r"\btea\s+factory\b", re.I)),
    ("Tea plantation visit", re.compile(r"\btea\s+(plantation|estate)\b", re.I)),
]

POSITIVE = set("amazing awesome beautiful best excellent great good fantastic wonderful lovely nice perfect enjoyable enjoyed recommend recommended worth worthwhile stunning spectacular breathtaking memorable incredible impressive interesting pleasant peaceful relaxing calm serene safe clear clean fun exciting authentic informative friendly magnificent gorgeous superb magical cool mesmerizing unforgettable scenic picturesque impressive".split())
NEGATIVE = set("bad terrible awful poor disappointing disappointed dangerous unsafe risky crowded overcrowded dirty polluted boring expensive overpriced waste difficult hard tough tiring exhausting slippery steep closed unavailable worst hassle noisy rough scam unpleasant avoid annoying harassment extortionate dubious problem problems".split())
POS_PHRASES = ["highly recommend", "must visit", "worth a visit", "worth visiting", "well worth", "good for", "great for", "perfect for", "best place", "loved", "enjoyed"]
NEG_PHRASES = ["not worth", "not recommend", "would not recommend", "too crowded", "too expensive", "too difficult", "not safe", "not allowed", "couldn't", "could not", "unable to", "waste of", "dangerous", "unsafe"]
NEGATORS = {"not","no","never","without","hardly"}

SPECIFIC_SKIP = {
    ("Diving", "scuba"),
    ("Rafting", "white water"),
    ("Rafting", "white-water"),
    ("Surfing", "kitesurf"),
    ("Surfing", "kite surf"),
    ("Surfing", "windsurf"),
    ("Surfing", "wind surf"),
    ("Boat ride", "boat safari"),
    ("Safari", "jeep safari"),
    ("Safari", "4x4 safari"),
    ("Train ride", "scenic train"),
    ("Cycling", "mountain biking"),
    ("Shopping", "souvenir"),
    ("Relaxing", "beach relaxing"),
}

def clean_text(s):
    return re.sub(r"\s+", " ", html.unescape(s or "")).strip()

def split_sentences(text):
    text = clean_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]

def sentiment(sentence):
    s = sentence.lower()
    pos = sum(2 for p in POS_PHRASES if p in s)
    neg = sum(2 for p in NEG_PHRASES if p in s)
    words = re.findall(r"[a-z']+", s)
    for i,w in enumerate(words):
        base = w.strip("'")
        negated = bool(NEGATORS.intersection(words[max(0, i-3):i]))
        if base in POSITIVE:
            if negated: neg += 1
            else: pos += 1
        if base in NEGATIVE:
            if negated: pos += 1
            else: neg += 1
    if pos > neg:
        return "positive", pos, neg
    if neg > pos:
        return "negative", pos, neg
    return "neutral", pos, neg

def conf(activity, sentence, sentiment_label, explicit=True, pos=0, neg=0):
    val = 0.84 if explicit else 0.70
    if sentiment_label != 'neutral':
        val += min(0.06, max(pos, neg)*0.02)
    if len(sentence) <= 260:
        val += 0.03
    if activity in ("Sunset watching", "Sunrise watching", "Photography", "Relaxing"):
        val -= 0.03
    return round(max(0.55, min(0.96, val)), 2)

def should_skip(act, low):
    for a, kw in SPECIFIC_SKIP:
        if act == a and kw in low:
            return True
    if act == 'Shopping' and 'workshop' in low:
        return True
    return False

def extract(place, rid, title, body):
    text = clean_text(((title or '') + '. ' + (body or '')).strip())
    out = []
    seen = set()
    for sent in split_sentences(text):
        low = sent.lower()
        for act, kws, regs in COMPILED:
            if not any(k.lower() in low for k in kws):
                continue
            if should_skip(act, low):
                continue
            if (act, sent) in seen:
                continue
            found = False
            for rgx in regs:
                if rgx.search(sent):
                    found = True
                    break
            if not found:
                continue
            sent_label, ps, ns = sentiment(sent)
            out.append({
                "subject": place,
                "subject_type": "Place",
                "relation": "HAS_ACTIVITY",
                "object": act,
                "object_type": "Activity",
                "sentiment": sent_label,
                "evidence_id": rid,
                "evidence": sent,
                "confidence": conf(act, sent, sent_label, True, ps, ns)
            })
            seen.add((act, sent))
        if VISIT_ACTION_RE.search(sent):
            context = f"{place}. {sent}"
            for act, rgx in PLACE_IMPLIED:
                if (act, sent) in seen:
                    continue
                if rgx.search(context):
                    sent_label, ps, ns = sentiment(sent)
                    out.append({
                        "subject": place,
                        "subject_type": "Place",
                        "relation": "HAS_ACTIVITY",
                        "object": act,
                        "object_type": "Activity",
                        "sentiment": sent_label,
                        "evidence_id": rid,
                        "evidence": sent,
                        "confidence": conf(act, sent, sent_label, False, ps, ns)
                    })
                    seen.add((act, sent))
    return out

