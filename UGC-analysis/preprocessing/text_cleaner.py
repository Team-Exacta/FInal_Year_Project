"""Clean and normalize review text."""

import re
import unicodedata
import html


# Emoji removal pattern (covers most emoji Unicode ranges)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001f926-\U0001f937"  # supplemental
    "\U00010000-\U0010ffff"  # supplementary
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)

# URL pattern
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Excessive punctuation (3+ of the same)
EXCESSIVE_PUNCT_PATTERN = re.compile(r"([!?.]){3,}")

# Multiple spaces
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def remove_emojis(text):
    """Remove emoji characters from text."""
    return EMOJI_PATTERN.sub("", text)


def remove_urls(text):
    """Remove URLs from text."""
    return URL_PATTERN.sub("", text)


def normalize_unicode(text):
    """Normalize Unicode characters (e.g., accented chars)."""
    return unicodedata.normalize("NFKD", text)


def reduce_punctuation(text):
    """Reduce excessive punctuation (e.g., '!!!' -> '!')."""
    return EXCESSIVE_PUNCT_PATTERN.sub(r"\1", text)


def decode_html_entities(text):
    """Decode HTML entities (e.g., &amp; -> &)."""
    return html.unescape(text)


def normalize_whitespace(text):
    """Normalize whitespace: collapse multiple spaces, strip."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple newlines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (but not newlines) to single space
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def clean_text(text):
    """Apply full cleaning pipeline to produce display-ready text.

    Returns cleaned text with original casing preserved.
    """
    if not text:
        return ""

    text = decode_html_entities(text)
    text = remove_emojis(text)
    text = remove_urls(text)
    text = normalize_unicode(text)
    text = reduce_punctuation(text)
    text = normalize_whitespace(text)

    return text


def clean_text_for_nlp(text):
    """Produce lowercased, cleaned text for NLP processing."""
    cleaned = clean_text(text)
    return cleaned.lower()


def clean_review(review):
    """Clean a review's text fields, adding text_clean and text_display.

    The original 'text' field is preserved unchanged.
    """
    original_text = review.get("text", "")
    review["text_display"] = clean_text(original_text)
    review["text_clean"] = clean_text_for_nlp(original_text)

    # Also clean the title
    original_title = review.get("title", "")
    review["title_clean"] = clean_text_for_nlp(original_title)

    return review
