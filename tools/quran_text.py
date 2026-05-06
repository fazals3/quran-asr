"""
Quran text normalization + tokenization contract (v2).

This project relies on *deterministic*, shared text preprocessing so that:
  - Quran reference text (precomputed assets),
  - ASR transcript text/tokens,
  - and alignment scoring (anchors / embeddings / DP),
all operate on the same representation.

Contract (locked-in decisions)
------------------------------
Normalization (`normalize_arabic`):
  - Remove Arabic diacritics (harakat) and tatweel.
  - Normalize common letter variants for robust matching:
      أ/إ/آ/ٱ -> ا
      ؤ -> و
      ئ/ى -> ي
      ة -> ه   (search-oriented; intentionally lossy)
  - Drop punctuation/symbols; keep letters/marks and digits (incl. Arabic-Indic digits).
  - Collapse whitespace runs to a single space and trim.

Tokenization (word tokens):
  - Token unit is whitespace-separated words after normalization.
  - For per-word alignment, individual tokens are additionally "token-normalized"
    by stripping surrounding punctuation and removing any remaining spaces.

Secondary representation:
  - Character 3-grams over the *space-stripped* normalized string are used for
    cheap similarity scoring and SimHash candidate retrieval.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional


ARABIC_DIACRITICS = {
    "\u0610",
    "\u0611",
    "\u0612",
    "\u0613",
    "\u0614",
    "\u0615",
    "\u0616",
    "\u0617",
    "\u0618",
    "\u0619",
    "\u061a",
    "\u064b",
    "\u064c",
    "\u064d",
    "\u064e",
    "\u064f",
    "\u0650",
    "\u0651",
    "\u0652",
    "\u0653",
    "\u0654",
    "\u0655",
    "\u0656",
    "\u0657",
    "\u0658",
    "\u0659",
    "\u065a",
    "\u065b",
    "\u065c",
    "\u065d",
    "\u065e",
    "\u065f",
    "\u0670",
    "\u06d6",
    "\u06d7",
    "\u06d8",
    "\u06d9",
    "\u06da",
    "\u06db",
    "\u06dc",
    "\u06dd",
    "\u06de",
    "\u06df",
    "\u06e0",
    "\u06e1",
    "\u06e2",
    "\u06e3",
    "\u06e4",
    "\u06e7",
    "\u06e8",
    "\u06ea",
    "\u06eb",
    "\u06ec",
    "\u06ed",
}

ARABIC_TATWEEL = "\u0640"

ARABIC_NORMALIZE_MAP = str.maketrans(
    {
        # Alif variants
        "\u0623": "\u0627",  # أ
        "\u0625": "\u0627",  # إ
        "\u0622": "\u0627",  # آ
        "\u0671": "\u0627",  # ٱ
        # Hamza-on-waw/yaa
        "\u0624": "\u0648",  # ؤ -> و
        "\u0626": "\u064a",  # ئ -> ي
        # Alif maqsurah
        "\u0649": "\u064a",  # ى -> ي
        # Taa marbuta (common normalization for search)
        "\u0629": "\u0647",  # ة -> ه
    }
)


def strip_arabic_diacritics(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in ARABIC_DIACRITICS and ch != ARABIC_TATWEEL)


def normalize_arabic(text: str) -> str:
    text = strip_arabic_diacritics(text)
    text = text.translate(ARABIC_NORMALIZE_MAP)
    kept: List[str] = []
    prev_space = False
    for ch in text:
        if ch.isspace():
            if not prev_space:
                kept.append(" ")
            prev_space = True
            continue

        cat = unicodedata.category(ch)
        if cat[0] in {"L", "M"} or cat == "Nd":
            kept.append(ch)
            prev_space = False
            continue

        # Keep Arabic-Indic digits too.
        if "\u0660" <= ch <= "\u0669":
            kept.append(ch)
            prev_space = False
            continue

        # Drop punctuation/symbols.

    return "".join(kept).strip()


_TRIM_RE = re.compile(r"^\\W+|\\W+$", flags=re.UNICODE)


def clean_token(token: str) -> str:
    token = token or ""
    cleaned = _TRIM_RE.sub("", token)
    return cleaned if cleaned else token


_ARABIC_PREFIXES = ("و", "ف", "ب", "ك", "ل", "س")


def expand_search_terms(simplified_text: str) -> str:
    # Adds common Arabic clitic/prefix variants to make search more forgiving.
    # Example: "والرحمن" -> ["والرحمن", "الرحمن", "رحمن"].
    out: List[str] = []
    for tok in (simplified_text or "").split():
        if not tok:
            continue
        out.append(tok)

        if tok.startswith("ال") and len(tok) > 2:
            out.append(tok[2:])

        if tok[:1] in _ARABIC_PREFIXES and len(tok) > 1:
            out.append(tok[1:])
            if tok[1:].startswith("ال") and len(tok) > 3:
                out.append(tok[3:])

    return " ".join(out)


def normalize_token(token: str) -> str:
    token = clean_token((token or "").strip())
    if not token:
        return ""
    simplified = normalize_arabic(token).replace(" ", "")
    return simplified.strip()


def tokenize_words(text: str) -> List[str]:
    norm = normalize_arabic(text or "")
    return [t for t in norm.split() if t]


def iter_char_ngrams(text: str, *, n: int = 3) -> Iterable[str]:
    s = normalize_arabic(text or "").replace(" ", "")
    if len(s) < n:
        return []
    return (s[i : i + n] for i in range(0, len(s) - n + 1))


def char_ngram_jaccard(a: str, b: str, *, n: int = 3) -> float:
    a_set = set(iter_char_ngrams(a, n=n))
    if not a_set:
        return 0.0
    b_set = set(iter_char_ngrams(b, n=n))
    if not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set) or 1
    return float(inter) / float(union)


def safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

