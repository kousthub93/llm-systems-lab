"""Parse a model's cited answer and score it against labels."""

from __future__ import annotations

import re
from typing import Iterable

from .metrics import best_over_golds, exact_match, f1_score

# Anything that tries to be a chunk id counts as a citation attempt.
# Real ids look like "d12#3"; a garbled "d12#3#3" is still an attempt, so it is
# counted (and judged invalid later) instead of being silently ignored.
DEFAULT_ID_PATTERN = r"^d\d+#\S*$"

_CITE_RE = re.compile(r"\[([^\[\]]+)\]")
_IDK_RE = re.compile(r"^\W*i\s*(?:do\s*not|don'?t|dont)\s*know", re.I)


def parse_answer(text: str, id_pattern: str = DEFAULT_ID_PATTERN) -> dict:
    """Split raw model output into the answer, the chunk ids it cited, and whether it abstained.

    The answer is the first non-empty line with bracketed citations removed.
    """
    id_re = re.compile(id_pattern)
    text = text.replace("’", "'")
    cited: list[str] = []
    for match in _CITE_RE.finditer(text):
        cited += [tok for tok in re.split(r"[,;\s]+", match.group(1)) if id_re.match(tok)]
    cited = list(dict.fromkeys(cited))  # drop repeats, keep order
    cleaned = _CITE_RE.sub("", text)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    answer = lines[0] if lines else ""
    idk = bool(_IDK_RE.match(answer))
    return {"answer": "" if idk else answer.strip(" .,;\t"), "cited": cited, "idk": idk}


def score_answer(
    raw_text: str,
    golds: Iterable[str],
    provided_ids: Iterable[str],
    gold_ids: Iterable[str],
    id_pattern: str = DEFAULT_ID_PATTERN,
) -> dict:
    """Score one generation.

    hit          - a correct chunk was among the chunks given to the model
    has_cite     - the answer cited at least one chunk id
    cite_valid   - every cited id was among the chunks given (no invented ids)
    cite_correct - at least one valid cited chunk contains the answer
    """
    golds = list(golds)
    parsed = parse_answer(raw_text, id_pattern)
    answer, cited = parsed["answer"], parsed["cited"]
    provided, gold = set(provided_ids), set(gold_ids)
    return {
        "answer": answer,
        "idk": parsed["idk"],
        "cited": cited,
        "em": best_over_golds(exact_match, answer, golds),
        "f1": best_over_golds(f1_score, answer, golds),
        "hit": bool(provided & gold),
        "has_cite": bool(cited),
        "cite_valid": bool(cited) and all(c in provided for c in cited),
        "cite_correct": any(c in gold for c in cited if c in provided),
    }
