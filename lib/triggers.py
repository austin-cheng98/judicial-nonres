"""The frozen trigger dictionary.

Thirteen expressions, fixed before annotation, unrevised. SHA-1 reported in the
paper. Case-insensitive over whitespace-normalized text, tolerating a few
intervening words, so "need not now decide" matches.

T11 ("reserve"/"reserved") is the one operational departure. In appellate prose
the bare stem mostly means reserved rights, easements, the Federal Reserve, so
it requires an object denoting a legal question or a deferral. Everything else
matches as written.

T08 and T09 are subsets of T07 by construction, kept separate because the design
names them separately. Prevalence counts a passage once, keeping every trigger
id that fired.
"""
import hashlib, re

SPEC = [
    ("T01", "need not decide",
     r"need(?:s|ed)?\s+not\s+(?:\w+\s+){0,2}?decide\b"),
    ("T02", "need not reach",
     r"need(?:s|ed)?\s+not\s+(?:\w+\s+){0,2}?reach\b"),
    ("T03", "do not decide",
     r"\bdo(?:es)?\s+not\s+(?:\w+\s+){0,2}?decide\b"),
    ("T04", "do not reach",
     r"\bdo(?:es)?\s+not\s+(?:\w+\s+){0,2}?reach\b"),
    ("T05", "decline to decide",
     r"declin(?:e|es|ed|ing)\s+to\s+(?:\w+\s+){0,2}?decide\b"),
    ("T06", "decline to address",
     r"declin(?:e|es|ed|ing)\s+to\s+(?:\w+\s+){0,2}?address\b"),
    ("T07", "without deciding",
     r"\bwithout\s+deciding\b"),
    ("T08", "assuming without deciding",
     r"\bassuming\s+(?:\w+\s+){0,4}?without\s+deciding\b"),
    ("T09", "assume without deciding",
     r"assum(?:e|es|ed)\s+(?:\w+\s+){0,4}?without\s+deciding\b"),
    ("T10", "express no opinion",
     r"express(?:es|ed|ing)?\s+no\s+(?:\w+\s+){0,2}?opinion\b"),
    ("T11", "reserve / reserved",
     r"reserv(?:e|es|ed|ing)\s+(?:\w+\s+){0,3}?"
     r"(?:question|questions|issue|issues|matter|judgment|decision)\b"
     r"|reserv(?:e|es|ed|ing)\s+(?:\w+\s+){0,3}?for\s+another\s+(?:day|case)\b"),
    ("T12", "leave for another day",
     r"leav(?:e|es|ing)\s+(?:\w+\s+){0,4}?"
     r"(?:for|to|until)\s+another\s+(?:day|case|occasion)\b"),
    ("T13", "not necessary to decide",
     r"\b(?:not|un)\s*necessary\s+to\s+(?:decide|reach|resolve|address)\b"),
]

NAME = {t: n for t, n, _ in SPEC}
PATTERNS = [(tid, name, re.compile(rx, re.I)) for tid, name, rx in SPEC]
UNION = re.compile("|".join(f"(?:{rx})" for _, _, rx in SPEC), re.I)

DICTIONARY_SHA1 = hashlib.sha1(
    "\n".join(f"{t}\t{n}\t{r}" for t, n, r in SPEC).encode()).hexdigest()


def find_all(text):
    """Return (trigger_id, start, end, matched_string) tuples, left-to-right."""
    hits = []
    for tid, _name, pat in PATTERNS:
        for m in pat.finditer(text):
            hits.append((tid, m.start(), m.end(), m.group(0)))
    hits.sort(key=lambda h: (h[1], -h[2]))
    return hits


def merge_overlaps(hits):
    """Collapse textually overlapping hits into one passage-level occurrence."""
    out = []
    for tid, s, e, txt in hits:
        if out and s < out[-1][2]:
            pid, ps, pe, ptxt = out[-1]
            ids = "+".join(sorted(set(pid.split("+") + [tid])))
            out[-1] = (ids, ps, max(pe, e), ptxt if len(ptxt) >= len(txt) else txt)
        else:
            out.append((tid, s, e, txt))
    return out


if __name__ == "__main__":
    print("dictionary sha1:", DICTIONARY_SHA1)
    for tid, name, rx in SPEC:
        print(f"{tid}  {name:26s}  {rx}")
