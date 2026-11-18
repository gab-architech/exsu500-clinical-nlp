"""Mask specialty names in clinical-note text for the leakage experiment.

One function used by every model family (TF-IDF and ClinicalBERT alike), so the
masking rule is identical across runs. This implements masking option C, agreed
with Gabriel on PR #28 (Nov 18, 2026): the option-A specialty names without
"cardiovascular" and "pulmonary", plus "cardiology" and "pulmonology".

Why those two are dropped: measured on the 2,230 final notes, "cardiovascular"
is a physical-exam heading 77% of the time ("CARDIOVASCULAR: regular rate") and
"pulmonary" is clinical content 96% of the time ("pulmonary embolism"), not the
filing label. Masking them removes structure and signal rather than the label.
"cardiology" and "pulmonology" are the actual specialty names that replace them.

Matching is whole-word and case-insensitive, longest term first, and every match
is replaced with the literal ``[MASK]`` -- the ClinicalBERT mask token, and a
single constant token for TF-IDF, so the replacement itself carries no class
information. See docs/clinicalbert_pipeline.md, section 2.

The term list is fixed before any model sees any score; it must not be tuned
after results exist.
"""

import re

MASK_TOKEN = "[MASK]"

# Option C discipline names (24 terms). Longest-first ordering at compile time
# below means a multi-word term ("pain management") is matched before any
# shorter term that could overlap it.
SPECIALTY_TERMS = (
    "orthopedic",
    "gastroenterology",
    "neurology",
    "neurosurgery",
    "urology",
    "nephrology",
    "obstetrics",
    "gynecology",
    "pediatrics",
    "radiology",
    "hematology",
    "oncology",
    "psychiatry",
    "psychology",
    "ophthalmology",
    "otolaryngology",
    "podiatry",
    "rheumatology",
    "endocrinology",
    "pain management",
    "physical medicine",
    "sleep medicine",
    "cardiology",
    "pulmonology",
)

# One precompiled case-insensitive whole-word pattern per term, longest first.
# re.escape is used defensively even though the terms are plain letters/spaces.
_PATTERNS = tuple(
    re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    for term in sorted(SPECIALTY_TERMS, key=len, reverse=True)
)


def mask_label_strings(text: str) -> str:
    """Replace every specialty name in ``text`` with ``[MASK]``.

    Whole-word and case-insensitive, longest term first. Non-string input is
    returned unchanged so the function is safe to map over a column that may
    contain missing values.
    """
    if not isinstance(text, str):
        return text
    masked = text
    for pattern in _PATTERNS:
        masked = pattern.sub(MASK_TOKEN, masked)
    return masked


def contains_masked_term(text: str) -> bool:
    """Return True if ``text`` contains any specialty term (i.e. would change
    under masking). Used to count how many notes the experiment actually
    touches; see docs/clinicalbert_pipeline.md, section 2, "Report"."""
    if not isinstance(text, str):
        return False
    return any(pattern.search(text) for pattern in _PATTERNS)
