"""Transparent keyword router: data (SQL) vs process (SOP RAG) vs insight (ML services)."""

from __future__ import annotations

import re

ROUTES = ("data", "process", "insight")

_PROCESS = [
    r"\bsop\b", r"\bprocess\b", r"\bprocedure", r"\bpolicy\b", r"\bpolicies\b", r"how (do|should|can) (i|we)",
    r"\bhow to\b", r"\bsteps?\b", r"who (approves|can approve|must approve|signs)", r"approval (process|matrix|level|limit)",
    r"\brules?\b", r"\ballowed\b", r"documents? (are )?required", r"\btolerance", r"\bguideline", r"what happens",
    r"\bshould\b", r"\bmust\b", r"\bresponsib", r"three[- ]way", r"\bonboard", r"quarantine", r"\binspection\b",
    r"sampling", r"\bcoa\b", r"evaluation", r"rating", r"amend", r"cancel", r"escalat", r"what is the (process|procedure)",
    r"\bwhen (can|should|is|are|do)\b", r"is it (ok|allowed)", r"prohibit", r"segregation", r"after[- ]hours",
    r"emergency", r"urgent (po|purchase|procurement|order)s?",
]
_INSIGHT = [
    r"\brisk", r"\banomal", r"\bflag", r"suspicious", r"\bfraud", r"stock ?-?outs?", r"run(ning)? (out|short)", r"shortage",
    r"duplicate vendor", r"duplicate supplier", r"reorder (recommendation|point)s? (too low|gap)", r"\bat risk\b",
    r"forecast", r"\bdemand\b", r"unusual", r"\boutlier", r"split po", r"red flag", r"should (we|i) (reorder|worry)",
    r"\bmay stock\b", r"need(s)? reorder",
]
_DATA = [
    r"which vendors?", r"who supplies", r"\bsuppl(y|ies|ier)\b", r"\btop \d*", r"\bspend\b", r"how many", r"\blist\b",
    r"price of", r"\bprice\b", r"lead ?time", r"open (po|purchase)", r"\bstock of\b", r"\btotal\b", r"\baverage\b",
    r"vendors? in\b", r"consum", r"show (me )?(the )?(pos|purchase orders)", r"\bgstin\b", r"\bbom\b", r"used in",
    r"\bcount\b", r"\bmaterials? in\b", r"\bdetails\b", r"\bwhere is\b", r"\bbuyer",
]


def _score(patterns: list[str], q: str) -> int:
    return sum(1 for p in patterns if re.search(p, q))


def route(question: str) -> tuple[str, dict]:
    """Return (route, scores). Ties prefer insight > process > data."""
    q = question.lower()
    scores = {"insight": _score(_INSIGHT, q), "process": _score(_PROCESS, q), "data": _score(_DATA, q)}
    # strong overrides: "how do/should we ..." questions are about the process, not the data
    if re.match(r"\s*(how (do|should|can|are|is|must)|what (is|are) the (process|procedure|steps|rules|policy))\b", q) \
            and not re.search(r"\b(show|list|which|how many)\b", q):
        return "process", scores
    if re.search(r"(high|medium)[- ]risk (po|purchase)|flagged (po|purchase)|stock ?-?out|duplicate (vendor|supplier)s?\b",
                 q) and not re.search(r"\b(sop|policy|procedure|process)\b", q):
        return "insight", scores
    if re.search(r"\b(sop|policy|procedure)\b", q):
        return "process", scores
    best = max(scores.values())
    if best == 0:
        return "data", scores
    for r in ("insight", "process", "data"):
        if scores[r] == best:
            return r, scores
    return "data", scores  # pragma: no cover
