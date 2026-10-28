"""Duplicate-vendor detection: fuzzy name/address similarity plus GSTIN/PAN/bank matches."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

from supplysight.utils import optional_import

_fuzz = optional_import("rapidfuzz.fuzz")
_LEGAL = r"\b(pvt|private|ltd|limited|llp|l\.l\.p\.|inds|industries|enterprises?|and|co|company|unit-ii)\b"
_CANON = {"shri": "shree", "chemical": "chemicals", "packagings": "packaging"}


def normalise_name(name: str) -> str:
    """Lower-case, strip legal forms/punctuation and canonicalise common spelling variants."""
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower().replace("&", " and "))
    s = re.sub(_LEGAL, " ", s)
    toks = [_CANON.get(t, t) for t in s.split()]
    return " ".join(toks)


def text_similarity(a: str, b: str) -> float:
    """Token-sort similarity in [0, 1] (rapidfuzz if installed, difflib otherwise)."""
    a2, b2 = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
    if _fuzz is not None:
        return float(_fuzz.token_sort_ratio(a2, b2)) / 100.0
    return SequenceMatcher(None, a2, b2).ratio()


def _id_score(a: pd.Series, b: pd.Series) -> tuple[float, str]:
    reasons, score = [], 0.0
    if a.stcd3 == b.stcd3:
        score, reasons = 1.0, ["same GSTIN"]
    elif str(a.stcd3)[2:12] == str(b.stcd3)[2:12]:
        score, reasons = 0.9, ["same PAN in GSTIN"]
    if a.bank_ref == b.bank_ref:
        score = 1.0
        reasons.append("same bank account")
    elif str(a.bank_ref)[:-3] == str(b.bank_ref)[:-3]:
        score = max(score, 0.6)
        reasons.append("near-identical bank account")
    return score, ", ".join(reasons)


def find_duplicate_vendors(lfa1: pd.DataFrame, threshold: float = 0.72) -> pd.DataFrame:
    """Return suspected duplicate vendor pairs with a similarity score and evidence.

    ``lifnr_b`` is always the more recently created record (the suspect one).
    similarity = max(0.45 * name + 0.15 * address + 0.40 * id_match, 0.6 * name + 0.25 * address)
    """
    v = lfa1.copy().reset_index(drop=True)
    v["_n"] = v.name1.map(normalise_name)
    v["_a"] = (v.stras.astype(str) + " " + v.ort01.astype(str)).str.lower()
    rows = []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            a, b = v.iloc[i], v.iloc[j]
            ids, why = _id_score(a, b)
            ns = text_similarity(a._n, b._n)
            if ids == 0 and ns < 0.8:
                continue
            ad = text_similarity(a._a, b._a)
            # identifier-backed match, or a near-identical name+address with fresh identifiers
            sim = max(0.45 * ns + 0.15 * ad + 0.40 * ids, 0.6 * ns + 0.25 * ad)
            if sim < threshold:
                continue
            if str(a.created_on) > str(b.created_on):
                a, b = b, a
            evidence = [f"name {ns:.0%}", f"address {ad:.0%}"] + ([why] if why else [])
            rows.append(
                {
                    "lifnr_a": a.lifnr,
                    "lifnr_b": b.lifnr,
                    "name_a": a.name1,
                    "name_b": b.name1,
                    "similarity": round(sim, 4),
                    "match_on": "; ".join(evidence),
                }
            )
    cols = ["lifnr_a", "lifnr_b", "name_a", "name_b", "similarity", "match_on"]
    return pd.DataFrame(rows, columns=cols).sort_values("similarity", ascending=False).reset_index(drop=True)


def vendor_dup_scores(pairs: pd.DataFrame) -> dict[str, tuple[float, str, str]]:
    """Map suspect vendor -> (similarity, matched original vendor, evidence)."""
    out: dict[str, tuple[float, str, str]] = {}
    for r in pairs.itertuples():
        if r.lifnr_b not in out or r.similarity > out[r.lifnr_b][0]:
            out[r.lifnr_b] = (float(r.similarity), r.lifnr_a, r.match_on)
    return out
