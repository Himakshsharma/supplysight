"""Copilot guardrails: prompt-injection detection, off-topic refusal, read-only SQL
validation and masking of sensitive vendor fields."""

from __future__ import annotations

import re

import pandas as pd

from supplysight.utils import optional_import

_sqlglot = optional_import("sqlglot")

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |the )?(previous|prior|above|system)",
    r"forget (everything|all|your) (instructions|rules|above)?",
    r"(reveal|show|print|repeat|leak) (me )?(your |the )?(system|hidden|initial) (prompt|instructions|message)",
    r"you are now\b",
    r"act as (an? )?(unrestricted|jailbroken|dan|developer mode)",
    r"\bdan mode\b",
    r"developer mode",
    r"pretend (that )?you (have no|are not bound)",
    r"bypass (the )?(guardrails|filters|rules|safety)",
    r"override (your |the )?(rules|instructions|guardrails)",
    r"<\s*/?\s*(system|assistant)\s*>",
    r"\bsystem prompt\b",
    r"jailbreak",
]
DESTRUCTIVE_WORDS = (
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|grant|revoke|replace|"
    r"vacuum|reindex|copy|merge|call|exec|execute|commit|rollback|savepoint|load_extension)\b"
)
DOMAIN_TERMS = set(
    """vendor vendors supplier suppliers supply supplies supplied po pos purchase order orders requisition
pr spend spent price prices cost material materials stock stockout stock-out inventory reorder lead time sop
procedure process policy approval approve approves approver urgent emergency gst gstin pan bank invoice three-way
match receipt grn goods inspection quarantine qc quality coa plant plants baddi vapi forecast demand consumption
buyer buyers mrp minbe safety risk risky anomaly anomalies flag flagged flags fraud duplicate split onboarding
contract info record packaging raw glycerin fragrance bottle cap carton label butter wax oil oils pigment emulsifier
production bom recipe sap ekpo ekko lfa1 mara marc eban mseg cream lotion shampoo lipstick sunscreen serum how many
top total average list show which who what quarter month year week weeks cosmetics chemicals open closed delivery
delivered tolerance rating evaluation amendment cancel cancellation index variance review reviewer""".split()
)
SENSITIVE_COLUMNS = {"stcd3", "bank_ref", "gstin"}
GSTIN_RE = re.compile(r"\b(\d{2})([A-Z]{5}\d{4}[A-Z])([1-9A-Z])Z([0-9A-Z])\b")
BANK_RE = re.compile(r"\b([A-Z]{4})0\d{6}/(\d{10,16})\b")


def detect_injection(text: str) -> str | None:
    """Return a reason string if the text looks like a prompt-injection attempt."""
    low = text.lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, low):
            return "Possible prompt-injection attempt detected"
    if re.search(DESTRUCTIVE_WORDS + r"\s+(table|from|into|database|set)\b", low) or re.search(
        r";\s*(drop|delete|update)\b", low
    ):
        return "Requests to modify data are not allowed (read-only assistant)"
    return None


def is_off_topic(text: str) -> bool:
    """True when the question has no procurement / planning vocabulary at all."""
    toks = set(re.findall(r"[a-z0-9\-]+", text.lower()))
    if re.search(r"\b\d{6}\b", text):  # material / vendor numbers
        return False
    return len(toks & DOMAIN_TERMS) == 0 or (
        len(toks & DOMAIN_TERMS) <= 1
        and toks & {"weather", "cricket", "movie", "recipe", "poem", "joke", "song", "football", "politics", "election"}
    )


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def validate_sql(sql: str, allowed_tables: list[str], row_limit: int = 200) -> tuple[bool, str, str]:
    """Validate a query for the read-only copilot. Returns (ok, reason, safe_sql).

    Rules: single statement; SELECT/WITH only; no DDL/DML keywords; only whitelisted
    tables; a LIMIT no larger than ``row_limit`` is enforced.
    """
    raw = _strip_comments(sql).strip().rstrip(";").strip()
    if not raw:
        return False, "empty query", ""
    if ";" in raw:
        return False, "multiple statements are not allowed", ""
    low = raw.lower()
    if not re.match(r"^\s*(select|with)\b", low):
        return False, "only SELECT queries are allowed", ""
    if re.search(DESTRUCTIVE_WORDS, low):
        return False, "query contains a forbidden keyword", ""
    if re.search(r"sqlite_|pg_catalog|information_schema|pg_sleep|randomblob", low):
        return False, "system catalogue access is not allowed", ""
    allowed = {t.lower() for t in allowed_tables}
    if _sqlglot is not None:  # pragma: no cover - optional precise parser
        try:
            tree = _sqlglot.parse_one(raw)
            if tree.key not in ("select", "union", "with"):
                return False, "only SELECT queries are allowed", ""
            ctes = {c.alias_or_name.lower() for c in tree.find_all(_sqlglot.exp.CTE)}
            tables = {t.name.lower() for t in tree.find_all(_sqlglot.exp.Table)} - ctes
        except Exception as exc:
            return False, f"could not parse SQL: {exc}", ""
    else:
        ctes = {m.lower() for m in re.findall(r"(?:with|,)\s*([a-z_][a-z0-9_]*)\s+as\s*\(", low)}
        tables = {m.lower() for m in re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_\.]*)", low)} - ctes
    bad = sorted(t for t in tables if t not in allowed)
    if bad:
        return False, f"table(s) not allowed: {', '.join(bad)}", ""
    m = re.search(r"\blimit\s+(\d+)\s*$", low)
    if m:
        if int(m.group(1)) > row_limit:
            raw = raw[: m.start()] + f"LIMIT {row_limit}"
    else:
        raw = f"{raw}\nLIMIT {row_limit}"
    return True, "ok", raw


def mask_value(v: object) -> object:
    """Mask a GSTIN or bank reference, keeping enough to recognise it."""
    if not isinstance(v, str):
        return v
    v = GSTIN_RE.sub(lambda m: f"{m.group(1)}{m.group(2)[:2]}******{m.group(2)[-1]}{m.group(3)}Z{m.group(4)}", v)
    return BANK_RE.sub(lambda m: f"{m.group(1)}0******/{'*' * (len(m.group(2)) - 4)}{m.group(2)[-4:]}", v)


def mask_frame(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """Mask sensitive columns (and GSTIN/bank patterns anywhere) unless role is admin."""
    if role == "admin" or df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if c.lower() in SENSITIVE_COLUMNS or out[c].dtype == object:
            out[c] = out[c].map(mask_value)
    return out


def mask_text(text: str, role: str) -> str:
    """Mask GSTIN / bank references in free text unless role is admin."""
    return text if role == "admin" else mask_value(text)
