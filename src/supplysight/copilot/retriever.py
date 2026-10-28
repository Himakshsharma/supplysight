"""Hybrid retrieval over SOP chunks: BM25 (lexical) + dense embeddings, fused with RRF.

Dense backend: ``sentence-transformers/all-MiniLM-L6-v2`` when installed and downloadable,
otherwise a TF-IDF + truncated-SVD (LSA) embedding that needs no network.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from supplysight.config import get_settings
from supplysight.utils import get_logger, optional_import

log = get_logger(__name__)
STOP = set("""a an the and or of to in for on at by with from is are was were be been this that these those it its
as do does did what which who whom how when where why can could should would may might must shall will i we you
our your my me us they them their there here about into than then so if not no yes any all each per vs via up""".split())


_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ers", "er", "als", "al", "ies", "es", "ed", "e", "s")
# domain query expansion: Indian number words and SAP / procurement abbreviations
EXPANSIONS = {
    "lakh": "5,00,000 approval limit value", "lakhs": "5,00,000 approval limit value", "crore": "approval value",
    "po": "purchase order", "pos": "purchase orders", "pr": "purchase requisition", "grn": "goods receipt",
    "gr": "goods receipt", "coa": "certificate analysis", "minbe": "reorder point", "rop": "reorder point",
    "supplier": "vendor", "suppliers": "vendors", "gst": "gstin", "fraud": "audit duplicate split",
    "weekend": "after-hours sunday", "night": "after-hours", "rejected": "quarantine rejection",
    "rated": "rating evaluation score", "rate": "rating evaluation", "rank": "rating evaluation",
    "flagged": "anomaly flag review", "flag": "anomaly review", "late": "delivery performance",
}


def stem(t: str) -> str:
    """Tiny suffix-stripping stemmer (approve/approval/approver -> approv)."""
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stop-words removed and a light suffix stemmer."""
    toks = re.findall(r"[a-z0-9₹%]+", text.lower())
    return [stem(t) for t in toks if t not in STOP and len(t) >= 2]


def expand_query(q: str) -> str:
    """Append domain expansions (e.g. 'lakh' -> '5,00,000 approval limit')."""
    extra = [EXPANSIONS[w] for w in re.findall(r"[a-z]+", q.lower()) if w in EXPANSIONS]
    return q + (" " + " ".join(extra) if extra else "")


@dataclass
class Chunk:
    """One retrievable SOP section."""

    id: str
    doc: str
    title: str
    section: str
    text: str


def chunk_sops(sops_dir: Path) -> list[Chunk]:
    """Split every SOP markdown file into one chunk per ``##`` section (title prepended)."""
    chunks = []
    for p in sorted(Path(sops_dir).glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip()
        doc_id = title.split()[0]
        parts = re.split(r"\n(?=## )", raw)
        for k, part in enumerate(parts[1:] if len(parts) > 1 else parts):
            lines = part.strip().splitlines()
            section = lines[0].lstrip("# ").strip()
            body = "\n".join(lines[1:]).strip()
            chunks.append(Chunk(id=f"{doc_id}#{k + 1}", doc=p.name, title=title, section=section,
                                text=f"{title}. {section}.\n{body}"))
    return chunks


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = docs
        self.avgdl = sum(map(len, docs)) / max(1, len(docs))
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q: list[str]) -> np.ndarray:
        """BM25 score of every document for the query tokens."""
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs, strict=True)):
            denom_const = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            for t in q:
                f = tf.get(t)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + denom_const)
        return out


class DenseEncoder:
    """Sentence-transformers if available, else TF-IDF + LSA."""

    def __init__(self, backend: str = "auto") -> None:
        self.backend = "lsa"
        self.st_model = None
        if backend in ("auto", "st"):
            st = optional_import("sentence_transformers")
            if st is not None:  # pragma: no cover - needs model download
                try:
                    self.st_model = st.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                    self.backend = "sentence-transformers"
                except Exception as exc:
                    log.warning("sentence-transformers unavailable (%s); using LSA", exc)

    def fit(self, texts: list[str]) -> DenseEncoder:
        """Fit the LSA projection (no-op for sentence-transformers)."""
        if self.backend == "lsa":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2),
                                         sublinear_tf=True, token_pattern=None)
            x = self.tfidf.fit_transform(texts)
            k = max(2, min(64, x.shape[0] - 1, x.shape[1] - 1))
            self.svd = TruncatedSVD(n_components=k, random_state=0).fit(x)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        """L2-normalised embeddings."""
        if self.backend == "sentence-transformers":  # pragma: no cover
            v = np.asarray(self.st_model.encode(texts, normalize_embeddings=True))
            return v
        x = self.tfidf.transform(texts)
        v = np.hstack([self.svd.transform(x), x.toarray() * 0.5])  # LSA + raw tf-idf for exact terms
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["st_model"] = None  # never pickle the transformer; reload lazily
        return d


@dataclass
class HybridRetriever:
    """BM25 + dense retrieval fused with reciprocal-rank fusion (k=60)."""

    chunks: list[Chunk]
    encoder: DenseEncoder
    bm25: BM25 = field(init=False)
    emb: np.ndarray = field(init=False)
    rrf_k: int = 60

    def __post_init__(self) -> None:
        self.bm25 = BM25([tokenize(c.text) for c in self.chunks])
        self.encoder.fit([c.text for c in self.chunks])
        self.emb = self.encoder.encode([c.text for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Top-k chunks with fused, BM25 and dense scores."""
        query = expand_query(query)
        b = self.bm25.scores(tokenize(query))
        d = self.emb @ self.encoder.encode([query])[0]
        rb = np.argsort(-b, kind="stable")
        rd = np.argsort(-d, kind="stable")
        fused = np.zeros(len(self.chunks))
        for rank, i in enumerate(rb):
            fused[i] += 1.0 / (self.rrf_k + rank + 1)
        for rank, i in enumerate(rd):
            fused[i] += 1.0 / (self.rrf_k + rank + 1)
        top = np.argsort(-fused, kind="stable")[:k]
        return [{"chunk": self.chunks[i], "score": float(fused[i]), "bm25": float(b[i]), "dense": float(d[i])}
                for i in top]


def index_path() -> Path:
    """Where the serialised retriever lives."""
    return get_settings().artifacts_dir / "copilot_index.joblib"


def build_index(sops_dir: Path | None = None) -> int:
    """Chunk SOPs, build the hybrid index and save it. Returns the number of chunks."""
    s = get_settings()
    chunks = chunk_sops(sops_dir or s.sops_dir)
    r = HybridRetriever(chunks, DenseEncoder(s.embedding_backend))
    joblib.dump(r, index_path())
    log.info("copilot index: %d chunks from %s (%s)", len(chunks), sops_dir or s.sops_dir, r.encoder.backend)
    return len(chunks)


_CACHE: dict = {}


def load_retriever() -> HybridRetriever:
    """Load (and cache) the retriever, building it on first use."""
    p = index_path()
    if not p.exists():
        build_index()
    mtime = p.stat().st_mtime
    if _CACHE.get("mtime") != mtime:
        _CACHE["r"], _CACHE["mtime"] = joblib.load(p), mtime
    return _CACHE["r"]
