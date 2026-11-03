"""Local embeddings (sentence-transformers) + ChromaDB vector store + hybrid retrieval."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable

import chromadb

from config import settings
from pdf_processor import Chunk

ProgressCB = Callable[[float, str], None]

_STOP = set("""a an the of to in on for and or is are was were be been being with by as at from that this these those
it its into than then there their what which who whom whose why how when where does do did can could should would will
shall may might must about explain describe define tell give me please show what's whats between using use used also
chapter section page figure fig equation eq textbook book""".split())


class RetrievalError(Exception):
    """Raised with a message that is safe to show to the student."""


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class Embedder:
    """Lazy wrapper around a local sentence-transformers model (downloaded once, then cached by HF)."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._query_cache: dict[str, list[float]] = {}

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device="cpu")
            except Exception as exc:
                raise RetrievalError(
                    f"Could not load the embedding model '{self.model_name}'. The first run needs internet "
                    f"access to download it (~130 MB) from Hugging Face. Details: {exc}") from exc
        return self._model

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            vecs = self._load().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc
        return vecs.tolist()

    def encode_query(self, text: str) -> list[float]:
        if text not in self._query_cache:
            prefix = settings.query_instruction if "bge" in self.model_name.lower() else ""
            if len(self._query_cache) > 256:
                self._query_cache.clear()
            self._query_cache[text] = self.encode_documents([prefix + text])[0]
        return self._query_cache[text]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
_client: chromadb.ClientAPI | None = None


def _chroma() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(settings.chroma_dir),
                                            settings=chromadb.Settings(anonymized_telemetry=False))
    return _client


@dataclass
class Hit:
    chunk: dict
    score: float

    @property
    def page_label(self) -> str:
        a, b = self.chunk["page_start"], self.chunk["page_end"]
        return f"PDF p. {a}" if a == b else f"PDF pp. {a}–{b}"

    @property
    def source_label(self) -> str:
        c = self.chunk
        parts = [c["chapter_title"]]
        if c.get("section_title"):
            parts.append(c["section_title"])
        return " › ".join(p for p in parts if p) + f" — {self.page_label}"


def _meta(c: Chunk) -> dict:
    d = asdict(c)
    d.pop("text")
    return {k: (v if v is not None else "") for k, v in d.items()}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zα-ω0-9]+(?:\.[0-9]+)?", text.lower()) if len(t) > 2 and t not in _STOP}


class TextbookIndex:
    def __init__(self, book_id: str, embedder: Embedder):
        self.book_id = book_id
        self.embedder = embedder
        self.col = _chroma().get_or_create_collection(
            name=f"book_{book_id}", metadata={"hnsw:space": "cosine"}, embedding_function=None)

    # ---- indexing ----
    def count(self) -> int:
        return self.col.count()

    def index_chunks(self, chunks: list[Chunk], progress: ProgressCB | None = None, batch: int = 64) -> int:
        """Embed and store chunks that are not already in the collection. Returns number added."""
        existing: set[str] = set()
        if self.col.count():
            existing = set(self.col.get(include=[])["ids"])
        todo = [c for c in chunks if c.chunk_id not in existing]
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            vecs = self.embedder.encode_documents([c.embed_text() for c in part])
            self.col.add(ids=[c.chunk_id for c in part], embeddings=vecs,
                         documents=[c.text for c in part], metadatas=[_meta(c) for c in part])
            if progress:
                done = min(i + batch, len(todo))
                progress(done / len(todo), f"Embedding chunks locally: {done} of {len(todo)}")
        return len(todo)

    def notes_sources(self) -> list[str]:
        res = self.col.get(where={"source_type": "notes"}, include=["metadatas"])
        return sorted({m["source_name"] for m in res["metadatas"]})

    # ---- retrieval ----
    def search(self, query: str, k: int | None = None, chapter_no: int | None = None,
               include_notes: bool = True, types: list[str] | None = None) -> list[Hit]:
        """Hybrid search: cosine similarity from Chroma, re-ranked with a small keyword/figure/equation boost."""
        k = k or settings.top_k
        if not query.strip() or not self.col.count():
            return []
        type_filter = {"chunk_type": {"$in": types}} if types else {"chunk_type": {"$ne": "references"}}
        scope = None
        if chapter_no is not None:
            scope = {"chapter_no": chapter_no}
            if include_notes:
                scope = {"$or": [scope, {"source_type": "notes"}]}
        elif not include_notes:
            scope = {"source_type": "textbook"}
        where = {"$and": [type_filter, scope]} if scope else type_filter
        try:
            res = self.col.query(query_embeddings=[self.embedder.encode_query(query)],
                                 n_results=min(k * 4, self.col.count()), where=where,
                                 include=["documents", "metadatas", "distances"])
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Search failed: {exc}") from exc

        q_tokens = _tokens(query)
        fig_refs = {m.group(1) for m in re.finditer(r"fig(?:ure)?\.?\s*(\d+\.\d+)", query, re.I)}
        eq_refs = {m.group(1) for m in re.finditer(r"(?:eq(?:uation)?n?\.?)\s*\(?(\d+\.\d+)", query, re.I)}
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1.0 - float(dist)
            if q_tokens:
                overlap = len(q_tokens & _tokens(doc)) / len(q_tokens)
                score += 0.12 * overlap
            if fig_refs and meta.get("figure_labels") in fig_refs:
                score += 0.35
            if eq_refs and any(f"({e})" in doc for e in eq_refs):
                score += 0.3
            hits.append(Hit(chunk={**meta, "text": doc}, score=round(score, 4)))
        hits.sort(key=lambda h: h.score, reverse=True)
        # explicit "Figure 2.3" / "equation (3.4)" references: fetch those chunks directly by label/text
        pinned = self._pinned(fig_refs, eq_refs, chapter_no if chapter_no is not None else None)
        pinned_ids = {h.chunk["chunk_id"] for h in pinned}
        return _dedupe(pinned + [h for h in hits if h.chunk["chunk_id"] not in pinned_ids])[:k]

    def _pinned(self, fig_refs: set[str], eq_refs: set[str], chapter_no: int | None) -> list[Hit]:
        out: list[Hit] = []
        try:
            if fig_refs:
                res = self.col.get(where={"figure_labels": {"$in": sorted(fig_refs)}},
                                   include=["documents", "metadatas"])
                out += [Hit(chunk={**m, "text": d}, score=1.0) for d, m in zip(res["documents"], res["metadatas"])]
            for e in sorted(eq_refs):
                where = {"chapter_no": int(e.split(".")[0])} if chapter_no is None else {"chapter_no": chapter_no}
                res = self.col.get(where=where, where_document={"$contains": f"({e})"},
                                   include=["documents", "metadatas"], limit=2)
                out += [Hit(chunk={**m, "text": d}, score=0.99) for d, m in zip(res["documents"], res["metadatas"])]
        except Exception:
            return []
        out.sort(key=lambda h: (h.chunk["chunk_type"] != "figure_caption", h.chunk["order"]))
        return out

    def section_chunks(self, section_id: str) -> list[dict]:
        res = self.col.get(where={"$and": [{"section_id": section_id}, {"source_type": "textbook"}]},
                           include=["documents", "metadatas"])
        items = [{**m, "text": d} for d, m in zip(res["documents"], res["metadatas"])]
        return sorted(items, key=lambda c: (c["chunk_type"] == "table", c["order"]))

    def chapter_chunks(self, chapter_no: int, types: list[str] | None = None) -> list[dict]:
        where: dict = {"chapter_no": chapter_no}
        if types:
            where = {"$and": [where, {"chunk_type": {"$in": types}}]}
        res = self.col.get(where=where, include=["documents", "metadatas"])
        items = [{**m, "text": d} for d, m in zip(res["documents"], res["metadatas"])]
        return sorted(items, key=lambda c: c["order"])

    def page_chunks(self, page: int) -> list[Hit]:
        res = self.col.get(where={"$and": [{"page_start": {"$lte": page}}, {"page_end": {"$gte": page}},
                                           {"source_type": "textbook"}]},
                           include=["documents", "metadatas"])
        items = sorted(zip(res["documents"], res["metadatas"]), key=lambda x: x[1]["order"])
        return [Hit(chunk={**m, "text": d}, score=1.0) for d, m in items]

    def get(self, ids: list[str]) -> list[Hit]:
        if not ids:
            return []
        res = self.col.get(ids=ids, include=["documents", "metadatas"])
        found = {i: Hit(chunk={**m, "text": d}, score=1.0)
                 for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])}
        return [found[i] for i in ids if i in found]


def _dedupe(hits: list[Hit]) -> list[Hit]:
    """Drop near-duplicate neighbours (chunks overlap by one paragraph)."""
    out: list[Hit] = []
    for h in hits:
        head = h.chunk["text"][:160]
        if any(head in o.chunk["text"] or o.chunk["text"][:160] in h.chunk["text"] for o in out):
            continue
        out.append(h)
    return out


def relevant(hits: list[Hit]) -> list[Hit]:
    return [h for h in hits if h.score >= settings.min_relevance]


def build_context(hits: list[Hit], max_chars: int | None = None) -> tuple[str, list[dict]]:
    """Number passages [S1], [S2]... for citation; returns prompt text and the source list shown to the user."""
    max_chars = max_chars or settings.max_context_chars
    parts, sources, used = [], [], 0
    for h in hits:
        text = h.chunk["text"]
        if used + len(text) > max_chars and parts:
            break
        n = len(sources) + 1
        kind = {"figure_caption": "figure caption", "table": "table", "questions": "end-of-chapter questions",
                "notes": "student notes"}.get(h.chunk.get("chunk_type"), "text")
        if h.chunk.get("source_type") == "notes":
            kind = "student's own notes (not the textbook)"
        parts.append(f"[S{n}] ({kind}; {h.source_label})\n{text}")
        sources.append({"tag": f"S{n}", "label": h.source_label, "chapter_no": h.chunk["chapter_no"],
                        "section_id": h.chunk.get("section_id", ""), "section_title": h.chunk.get("section_title", ""),
                        "page_start": h.chunk["page_start"], "page_end": h.chunk["page_end"],
                        "chunk_id": h.chunk["chunk_id"], "source_type": h.chunk.get("source_type", "textbook"),
                        "score": h.score})
        used += len(text)
    return "\n\n".join(parts), sources


def delete_book_index(book_id: str) -> None:
    try:
        _chroma().delete_collection(f"book_{book_id}")
    except Exception:
        pass
