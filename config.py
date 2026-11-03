"""Central configuration. All values come from environment variables (loaded from .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip())
    llm_temperature: float = 0.3
    llm_max_output_tokens: int = field(default_factory=lambda: _int("LLM_MAX_OUTPUT_TOKENS", 8192))
    llm_timeout_seconds: int = field(default_factory=lambda: _int("LLM_TIMEOUT_SECONDS", 120))
    llm_max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 4))
    verify_answers: bool = field(default_factory=lambda: _bool("VERIFY_ANSWERS", True))

    # --- Embeddings / retrieval ---
    # bge-small: 33M params, 512-token window, runs comfortably on a laptop CPU.
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip())
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    top_k: int = field(default_factory=lambda: _int("TOP_K", 6))
    # cosine similarity (bge-small, normalised) below this counts as "not relevant to the question"
    min_relevance: float = field(default_factory=lambda: float(os.getenv("MIN_RELEVANCE", "0.5")))
    relative_cutoff: float = 0.15        # drop hits scoring more than this below the best match
    max_context_chars: int = field(default_factory=lambda: _int("MAX_CONTEXT_CHARS", 14000))
    lesson_chars: int = 7000             # size of one "lesson" in chapter-teaching mode
    history_turns: int = 6               # previous messages sent to the LLM

    # --- Chunking ---
    chunk_target_chars: int = 1200
    chunk_max_chars: int = 1900

    # --- Storage ---
    data_dir: Path = BASE_DIR / "data"
    max_pdf_mb: int = field(default_factory=lambda: _int("MAX_PDF_MB", 250))
    page_image_dpi: int = 130

    @property
    def books_dir(self) -> Path:
        return self.data_dir / "books"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def with_api_key(self, key: str) -> "Settings":
        """Return a copy that uses a key entered in the UI (kept only in session memory)."""
        from dataclasses import replace
        return replace(self, gemini_api_key=key.strip())


settings = Settings()
settings.books_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)


def api_key_problem(cfg: Settings) -> str | None:
    """Return a human-readable problem with the API key, or None if it looks usable."""
    key = cfg.gemini_api_key
    if not key:
        return "No Gemini API key found. Add GEMINI_API_KEY to your .env file (or enter it in the sidebar)."
    if key in {"your_free_api_key", "your_api_key_here"}:
        return "GEMINI_API_KEY is still the placeholder value from .env.example."
    if len(key) < 20:
        return "GEMINI_API_KEY looks too short to be a valid key."
    return None
