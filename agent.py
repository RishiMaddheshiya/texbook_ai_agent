"""LangGraph teaching agent: classify → retrieve → plan → teach → verify → format → update learning state.
Also contains the Gemini client (retries/backoff), quiz generation/grading and progress persistence."""
from __future__ import annotations

import difflib
import json
import random
import re
import time
from datetime import datetime
from typing import Any, Callable, Literal, Optional, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import Settings, api_key_problem, settings as default_settings
from pdf_processor import Chapter, ProcessedBook, render_page_png
from prompts import (FOLLOWUP_INSTRUCTIONS, MODE_INSTRUCTIONS, QUIZ_EVAL_PROMPT, QUIZ_PROMPT, VERIFY_PROMPT,
                     system_prompt)
from retrieval import Hit, RetrievalError, TextbookIndex, build_context, relevant

LEARNING_MODES = [
    "Complete Chapter Teaching", "Concept Explanation", "Mathematical Derivations", "Diagram Explanation",
    "Numerical Problem Solver", "Exam Preparation", "Quick Revision", "Quiz Mode",
]
MODE_INTENT = dict(zip(LEARNING_MODES, [
    "chapter_teaching", "concept", "derivation", "diagram", "numerical", "exam_prep", "revision", "quiz"]))
INTENT_LABELS = {
    "chapter_teaching": "Chapter teaching", "concept": "Concept explanation", "derivation": "Equation derivation",
    "numerical": "Numerical solving", "diagram": "Diagram interpretation", "comparison": "Comparison",
    "exam_prep": "Exam preparation", "revision": "Revision", "quiz": "Quiz generation",
    "general": "General textbook question",
}


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Raised with a message that is safe to show to the student (never contains the API key)."""


def _retry_after(exc: Exception) -> float | None:
    m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", str(exc), re.I)
    return float(m.group(1)) if m else None


class GeminiClient:
    def __init__(self, cfg: Settings):
        problem = api_key_problem(cfg)
        if problem:
            raise LLMError(problem)
        from google import genai
        from google.genai import errors, types
        self._types, self._errors, self.cfg = types, errors, cfg
        self.client = genai.Client(api_key=cfg.gemini_api_key,
                                   http_options=types.HttpOptions(timeout=cfg.llm_timeout_seconds * 1000))
        self.calls = 0
        self.last_seconds = 0.0
        self.on_status: Callable[[str], None] | None = None  # UI hook: "rate limited, retrying in 12 s"
        self._thinking_ok = bool(cfg.gemini_thinking_level)

    def _config(self, system: str, schema, temperature, max_tokens):
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system,
            temperature=self.cfg.llm_temperature if temperature is None else temperature,
            max_output_tokens=max_tokens or self.cfg.llm_max_output_tokens)
        if self._thinking_ok:
            # Less "thinking" = much faster first token. Thinking tokens also count towards max_output_tokens.
            config.thinking_config = t.ThinkingConfig(thinking_level=self.cfg.gemini_thinking_level.upper())
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = schema
        return config

    def _with_retries(self, call: Callable[[], Any], started: Callable[[], bool] = lambda: False):
        """Run `call` with exponential backoff on rate limits, overload and network errors."""
        delay = 2.0
        last: Exception | None = None
        for attempt in range(self.cfg.llm_max_retries + 1):
            t0 = time.perf_counter()
            try:
                self.calls += 1
                result = call()
                self.last_seconds = time.perf_counter() - t0
                return result
            except self._errors.APIError as exc:
                last = exc
                code = getattr(exc, "code", 0) or 0
                text = str(exc)
                if code == 400 and self._thinking_ok and re.search(r"thinking", text, re.I):
                    self._thinking_ok = False  # this model does not accept the thinking level: retry without it
                    continue
                if code == 429 and re.search(r"per ?day|PerDay|daily", text, re.I):
                    raise LLMError("The daily free-tier quota for this Gemini model is used up. It resets every "
                                   "24 hours; you can also set GEMINI_MODEL in .env to another free model.") from exc
                if code not in (429, 500, 502, 503, 504) or attempt == self.cfg.llm_max_retries or started():
                    raise LLMError(self._friendly(code, text)) from exc
                wait = _retry_after(exc) or delay
                why = "Gemini rate limit reached" if code == 429 else "Gemini is busy"
            except (httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError) as exc:
                last = exc
                if attempt == self.cfg.llm_max_retries or started():
                    raise LLMError("The Gemini API did not respond in time (network problem or timeout). "
                                   "Please check your internet connection and try again.") from exc
                wait, why = delay, "Network problem"
            wait = min(wait, self.cfg.llm_max_retry_wait) + random.uniform(0, 1)
            if self.on_status:
                self.on_status(f"{why} — retrying in {wait:.0f} s (attempt {attempt + 2} of {self.cfg.llm_max_retries + 1})")
            time.sleep(wait)
            delay = min(delay * 2, self.cfg.llm_max_retry_wait)
        raise LLMError(f"Gemini request failed: {last}")

    def generate(self, system: str, contents: list[Any], schema: type[BaseModel] | None = None,
                 temperature: float | None = None, max_tokens: int | None = None):
        def call():
            resp = self.client.models.generate_content(
                model=self.cfg.gemini_model, contents=contents,
                config=self._config(system, schema, temperature, max_tokens))
            return self._read(resp, schema)
        return self._with_retries(call)

    def generate_stream(self, system: str, contents: list[Any], on_text: Callable[[str], None],
                        temperature: float | None = None, max_tokens: int | None = None) -> str:
        """Stream a plain-text answer, calling on_text(text_so_far) as chunks arrive. Returns the full text.
        Retries only happen before the first chunk, so the student never sees a half answer restart."""
        got_chunk = {"v": False}

        def call():
            text, finish, block = "", "", None
            stream = self.client.models.generate_content_stream(
                model=self.cfg.gemini_model, contents=contents,
                config=self._config(system, None, temperature, max_tokens))
            for chunk in stream:
                try:
                    piece = chunk.text or ""
                except Exception:
                    piece = ""
                try:
                    finish = str(chunk.candidates[0].finish_reason or finish)
                except Exception:
                    pass
                block = getattr(getattr(chunk, "prompt_feedback", None), "block_reason", None) or block
                if piece:
                    got_chunk["v"] = True
                    text += piece
                    on_text(text)
            if not text:
                raise LLMError(f"The model returned no text (finish reason: {finish or block or 'unknown'}). "
                               "Try rephrasing the question.")
            if "MAX_TOKENS" in finish:
                text += "\n\n*(The answer hit the length limit — type **continue** to get the rest.)*"
            return text
        return self._with_retries(call, started=lambda: got_chunk["v"])

    def _friendly(self, code: int, text: str) -> str:
        model = self.cfg.gemini_model
        if code == 400 and re.search(r"api[_ ]?key", text, re.I):
            return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file (create one at aistudio.google.com)."
        if code in (401, 403):
            return "Gemini refused the request (permission denied). Check that your API key is valid and enabled."
        if code == 404:
            return (f"The model '{model}' is not available to your API key. Set GEMINI_MODEL in .env to a model "
                    f"listed at ai.google.dev/gemini-api/docs/models (e.g. gemini-3.5-flash-lite).")
        if code == 429:
            return ("Gemini free-tier rate limit reached (too many requests per minute). Wait about a minute and "
                    "try again, or turn off 'Double-check answers' in the sidebar to halve API usage.")
        if code >= 500:
            return "Gemini is temporarily overloaded. Please try again in a few moments."
        return f"Gemini request failed (HTTP {code})."

    def _read(self, resp, schema):
        text = None
        try:
            text = resp.text
        except Exception:
            text = None
        finish = ""
        try:
            finish = str(resp.candidates[0].finish_reason or "")
        except Exception:
            pass
        if schema is not None:
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, schema):
                return parsed
            if text:
                try:
                    return schema.model_validate_json(text)
                except Exception as exc:
                    raise LLMError(f"The model returned malformed structured output ({exc.__class__.__name__}).")
        if not text:
            block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
            raise LLMError(f"The model returned no text (finish reason: {finish or block or 'unknown'}). "
                           "Try rephrasing the question.")
        if "MAX_TOKENS" in finish:
            text += "\n\n*(The answer hit the length limit — type **continue** to get the rest.)*"
        return text


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------
class VerificationResult(BaseModel):
    verdict: Literal["pass", "revise"]
    unsupported_claims: list[str] = Field(default_factory=list)
    math_or_unit_issues: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    addresses_question: bool = True
    revised_answer: str = ""


class QuizQuestion(BaseModel):
    question: str
    qtype: Literal["mcq", "true_false", "fill_blank", "conceptual", "numerical"]
    options: list[str] = Field(default_factory=list)
    correct_answer: str
    accepted_answers: list[str] = Field(default_factory=list)
    explanation: str
    topic: str
    source_tag: str = ""


class QuizSet(BaseModel):
    questions: list[QuizQuestion]


class AnswerEvaluation(BaseModel):
    score: float
    is_correct: bool
    feedback: str


# ---------------------------------------------------------------------------
# Learning state (persisted per book)
# ---------------------------------------------------------------------------
def new_learning_state() -> dict:
    return {
        "current_chapter": None, "current_topic": "", "discussed": [], "recent_topics": [],
        "completed_sections": [], "chapters_started": [], "lesson_pointer": {}, "quiz_history": [],
        "weak_topics": {}, "asked_questions": [], "last_chunk_ids": [], "last_intent": "", "difficulty": "",
        "quiz": None,
    }


def load_progress(book: ProcessedBook) -> dict:
    path = book.folder / "progress.json"
    state = new_learning_state()
    if path.exists():
        try:
            state.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return state


def save_progress(book: ProcessedBook, learning: dict) -> None:
    try:
        (book.folder / "progress.json").write_text(json.dumps(learning, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # inputs
    query: str
    mode: str
    difficulty: str
    chapter_no: Optional[int]
    history: list[dict]
    learning: dict
    action: dict
    verify: bool
    # produced by nodes
    intent: str
    followup: str
    retrieval_query: str
    hits: list[dict]
    context: str
    sources: list[dict]
    image_pages: list[int]
    lesson: dict
    plan: dict
    plan_text: str
    low_context: bool
    draft: str
    answer: str
    verification: dict
    warnings: list[str]
    suggestions: list[dict]
    quiz: dict
    timings: dict


# ---------------------------------------------------------------------------
# Node 1 helpers: query classification (rule-based, no API call)
# ---------------------------------------------------------------------------
_FOLLOWUP_PATTERNS = [
    ("simpler", r"\b(simpl|easier|easy way|basic terms|samajh nahi|samjh nahi|nahi samjha|confus|eli5|layman|aasan)"),
    ("detail", r"\b(more detail|in detail|elaborate|deeper|in depth|expand|vistar|detail me)"),
    ("example", r"\b(another example|more examples?|give (me )?an? example|example do|udaharan)"),
    ("exam", r"\b(exam[- ]?ready|exam answer|for (the )?exam|marks answer|answer for exam)"),
    ("continue", r"^(continue|go on|carry on|aage( batao)?|keep going|and then\??)$|where we stopped|left off"),
    ("again", r"\b(again|repeat|phir se|dobara|once more)\b"),
    ("reference", r"\b(first|second|third|fourth|fifth|last|previous|above|that|this) "
                  r"(equation|eqn|step|term|figure|point|part|formula|example|diagram|line)"),
    ("why", r"^(why|how come|kyu|kyon|kaise)\b.*\b(this|that|it|happen|hota|hai)\b"),
]
_INTENT_CUES = [
    ("quiz", r"\b(quiz|test me|mcqs?\b|ask me (some )?questions)"),
    ("exam_prep", r"\b(exam|important questions|\d+[- ]?marks?\b|long[- ]answer|viva|question paper|pyq)"),
    ("revision", r"\b(revise|revision|quick summary|summari[sz]e|cheat ?sheet|formula sheet|key points|"
                 r"short notes)"),
    ("comparison", r"\b(compare|comparison|differences? between|distinguish|contrast|vs\.?|versus)\b"),
    ("derivation", r"\b(deriv|prove|proof|show that|obtain (the )?expression|expression for)"),
    ("diagram", r"\b(fig(ure)?\.?\s*\d|diagram|graph|plot|sketch|draw|schematic|illustration|ray diagram|"
                r"picture|image on page)"),
    ("chapter_teaching", r"\b(teach (me )?(the |this |whole )?chapter|start (the )?chapter|next section|"
                         r"continue (the )?chapter|next lesson)"),
]
_NUMERIC = re.compile(r"\d+(\.\d+)?\s*(k?e?v|kv|kev|mev|nm|pm|å|a\b|mrad|rad|°|deg|mm|µm|um|m/s|a\b|na|pa|k\b)",
                      re.I)
_NUM_CUE = re.compile(r"\b(calculate|compute|find the|estimate|evaluate|how (much|many|large|small)|numerical|"
                      r"solve|value of)\b", re.I)


def classify_query(query: str, mode: str, has_history: bool, last_intent: str,
                   action: dict | None) -> tuple[str, str]:
    """Return (intent, follow-up kind)."""
    if action:
        return {"teach": "chapter_teaching", "quiz": "quiz", "exam": "exam_prep",
                "revision": "revision"}.get(action.get("type", ""), MODE_INTENT.get(mode, "concept")), \
            action.get("followup", "")
    q = query.strip().lower()
    mode_intent = MODE_INTENT.get(mode, "concept")
    followup = ""
    if has_history and len(q.split()) <= 16:
        for kind, pat in _FOLLOWUP_PATTERNS:
            if re.search(pat, q):
                followup = kind
                break
    for intent, pat in _INTENT_CUES:
        if re.search(pat, q):
            if intent == "exam_prep" and followup == "exam":
                break  # "make it exam-ready" is a follow-up, not a new exam set
            return intent, followup
    if _NUM_CUE.search(q) and (re.search(r"\d", q) or _NUMERIC.search(q)) or len(_NUMERIC.findall(q)) >= 2:
        return "numerical", followup
    if followup:
        base = last_intent if last_intent and last_intent not in ("quiz", "chapter_teaching") else "concept"
        if last_intent == "chapter_teaching" and followup == "continue":
            return "chapter_teaching", ""
        return base, followup
    if mode_intent == "chapter_teaching":
        return ("chapter_teaching", "") if q in {"", "next", "continue", "start", "aage", "go", "ok", "okay"} \
            else ("concept", "")
    if mode_intent == "quiz":
        return ("quiz", "") if q in {"", "start", "start quiz", "quiz", "new quiz"} else ("concept", "")
    if re.match(r"^(which|what) chapters?\b|^(list|show) (the )?chapters", q):
        return "general", ""
    return mode_intent, ""


# ---------------------------------------------------------------------------
# Helpers for lessons and chapter digests
# ---------------------------------------------------------------------------
def chapter_sequence(book: ProcessedBook, chapter: Chapter) -> list[tuple[str, str]]:
    """Ordered (section_id, title) teaching units, starting with the chapter introduction."""
    seq = [(f"c{chapter.number}-intro", f"{chapter.title} — introduction")]
    seq += [(s.id, s.title) for s in chapter.teaching_sections()]
    return seq


def section_lessons(index: TextbookIndex, section_id: str, lesson_chars: int) -> list[list[dict]]:
    chunks = [c for c in index.section_chunks(section_id) if c["chunk_type"] not in ("references",)]
    lessons, cur, size = [], [], 0
    for c in chunks:
        if cur and size + len(c["text"]) > lesson_chars:
            lessons.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += len(c["text"])
    if cur:
        lessons.append(cur)
    return lessons


def chapter_digest(index: TextbookIndex, chapter: Chapter, budget: int, with_questions: bool = False) -> list[Hit]:
    """Representative passages from every section of a chapter, within a character budget."""
    teach = [f"c{chapter.number}-intro"] + [s.id for s in chapter.teaching_sections()]
    per_section = max(900, budget // max(1, len(teach)))
    hits: list[Hit] = []
    for sid in teach:
        chunks = [c for c in index.section_chunks(sid) if c["chunk_type"] in ("text", "table")]
        # prefer the opening chunk and chunks that carry numbered equations
        ranked = chunks[:1] + sorted(chunks[1:], key=lambda c: -len(re.findall(r"\(\d+\.\d+[a-z]?\)", c["text"])))
        used = 0
        for c in ranked:
            if used and used + len(c["text"]) > per_section:
                continue
            hits.append(Hit(chunk=c, score=1.0))
            used += len(c["text"])
    if with_questions:
        qs = index.chapter_chunks(chapter.number, ["questions"])[:3]
        hits += [Hit(chunk=c, score=1.0) for c in qs]
    hits.sort(key=lambda h: h.chunk["order"])
    return hits


def _merge_hits(*groups: list[Hit], k: int) -> list[Hit]:
    best: dict[str, Hit] = {}
    for g in groups:
        for h in g:
            cid = h.chunk["chunk_id"]
            if cid not in best or h.score > best[cid].score:
                best[cid] = h
    return sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]


def _history_text(history: list[dict], turns: int, followup: str) -> str:
    if not history:
        return ""
    recent = history[-turns:]
    lines = []
    for i, m in enumerate(recent):
        limit = 6000 if (followup and i == len(recent) - 1 and m["role"] == "assistant") else 1200
        content = m["content"].split("\n---\n**📚")[0]  # drop the appended source list
        if len(content) > limit:
            content = content[:limit] + " …"
        lines.append(f"{'STUDENT' if m['role'] == 'user' else 'PROFESSOR'}: {content}")
    return "\n\n".join(lines)


_PAGE_REF = re.compile(r"\b(?:page|pg\.?|p\.)\s*(\d{1,4})\b", re.I)
_FIG_REF = re.compile(r"\bfig(?:ure)?\.?\s*(\d+\.\d+)", re.I)


def clean_answer(text: str, valid_tags: set[str]) -> str:
    text = text.strip()
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.S)
    # remove citation tags that do not correspond to a supplied passage
    text = re.sub(r"\[(S\d+)\]", lambda m: m.group(0) if m.group(1) in valid_tags else "", text)
    text = re.sub(r"\[(S\d+(?:\s*,\s*S\d+)+)\]",
                  lambda m: "".join(f"[{t.strip()}]" for t in m.group(1).split(",") if t.strip() in valid_tags),
                  text)
    # strip a model-written sources list (we append the real one)
    text = re.split(r"\n#{1,4}\s*(?:📚\s*)?(?:Sources|References)\s*\n", text)[0]
    return text.rstrip()


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class TeachingAgent:
    def __init__(self, book: ProcessedBook, index: TextbookIndex, llm: GeminiClient | None,
                 cfg: Settings | None = None, on_token: Callable[[str], None] | None = None):
        self.book, self.index, self.llm = book, index, llm
        self.cfg = cfg or default_settings
        self.on_token = on_token  # optional UI hook: receives the answer text as it streams in
        self.outline = "\n".join(f"Chapter {c.number}: {c.title} (PDF pp. {c.page_start}–{c.page_end})"
                                 for c in book.chapters)
        self.graph = self._build()

    # ---- graph wiring ----
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("classify", self.node_classify)
        g.add_node("retrieve", self.node_retrieve)
        g.add_node("plan", self.node_plan)
        g.add_node("teach", self.node_teach)
        g.add_node("verify", self.node_verify)
        g.add_node("format", self.node_format)
        g.add_node("quiz", self.node_quiz)
        g.add_node("update_state", self.node_update_state)
        g.add_edge(START, "classify")
        g.add_edge("classify", "retrieve")
        g.add_conditional_edges("retrieve", self._route_after_retrieve,
                                {"quiz": "quiz", "plan": "plan", "done": "update_state"})
        g.add_edge("plan", "teach")
        g.add_edge("teach", "verify")
        g.add_edge("verify", "format")
        g.add_edge("format", "update_state")
        g.add_edge("quiz", "update_state")
        g.add_edge("update_state", END)
        return g.compile()

    @staticmethod
    def _route_after_retrieve(state: AgentState) -> str:
        if state.get("answer"):
            return "done"  # e.g. chapter finished: nothing to send to the LLM
        return "quiz" if state["intent"] == "quiz" else "plan"

    def run(self, state: AgentState, on_step: Callable[[str], None] | None = None) -> AgentState:
        final: dict = dict(state)
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        for update in self.graph.stream(state, stream_mode="updates"):
            now = time.perf_counter()
            for node, delta in update.items():
                timings[node] = round(now - t0, 2)
                if delta:
                    final.update(delta)
                if on_step:
                    on_step(node)
            t0 = now
        final["timings"] = timings
        return final  # type: ignore[return-value]

    def _require_llm(self) -> GeminiClient:
        if self.llm is None:
            raise LLMError("No Gemini API key is configured.")
        return self.llm

    def _chapter(self, state: AgentState) -> Chapter | None:
        n = state.get("chapter_no")
        return self.book.chapter(n) if n is not None else None

    # ---- Node 1: Query Classifier ----
    def node_classify(self, state: AgentState) -> dict:
        learning = state.get("learning") or new_learning_state()
        has_history = any(m["role"] == "assistant" for m in state.get("history", []))
        intent, followup = classify_query(state.get("query", ""), state.get("mode", ""), has_history,
                                          learning.get("last_intent", ""), state.get("action"))
        rq = state.get("query", "")
        if followup and learning.get("current_topic"):
            rq = f"{learning['current_topic']} {rq}"
        return {"intent": intent, "followup": followup, "retrieval_query": rq, "warnings": []}

    # ---- Node 2: Context Retriever ----
    def node_retrieve(self, state: AgentState) -> dict:
        intent, cfg = state["intent"], self.cfg
        chapter = self._chapter(state)
        learning = state.get("learning") or new_learning_state()
        query = state.get("query", "")
        image_pages: list[int] = []
        lesson: dict = {}
        try:
            if intent == "chapter_teaching":
                return self._retrieve_lesson(state, chapter, learning)

            if intent in ("exam_prep", "revision", "quiz") and chapter is not None and not self._has_topic(query):
                hits = chapter_digest(self.index, chapter, int(cfg.max_context_chars * 1.4),
                                      with_questions=intent != "revision")
                context, sources = build_context(hits, int(cfg.max_context_chars * 1.5))
                return {"hits": [h.chunk for h in hits], "context": context, "sources": sources,
                        "image_pages": [], "low_context": False, "lesson": {}}

            k = cfg.top_k + (4 if intent in ("exam_prep", "revision", "quiz", "comparison") else 0)
            rq = state.get("retrieval_query") or query
            groups = [self.index.search(rq, k)]
            if chapter is not None:
                in_ch = self.index.search(rq, k, chapter_no=chapter.number)
                for h in in_ch:
                    h.score += 0.04  # small preference for the chapter the student is studying
                groups.append(in_ch)
            if state.get("followup") and learning.get("last_chunk_ids"):
                groups.append(self.index.get(learning["last_chunk_ids"][:6]))
            page_refs = [int(p) for p in _PAGE_REF.findall(query) if 1 <= int(p) <= self.book.page_count]
            for p in page_refs[:2]:
                groups.append(self.index.page_chunks(p)[:4])
                image_pages.append(p)
            hits = _merge_hits(*groups, k=k)
            # keep only hits close to the best (non-pinned) match, but always at least three
            free = [h.score for h in hits if h.score < 0.99]
            if free:
                floor = max(free) - cfg.relative_cutoff
                kept = [h for h in hits if h.score >= floor]
                hits = kept if len(kept) >= 3 else hits[:3]
        except RetrievalError as exc:
            return {"hits": [], "context": "", "sources": [], "image_pages": [], "low_context": True,
                    "lesson": lesson, "warnings": [str(exc)]}

        low = not relevant(hits) and not image_pages
        if intent in ("diagram", "derivation") or image_pages:
            image_pages = self._pick_image_pages(intent, query, hits, image_pages)
        context, sources = build_context(hits)
        return {"hits": [{**h.chunk, "_score": h.score} for h in hits], "context": context, "sources": sources,
                "image_pages": image_pages, "low_context": low, "lesson": lesson}

    @staticmethod
    def _has_topic(query: str) -> bool:
        stripped = re.sub(r"\b(exam|prep(aration)?|revision|revise|quick|quiz|start|questions?|generate|make|"
                          r"give|me|for|this|the|chapter|a|an|sheet|notes|important|please|of|on|new)\b", "",
                          query.lower())
        return len(re.findall(r"[a-z]{3,}", stripped)) >= 2

    def _pick_image_pages(self, intent: str, query: str, hits: list[Hit], pages: list[int]) -> list[int]:
        pages = list(pages)
        for lab in _FIG_REF.findall(query):
            p = self.book.figure_index.get(lab)
            if p and p not in pages:
                pages.append(p)
        if not pages:
            if intent == "diagram":
                cap = next((h for h in hits if h.chunk["chunk_type"] == "figure_caption"), None)
                if cap:
                    pages.append(cap.chunk["page_start"])
                elif hits:
                    pages.append(hits[0].chunk["page_start"])
            elif intent == "derivation":
                refs = re.findall(r"(?:eq(?:uation)?n?\.?)\s*\(?(\d+\.\d+[a-z]?)", query, re.I)
                eq = [h for h in hits if any(f"({r})" in h.chunk["text"] for r in refs)] or \
                    [h for h in hits if re.search(r"\(\d+\.\d+[a-z]?\)", h.chunk["text"])][:1]
                for h in eq[:2]:
                    if h.chunk["page_start"] not in pages:
                        pages.append(h.chunk["page_start"])
        return pages[:2]

    def _retrieve_lesson(self, state: AgentState, chapter: Chapter | None, learning: dict) -> dict:
        if chapter is None:
            return {"answer": "Please select a chapter in the sidebar first, then press **Teach next lesson**.",
                    "hits": [], "sources": [], "context": "", "image_pages": [], "lesson": {}}
        action = state.get("action") or {}
        seq = chapter_sequence(self.book, chapter)
        ptr = dict(learning.get("lesson_pointer", {}).get(str(chapter.number), {"section_idx": 0, "part": 0}))
        if action.get("op") == "restart":
            ptr = {"section_idx": 0, "part": 0}
        elif action.get("op") == "jump" and action.get("section_id"):
            ids = [s[0] for s in seq]
            if action["section_id"] in ids:
                ptr = {"section_idx": ids.index(action["section_id"]), "part": 0}
        elif action.get("op") == "repeat" and learning.get("last_lesson"):
            last = learning["last_lesson"]
            if last.get("chapter") == chapter.number:
                ptr = {"section_idx": last["section_idx"], "part": last["part"]}

        # skip empty sections (e.g. a chapter without an introduction paragraph)
        while ptr["section_idx"] < len(seq):
            lessons = section_lessons(self.index, seq[ptr["section_idx"]][0], self.cfg.lesson_chars)
            if ptr["part"] < len(lessons):
                break
            ptr = {"section_idx": ptr["section_idx"] + 1, "part": 0}
        if ptr["section_idx"] >= len(seq):
            done = (f"🎉 You have completed all lessons of **{chapter.label}**.\n\n"
                    "Suggested next steps: take a **Quiz** on this chapter, generate **Exam Preparation** material, "
                    "or restart the chapter from the beginning.")
            return {"answer": done, "hits": [], "sources": [], "context": "", "image_pages": [],
                    "lesson": {"chapter": chapter.number, "finished": True, "pointer": ptr}}

        sid, stitle = seq[ptr["section_idx"]]
        chunks = lessons[ptr["part"]]
        hits = [Hit(chunk=c, score=1.0) for c in chunks]
        context, sources = build_context(hits, self.cfg.lesson_chars + 2000)
        caption_pages = [c["page_start"] for c in chunks if c["chunk_type"] == "figure_caption"]
        eq_pages = [c["page_start"] for c in chunks if re.search(r"\(\d+\.\d+[a-z]?\)", c["text"])]
        image_pages = list(dict.fromkeys(caption_pages + eq_pages))[:2]
        lesson = {
            "chapter": chapter.number, "section_id": sid, "section_title": stitle,
            "section_idx": ptr["section_idx"], "part": ptr["part"], "parts_total": len(lessons),
            "sections_total": len(seq), "first_lesson": ptr["section_idx"] == 0 and ptr["part"] == 0,
            "roadmap": [t for _, t in seq],
            "subsections": next((s.subsections for s in chapter.sections if s.id == sid), []),
        }
        return {"hits": chunks, "context": context, "sources": sources, "image_pages": image_pages,
                "low_context": False, "lesson": lesson}

    # ---- Node 3: Teaching Planner (deterministic) ----
    def node_plan(self, state: AgentState) -> dict:
        confident = [h for h in state.get("hits", []) if h.get("_score", 1.0) >= self.cfg.min_relevance]
        eq_text = "\n".join(h["text"] for h in confident)
        equations = sorted(set(re.findall(r"\((\d+\.\d+[a-z]?)\)", eq_text)),
                           key=lambda e: [int(x) if x.isdigit() else 0 for x in re.split(r"[.a-z]", e) if x])
        figures = [h["figure_labels"] for h in state.get("hits", []) if h.get("chunk_type") == "figure_caption"]
        sections = list(dict.fromkeys(s["section_title"] for s in state.get("sources", []) if s.get("section_title")))
        intent, followup = state["intent"], state.get("followup", "")
        needs_example = intent in ("concept", "chapter_teaching", "numerical") or followup == "example"
        lines = ["TEACHING PLAN",
                 f"- Student request: {state.get('query') or '(button) ' + INTENT_LABELS.get(intent, intent)}",
                 f"- Task type: {INTENT_LABELS.get(intent, intent)}" + (f" (follow-up: {followup})" if followup else ""),
                 f"- Explanation level: {state.get('difficulty', 'Intermediate')}",
                 f"- Textbook sections in the passages: {'; '.join(sections[:8]) or 'none'}",
                 f"- Numbered equations present in the passages: {', '.join(equations[:15]) or 'none'}",
                 f"- Figure captions present: {', '.join('Figure ' + f for f in figures[:10]) or 'none'}",
                 f"- Page images attached: {', '.join('PDF p. ' + str(p) for p in state.get('image_pages', [])) or 'none'}",
                 f"- Include worked/real-world example: {'yes' if needs_example else 'only if it helps'}"]
        if state.get("low_context"):
            lines.append("- WARNING: no closely matching textbook passage was found. Say so explicitly.")
        lesson = state.get("lesson") or {}
        if lesson.get("section_id"):
            lines.append(f"- Lesson: section {lesson['section_idx'] + 1} of {lesson['sections_total']} "
                         f"(\"{lesson['section_title']}\"), part {lesson['part'] + 1} of {lesson['parts_total']}")
            if lesson.get("subsections"):
                lines.append(f"- Sub-sections of this section: {'; '.join(lesson['subsections'])}")
        plan = {"equations": equations, "figures": figures, "sections": sections,
                "needs_example": needs_example}
        return {"plan": plan, "plan_text": "\n".join(lines)}

    # ---- Node 4: Teaching Agent ----
    def _images(self, pages: list[int]) -> tuple[list[Any], list[int]]:
        from google.genai import types
        parts, ok = [], []
        for p in pages[:2]:
            png = render_page_png(self.book, p)
            if png:
                parts.append(f"[Attached image: rendered PDF page {p}]")
                parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))
                ok.append(p)
        return parts, ok

    def node_teach(self, state: AgentState) -> dict:
        llm = self._require_llm()
        intent, followup = state["intent"], state.get("followup", "")
        image_parts, shown = self._images(state.get("image_pages", []))
        template = MODE_INSTRUCTIONS.get(intent, MODE_INSTRUCTIONS["general"])
        lesson = state.get("lesson") or {}
        if intent == "chapter_teaching":
            if lesson.get("first_lesson"):
                road = "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(lesson.get("roadmap", [])))
                roadmap = ("This is the FIRST lesson of the chapter: begin with a short '🗺️ Chapter roadmap' "
                           f"listing these sections in order, then teach the lesson:\n{road}")
            else:
                roadmap = "Do not repeat the chapter roadmap; continue the course naturally from the previous lesson."
            template = template.format(roadmap_instruction=roadmap)
        elif intent == "diagram":
            img = (f"The rendered PDF page(s) {', '.join(map(str, shown))} are attached as images. Describe only "
                   "what is actually visible; if the figure the student means is not on the attached page, say so."
                   if shown else
                   "No page image is available. Explain only from the caption/passages and state clearly that "
                   "you could not inspect the diagram itself.")
            template = template.format(image_instruction=img)
        parts = [template]
        if followup:
            parts.append("FOLLOW-UP: " + FOLLOWUP_INSTRUCTIONS[followup])
        parts.append(state.get("plan_text", ""))
        hist = _history_text(state.get("history", []), self.cfg.history_turns, followup)
        if hist:
            parts.append("CONVERSATION SO FAR (most recent last):\n" + hist)
        parts.append("TEXTBOOK PASSAGES:\n" + (state.get("context") or "(no relevant passages found)"))
        parts.append("STUDENT'S MESSAGE:\n" + (state.get("query") or "(pressed the button for this task)"))
        system = system_prompt(self.book.name, state.get("difficulty", "Intermediate"), self.outline)
        contents = ["\n\n".join(parts), *image_parts]
        if self.on_token and hasattr(llm, "generate_stream"):
            draft = llm.generate_stream(system, contents, self.on_token)
        else:
            draft = llm.generate(system, contents)
        return {"draft": draft, "image_pages": shown}

    # ---- Node 5: Answer Verifier ----
    def node_verify(self, state: AgentState) -> dict:
        tags = {s["tag"] for s in state.get("sources", [])}
        answer = clean_answer(state.get("draft", ""), tags)
        warnings = list(state.get("warnings", []))
        report: dict = {"checked_by_llm": False}
        allowed_pages = {p for s in state.get("sources", []) for p in range(s["page_start"], s["page_end"] + 1)}
        allowed_pages |= set(state.get("image_pages", []))
        bad_pages = sorted({int(p) for p in re.findall(r"\bPDF\s+p+\.\s*(\d+)", answer)} - allowed_pages)
        if bad_pages:
            warnings.append(f"The answer mentions PDF page(s) {bad_pages} that were not among the retrieved sources.")
        if answer.count("$$") % 2:
            answer += "\n$$"
        if state.get("low_context") and "beyond the textbook" not in answer.lower():
            answer = ("> ⚠️ The retrieved textbook passages do not cover this question directly; the explanation "
                      "below is general knowledge, not taken from the book.\n\n" + answer)
        if state.get("verify") and self.llm is not None and state.get("context"):
            try:
                prompt = VERIFY_PROMPT.format(question=state.get("query") or INTENT_LABELS.get(state["intent"], ""),
                                              context=state["context"], draft=answer)
                res: VerificationResult = self.llm.generate(
                    "You verify textbook-grounded teaching answers for accuracy.", [prompt],
                    schema=VerificationResult, temperature=0.1)
                report = {"checked_by_llm": True, **res.model_dump(exclude={"revised_answer"})}
                if res.verdict == "revise" and len(res.revised_answer) >= 0.6 * len(answer):
                    answer = clean_answer(res.revised_answer, tags)
                    report["revised"] = True
            except LLMError as exc:
                warnings.append(f"Verification step skipped: {exc}")
        return {"answer": answer, "verification": report, "warnings": warnings}

    # ---- Node 6: Response Formatter ----
    def node_format(self, state: AgentState) -> dict:
        answer = state.get("answer", "")
        sources = state.get("sources", [])
        cited = set(re.findall(r"\[(S\d+)\]", answer))
        shown = [s for s in sources if s["tag"] in cited]
        if not shown and not state.get("low_context"):
            shown = sources[:3]
        blocks = []
        if shown:
            lines = ["**📚 Textbook sources**"]
            for s in shown:
                note = " *(your notes, not the textbook)*" if s.get("source_type") == "notes" else ""
                lines.append(f"- **[{s['tag']}]** {s['label']}{note}")
            blocks.append("\n".join(lines))
        if state.get("image_pages"):
            blocks.append("🖼️ Page image(s) read: " + ", ".join(f"PDF p. {p}" for p in state["image_pages"]))
        rec_md, suggestions = self._recommendations(state)
        if rec_md:
            blocks.append(rec_md)
        if blocks:
            answer = answer + "\n\n---\n" + "\n\n".join(blocks)
        return {"answer": answer, "sources": shown, "suggestions": suggestions}

    def _recommendations(self, state: AgentState) -> tuple[str, list[dict]]:
        """Smart learning recommendations computed from the book structure (no LLM call)."""
        srcs = [s for s in state.get("sources", []) if s.get("source_type") == "textbook" and s["chapter_no"] > 0]
        lesson = state.get("lesson") or {}
        learning = state.get("learning") or {}
        if not srcs and not lesson:
            return "", []
        ch_no = lesson.get("chapter") or srcs[0]["chapter_no"]
        chapter = self.book.chapter(ch_no)
        if chapter is None:
            return "", []
        seq = chapter_sequence(self.book, chapter)
        ids = [s[0] for s in seq]
        anchor = lesson.get("section_id") or srcs[0]["section_id"]
        if anchor not in ids:
            anchor = next((s["section_id"] for s in srcs if s["section_id"] in ids), "")
        lines, suggestions = [], []
        if anchor in ids:
            i = ids.index(anchor)
            if i > 0 and state["intent"] != "chapter_teaching":
                lines.append(f"Prerequisite: *{seq[i - 1][1]}*")
                suggestions.append({"label": f"Teach: {seq[i - 1][1][:48]}",
                                    "action": {"type": "teach", "op": "jump", "chapter": ch_no,
                                               "section_id": seq[i - 1][0]}})
            if i + 1 < len(seq) and state["intent"] != "chapter_teaching":
                lines.append(f"Next section: *{seq[i + 1][1]}*")
                suggestions.append({"label": f"Teach: {seq[i + 1][1][:48]}",
                                    "action": {"type": "teach", "op": "jump", "chapter": ch_no,
                                               "section_id": seq[i + 1][0]}})
        related = list(dict.fromkeys(s["section_title"] for s in srcs
                                     if s["section_id"] != anchor and s["section_title"]
                                     and s.get("score", 1.0) >= self.cfg.min_relevance))[:2]
        if related:
            lines.append("Related: " + "; ".join(f"*{r}*" for r in related))
        eqs = state.get("plan", {}).get("equations", [])[:5]
        if eqs:
            lines.append("Key equations to know: " + ", ".join(f"({e})" for e in eqs))
        weak = sorted(learning.get("weak_topics", {}).items(), key=lambda kv: -kv[1])[:3]
        if weak:
            lines.append("Topics to revise (from your quizzes): " + ", ".join(t for t, _ in weak))
        md = ("**🧭 Study next** — " + " · ".join(lines)) if lines else ""
        return md, suggestions

    # ---- Quiz generation (Quiz Mode) ----
    def node_quiz(self, state: AgentState) -> dict:
        llm = self._require_llm()
        chapter = self._chapter(state)
        learning = state.get("learning") or new_learning_state()
        n = int((state.get("action") or {}).get("n", 5))
        scope = chapter.label if chapter else (state.get("query") or "the textbook")
        if state.get("query") and self._has_topic(state["query"]):
            scope = f"{state['query']} ({scope})"
        prompt = QUIZ_PROMPT.format(
            n=n, scope=scope, difficulty=state.get("difficulty", "Intermediate"),
            numerical_hint=" and one numerical question if the passages contain a usable formula",
            avoid="; ".join(learning.get("asked_questions", [])[-15:]) or "none")
        prompt += "\n\nTEXTBOOK PASSAGES:\n" + state.get("context", "")
        system = system_prompt(self.book.name, state.get("difficulty", "Intermediate"), self.outline)
        result: QuizSet = llm.generate(system, [prompt], schema=QuizSet, temperature=0.6)
        by_tag = {s["tag"]: s for s in state.get("sources", [])}
        questions = []
        for q in result.questions[:n]:
            q = _normalise_question(q)
            if q is None:
                continue
            src = by_tag.get(q.source_tag.strip("[] "), {})
            q.explanation = re.sub(r"\s*\[S\d+\]", "", q.explanation)
            questions.append({**q.model_dump(), "source_label": src.get("label", ""),
                              "section_title": src.get("section_title", "")})
        if not questions:
            raise LLMError("The model did not return usable quiz questions. Please try again.")
        quiz = {"chapter_no": chapter.number if chapter else None, "scope": scope, "questions": questions,
                "index": 0, "results": [], "finished": False, "started": datetime.now().isoformat(timespec="minutes")}
        msg = (f"📝 **Quiz ready:** {len(questions)} questions on *{scope}*. Answer them one at a time in the "
               f"**Quiz** tab — the correct answer is revealed only after you respond.")
        return {"quiz": quiz, "answer": msg, "sources": []}

    # ---- Node 7: Learning State Manager ----
    def node_update_state(self, state: AgentState) -> dict:
        learning = dict(state.get("learning") or new_learning_state())
        intent = state.get("intent", "")
        lesson = state.get("lesson") or {}
        chapter_no = lesson.get("chapter") or state.get("chapter_no")
        if chapter_no is None and state.get("sources"):
            chapter_no = state["sources"][0]["chapter_no"]
        if chapter_no is not None and chapter_no > 0:
            learning["current_chapter"] = chapter_no
            if chapter_no not in learning["chapters_started"]:
                learning["chapters_started"] = learning["chapters_started"] + [chapter_no]
        topic = learning.get("current_topic", "")
        if lesson.get("section_id"):
            topic = lesson["section_title"]
        elif not state.get("followup") and state.get("query"):
            topic = state["query"].strip()[:90]
        if topic:
            learning["current_topic"] = topic
            learning["recent_topics"] = ([topic] + [t for t in learning["recent_topics"] if t != topic])[:10]
            if topic not in learning["discussed"]:
                learning["discussed"] = (learning["discussed"] + [topic])[-60:]
        if lesson.get("section_id"):
            ptr = {"section_idx": lesson["section_idx"], "part": lesson["part"] + 1}
            if ptr["part"] >= lesson["parts_total"]:
                if lesson["section_id"] not in learning["completed_sections"]:
                    learning["completed_sections"] = learning["completed_sections"] + [lesson["section_id"]]
                ptr = {"section_idx": lesson["section_idx"] + 1, "part": 0}
            learning["lesson_pointer"] = {**learning.get("lesson_pointer", {}), str(lesson["chapter"]): ptr}
            learning["last_lesson"] = {"chapter": lesson["chapter"], "section_idx": lesson["section_idx"],
                                       "part": lesson["part"]}
        elif lesson.get("finished"):
            learning["lesson_pointer"] = {**learning.get("lesson_pointer", {}),
                                          str(lesson["chapter"]): lesson["pointer"]}
        if state.get("quiz"):
            learning["quiz"] = state["quiz"]
            learning["asked_questions"] = (learning.get("asked_questions", []) +
                                           [q["question"][:120] for q in state["quiz"]["questions"]])[-40:]
        if state.get("sources"):
            learning["last_chunk_ids"] = [s["chunk_id"] for s in state["sources"]]
        if intent and intent != "quiz":
            learning["last_intent"] = intent
        learning["difficulty"] = state.get("difficulty", "")
        return {"learning": learning}


# ---------------------------------------------------------------------------
# Quiz grading and report (outside the graph: one question at a time)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9α-ωα-ω.+\-/]", "", s.lower().replace("×", "x"))


def _normalise_question(q: QuizQuestion) -> QuizQuestion | None:
    if not q.question.strip() or not q.correct_answer.strip():
        return None
    if q.qtype == "true_false":
        q.options = ["True", "False"]
        q.correct_answer = "True" if q.correct_answer.strip().lower().startswith(("t", "sahi", "सही")) else "False"
    elif q.qtype == "mcq":
        opts = [o.strip() for o in q.options if o.strip()]
        if len(opts) < 2:
            return None
        letter = re.fullmatch(r"\(?([A-Da-d])\)?[.)]?", q.correct_answer.strip())
        if letter and q.correct_answer.strip() not in opts:
            idx = ord(letter.group(1).upper()) - 65
            if idx < len(opts):
                q.correct_answer = opts[idx]
        if q.correct_answer not in opts:
            best = difflib.get_close_matches(q.correct_answer, opts, n=1, cutoff=0.5)
            if not best:
                return None
            q.correct_answer = best[0]
        q.options = opts
    return q


def grade_answer(llm: GeminiClient | None, question: dict, student: str) -> dict:
    """Grade one answer. MCQ/true-false/fill-blank are graded locally; written answers by the LLM."""
    qtype, correct = question["qtype"], question["correct_answer"]
    student = student.strip()
    if qtype in ("mcq", "true_false"):
        chosen = student
        letter = re.fullmatch(r"\(?([A-Da-d])\)?[.)]?", student)
        if letter and qtype == "mcq" and student not in question["options"]:
            i = ord(letter.group(1).upper()) - 65
            chosen = question["options"][i] if i < len(question["options"]) else student
        ok = _norm(chosen) == _norm(correct)
        return {"score": 1.0 if ok else 0.0, "correct": ok, "feedback": question["explanation"]}
    if qtype == "fill_blank":
        accepted = [correct] + question.get("accepted_answers", [])
        ok = any(_norm(student) == _norm(a) or difflib.SequenceMatcher(None, _norm(student), _norm(a)).ratio() > 0.85
                 for a in accepted if a)
        if ok or llm is None:
            return {"score": 1.0 if ok else 0.0, "correct": ok, "feedback": question["explanation"]}
    if llm is None:
        return {"score": 0.0, "correct": False, "feedback": question["explanation"], "ungraded": True}
    try:
        res: AnswerEvaluation = llm.generate(
            "You are a fair, encouraging examiner.",
            [QUIZ_EVAL_PROMPT.format(qtype=qtype, question=question["question"], correct=correct,
                                     explanation=question["explanation"], student=student)],
            schema=AnswerEvaluation, temperature=0.1)
        score = max(0.0, min(1.0, float(res.score)))
        return {"score": score, "correct": res.is_correct or score >= 0.7, "feedback": res.feedback}
    except LLMError as exc:
        return {"score": 0.0, "correct": False, "ungraded": True,
                "feedback": f"(Could not grade automatically: {exc})\n\n{question['explanation']}"}


def record_quiz_answer(learning: dict, result: dict, student: str) -> dict:
    quiz = learning["quiz"]
    q = quiz["questions"][quiz["index"]]
    quiz["results"].append({**result, "student": student, "index": quiz["index"]})
    if not result["correct"] and not result.get("ungraded"):
        topic = q.get("topic") or q.get("section_title") or "general"
        learning["weak_topics"][topic] = learning["weak_topics"].get(topic, 0) + 1
    quiz["index"] += 1
    if quiz["index"] >= len(quiz["questions"]):
        quiz["finished"] = True
        graded = [r for r in quiz["results"] if not r.get("ungraded")]
        score = sum(r["score"] for r in graded)
        learning["quiz_history"].append({
            "chapter_no": quiz["chapter_no"], "scope": quiz["scope"], "score": round(score, 2),
            "total": len(graded), "date": datetime.now().isoformat(timespec="minutes"),
            "missed_topics": [quiz["questions"][r["index"]]["topic"] for r in graded if not r["correct"]]})
    return learning


def quiz_report(quiz: dict) -> str:
    graded = [r for r in quiz["results"] if not r.get("ungraded")]
    total = len(graded) or 1
    score = sum(r["score"] for r in graded)
    pct = 100 * score / total
    lines = [f"### 📊 Quiz report — {quiz['scope']}",
             f"**Score: {score:g} / {len(graded)} ({pct:.0f}%)**", ""]
    by_type: dict[str, list[float]] = {}
    for r in graded:
        by_type.setdefault(quiz["questions"][r["index"]]["qtype"], []).append(r["score"])
    lines.append("| Question type | Score |\n| --- | --- |")
    for t, sc in by_type.items():
        lines.append(f"| {t.replace('_', ' ')} | {sum(sc):g} / {len(sc)} |")
    missed = [quiz["questions"][r["index"]] for r in graded if not r["correct"]]
    if missed:
        lines += ["", "**Weak topics:** " + ", ".join(dict.fromkeys(q["topic"] for q in missed))]
        secs = list(dict.fromkeys(q["source_label"] for q in missed if q.get("source_label")))
        if secs:
            lines += ["", "**Revise these sections:**", ""] + [f"- {s}" for s in secs]
    verdict = ("Excellent — you are exam-ready on this material." if pct >= 85 else
               "Good work — revise the weak topics once more." if pct >= 60 else
               "Revise this chapter (try Quick Revision mode), then take the quiz again.")
    lines += ["", f"**Recommendation:** {verdict}"]
    if any(r.get("ungraded") for r in quiz["results"]):
        lines.append("\n*Some written answers could not be graded automatically and are excluded from the score.*")
    return "\n".join(lines)
