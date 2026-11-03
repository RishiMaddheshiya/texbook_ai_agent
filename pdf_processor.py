"""PDF ingestion: validation, text extraction with symbol-font repair, chapter/section detection,
figure/table discovery and semantic chunking. Results are cached on disk per PDF hash."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pymupdf

from config import settings

ProgressCB = Callable[[float, str], None]


class PDFProcessingError(Exception):
    """Raised with a message that is safe to show to the student."""


# ---------------------------------------------------------------------------
# Symbol-font repair
# ---------------------------------------------------------------------------
# Many publisher PDFs draw Greek letters and math operators with private-encoding fonts, so plain
# extraction yields "l ¼ h=p" instead of "λ = h/p". These maps were built by rendering each glyph
# of the Springer/Advent fonts and reading it; the standard Adobe Symbol map covers most other books.
_SYMBOL_STD = dict(zip(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIKLMNOPQRSTUWXYZ",
    "αβχδεφγηιϕκλμνοπθρστυϖωξψζΑΒΧΔΕΦΓΗΙΚΛΜΝΟΠΘΡΣΤΥΩΞΨΖ",
))
_FONT_MAPS: dict[str, dict[str, str]] = {
    "AdvPi1": {
        "a": "α", "b": "β", "c": "ψ", "d": "δ", "e": "ε", "f": "φ", "g": "γ", "j": "φ", "l": "λ",
        "m": "μ", "n": "ν", "o": "ω", "p": "π", "q": "∂", "r": "ρ", "s": "σ", "t": "τ", "w": "χ",
        "x": "ξ", "y": "θ", "z": "ζ", "Z": "η", "D": "Δ", "F": "Φ", "C": "Ψ", "G": "Γ", "O": "Ω",
        "S": "Σ", "8": "°", "5": "⟨", "4": "⟩", "^": "≃",
    },
    "AdvSymb": {**_SYMBOL_STD, "„": "≠", "\x02": "°"},
    "AdvP4C4E74": {
        "\x02": "∼", "\x03": "×", "¼": "=", "ð": "(", "Þ": ")", "þ": "+", "\x04": "−", "\x05": "±",
        "\x06": "×", "½": "[", "\x07": "]", "j": "|", "1": "∞", "0": "′", "6": "≠", "^": "∧",
        "/": "∝", "f": "{", "g": "}", "p": "√", "r": "∇", "\x08": "±", "\t": "≤", "\n": "≈",
        "\x0b": "×", "ﬃ": "≅", "!": "→", ")": "⇒", "\x0c": "≥", "’": "≃", "h": "⟨", "i": "⟩",
        "?": "⊥", "\r": "×",
    },
    "AdvP4C4E51": {":": ".", "=": "/", ";": ",", "@": "∂"},
    "AdvP4C4E59": {"\x02": "̅", ";": ","},
    "AdvP4C4E46": {
        **{c: "[" for c in "\x02\x04\"h246"}, **{c: "]" for c in "\x03\x05#i357\x0b"},
        **{c: "(" for c in "0@\x06B!\x08t"}, **{c: ")" for c in "1A\x07C\t"},
        **{c: "{" for c in "n\n8:("}, **{c: "}" for c in "o9=)"},
        **{c: "∑" for c in "XP"}, **{c: "∫" for c in "ðÐR><;v"}, **{c: "√" for c in "ﬃpqsu"},
    },
    "AdvPi3": {"&": "•", "1": "®"},
    "AdvTpi5": {"b": "≡"},
    "AdvP8034": {"l": "ℓ"},
}
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f]")


def _font_map(font: str) -> dict[str, str] | None:
    name = font.split("+", 1)[-1]  # drop subset prefix "ABCDEF+"
    if name in _FONT_MAPS:
        return _FONT_MAPS[name]
    base = name.split("-", 1)[0]
    if base in _FONT_MAPS:
        return _FONT_MAPS[base]
    if "symbol" in name.lower():  # legacy Symbol font without a ToUnicode table
        return _SYMBOL_STD
    return None


def repair_span_text(text: str, font: str) -> str:
    fmap = _font_map(font)
    if fmap:
        text = "".join(fmap.get(ch, ch) for ch in text)
    return _CTRL.sub("", text)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Block:
    text: str
    kind: str  # para | heading | caption | equation
    page: int  # 1-based PDF page
    y0: float = 0.0


@dataclass
class PageContent:
    page: int
    blocks: list[Block]
    has_images: bool
    is_scanned: bool
    figure_labels: list[str]
    table_labels: list[str]


@dataclass
class Section:
    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    kind: str = "core"  # core | other | questions | references
    subsections: list[str] = field(default_factory=list)


@dataclass
class Chapter:
    number: int
    title: str
    page_start: int
    page_end: int
    part: str = ""
    sections: list[Section] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"Chapter {self.number}: {self.title}"

    def teaching_sections(self) -> list[Section]:
        return [s for s in self.sections if s.kind in ("core", "other")]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    book_id: str
    source_type: str  # textbook | notes
    source_name: str
    chapter_no: int
    chapter_title: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    chunk_type: str  # text | figure_caption | table | questions
    figure_labels: str
    order: int

    def embed_text(self) -> str:
        head = f"{self.chapter_title} | {self.section_title}".strip(" |")
        return f"{head}\n{self.text}" if head else self.text


@dataclass
class ProcessedBook:
    book_id: str
    name: str
    file_name: str
    page_count: int
    detection_method: str
    chapters: list[Chapter]
    figure_index: dict[str, int]
    table_index: dict[str, int]
    image_pages: list[int]
    scanned_pages: list[int]
    chunk_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def folder(self) -> Path:
        return settings.books_dir / self.book_id

    @property
    def pdf_path(self) -> Path:
        return self.folder / "source.pdf"

    def chapter(self, number: int) -> Chapter | None:
        return next((c for c in self.chapters if c.number == number), None)

    def chapter_for_page(self, page: int) -> Chapter | None:
        return next((c for c in self.chapters if c.page_start <= page <= c.page_end), None)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def book_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def validate_pdf(data: bytes, file_name: str = "") -> pymupdf.Document:
    if not data:
        raise PDFProcessingError("The uploaded file is empty.")
    if len(data) > settings.max_pdf_mb * 1024 * 1024:
        raise PDFProcessingError(f"The file is larger than {settings.max_pdf_mb} MB.")
    if not data[:1024].lstrip().startswith(b"%PDF"):
        raise PDFProcessingError(f"'{file_name or 'file'}' is not a valid PDF (missing %PDF header).")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # corrupt xref, truncated file, ...
        raise PDFProcessingError(f"The PDF could not be opened: {exc}") from exc
    if doc.needs_pass:
        raise PDFProcessingError("The PDF is password-protected. Please upload an unlocked copy.")
    if doc.page_count == 0:
        raise PDFProcessingError("The PDF has no pages.")
    return doc


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------
_CAPTION_RE = re.compile(r"^(?:FIGURE|Figure|FIG\.|Fig\.)\s*(\d+[.\-]\d+[a-z]?)")
_TABLE_RE = re.compile(r"^(?:TABLE|Table)\s+(\d+[.\-]\d+)")
_FOOTER_RE = re.compile(r"\.{8,}")
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.I)
_MATHY = set("=±×∼≈≤≥∝∇√∫∑∂αβγδεζηθλμνπρστφχψωΔΦΨΩΓΣ′^_")


def _line_text(line: dict) -> str:
    spans = [s for s in line["spans"] if s["text"]]
    if not spans:
        return ""
    main = max(spans, key=lambda s: (s["size"], len(s["text"].strip())))
    base_size, base_y = main["size"], main["origin"][1]
    tokens: list[list[str]] = []  # [kind, text] with kind in {"", "^", "_"}
    for s in spans:
        txt = repair_span_text(s["text"], s["font"])
        if not txt:
            continue
        kind = ""
        if s["size"] < 0.82 * base_size and txt.strip():
            dy = s["origin"][1] - base_y
            if s["flags"] & 1 or dy < -0.12 * base_size:
                kind = "^"
            elif dy > 0.12 * base_size:
                kind = "_"
            txt = txt.strip()
        if tokens and tokens[-1][0] == kind:
            tokens[-1][1] += txt
        else:
            tokens.append([kind, txt])
    out = []
    for kind, txt in tokens:
        if kind:
            txt = txt if len(txt) == 1 else "{" + txt + "}"
        out.append(kind + txt)
    return "".join(out).strip()


def _join_lines(lines: list[str]) -> str:
    text = ""
    for ln in lines:
        if not ln:
            continue
        if text.endswith("-") and ln[:1].islower() and len(text) > 1 and text[-2].isalpha():
            text = text[:-1] + ln  # de-hyphenate "elec-" + "tron"
        elif text:
            text += " " + ln
        else:
            text = ln
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _is_equation(text: str) -> bool:
    if len(text) > 90:
        return False
    mathy = sum(ch in _MATHY for ch in text)
    words = len(re.findall(r"[A-Za-z]{4,}", text))
    return mathy >= 1 and words <= 3 or bool(re.search(r"\(\d+\.\d+[a-z]?\)\s*$", text)) and words <= 3


def extract_page(page: pymupdf.Page, body_size: float | None = None) -> PageContent:
    pno = page.number + 1
    height = page.rect.height
    d = page.get_text("dict")
    has_images = False
    raw: list[tuple[str, tuple, float, bool]] = []  # text, bbox, max span size, bold
    for b in d["blocks"]:
        if b["type"] == 1:
            has_images = True
            continue
        lines = [_line_text(ln) for ln in b["lines"]]
        text = _join_lines(lines)
        if not text:
            continue
        sizes = [s["size"] for ln in b["lines"] for s in ln["spans"] if s["text"].strip()]
        bold = all(("bold" in s["font"].lower() or "-b" in s["font"].lower() or s["flags"] & 16)
                   for ln in b["lines"] for s in ln["spans"] if s["text"].strip())
        y0, y1 = b["bbox"][1], b["bbox"][3]
        # running headers/footers: dot leaders, bare page numbers in the top/bottom margin
        in_margin = y1 < height * 0.06 or y0 > height * 0.92
        if _FOOTER_RE.search(text) or (in_margin and (_PAGE_NUM_RE.match(text) or len(text) < 60)):
            continue
        raw.append((text, b["bbox"], max(sizes) if sizes else 0.0, bold))
    if not has_images and page.get_images():
        has_images = True

    # merge display-equation fragments (numerator / denominator / equation number) that overlap vertically
    merged: list[tuple[str, tuple, float, bool]] = []
    for item in raw:
        text, bbox, size, bold = item
        if merged:
            ptext, pbox, psize, pbold = merged[-1]
            v_overlap = min(pbox[3], bbox[3]) - max(pbox[1], bbox[1])
            if (v_overlap > 0 and len(text) < 60 and len(ptext) < 120
                    and (_is_equation(text) or _is_equation(ptext))):
                below = bbox[1] > pbox[1] + 3 and bbox[0] < pbox[2]
                joined = f"{ptext} / {text}" if below else f"{ptext} {text}"
                merged[-1] = (joined, (min(pbox[0], bbox[0]), min(pbox[1], bbox[1]),
                                       max(pbox[2], bbox[2]), max(pbox[3], bbox[3])), max(psize, size), pbold)
                continue
        merged.append(item)

    blocks: list[Block] = []
    figs, tabs = [], []
    for text, bbox, size, bold in merged:
        kind = "para"
        m = _CAPTION_RE.match(text)
        if m:
            kind = "caption"
            figs.append(m.group(1).replace("-", "."))
        elif _TABLE_RE.match(text):
            tabs.append(_TABLE_RE.match(text).group(1).replace("-", "."))
        elif _is_equation(text):
            kind = "equation"
        elif len(text) < 120 and (bold or (body_size and size >= body_size * 1.15)) and not text.endswith("."):
            kind = "heading"
        blocks.append(Block(text=text, kind=kind, page=pno, y0=bbox[1]))
    text_len = sum(len(b.text) for b in blocks)
    return PageContent(page=pno, blocks=blocks, has_images=has_images,
                       is_scanned=has_images and text_len < 40,
                       figure_labels=figs, table_labels=tabs)


def _body_font_size(doc: pymupdf.Document) -> float:
    sizes: Counter = Counter()
    n = doc.page_count
    for i in range(0, n, max(1, n // 25)):
        for b in doc[i].get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for s in ln["spans"]:
                    sizes[round(s["size"], 1)] += len(s["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------
_NUM_PREFIX = re.compile(r"^\s*(\d+)\.(\d+)")
_CH_HEADING = re.compile(r"^\s*(?:chapter|unit|lesson)\s+(\d+|[IVXLC]+)\b[\s:.\-–]*(.*)$", re.I)
_QUESTIONS = re.compile(r"question|exercise|problem", re.I)
_REFERENCES = re.compile(
    r"reference|\burls?\b|journal|\bbooks?\b|proceeding|bibliograph|further reading|web ?site|sources of|"
    r"companion text|software", re.I)
_SKIP_TOP = re.compile(r"^(index|contents|preface|foreword|acknowledg|about the|appendix|glossary|"
                       r"list of|figure credits|bibliography|references)", re.I)


def _roman(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = vals.get(ch, 0)
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def _section_kind(title: str, numbered: bool, after_backmatter: bool, after_numbered: bool = False) -> str:
    if _QUESTIONS.search(title):
        return "questions"
    if numbered:
        return "core"
    # unnumbered entries after the numbered body of a chapter are end-matter lists in most textbooks
    if after_backmatter or after_numbered or _REFERENCES.search(title):
        return "references"
    return "other"


def _chapters_from_toc(doc: pymupdf.Document) -> list[Chapter]:
    toc = [t for t in doc.get_toc(simple=True) if t[2] >= 1]
    if len(toc) < 3:
        return []
    top = min(t[0] for t in toc)
    # The level that holds chapters is the shallowest level whose entries have numbered children
    # ("1.1 ...") or are titled "Chapter N".
    candidates: list[tuple[int, str, int, list, str]] = []
    current_part = ""
    for idx, (lvl, title, page) in enumerate(toc):
        if lvl != top and lvl != top + 1:
            continue
        children = []
        for lvl2, t2, p2 in toc[idx + 1:]:
            if lvl2 <= lvl:
                break
            children.append((lvl2, t2, p2))
        if lvl == top and re.match(r"^\s*part\b", title, re.I):
            current_part = title.strip()
            continue
        candidates.append((lvl, title.strip(), page, children, current_part))

    chapters: list[Chapter] = []
    used_numbers: set[int] = set()
    levels: dict[int, int] = {}
    for lvl, title, page, children, part in candidates:
        if _SKIP_TOP.match(title):
            continue
        number = None
        m = _CH_HEADING.match(title)
        if m:
            g = m.group(1)
            number = int(g) if g.isdigit() else _roman(g)
            title = m.group(2).strip() or title
        else:
            nums = [int(_NUM_PREFIX.match(c[1]).group(1)) for c in children if _NUM_PREFIX.match(c[1])]
            if nums:
                number = Counter(nums).most_common(1)[0][0]
            else:
                lead = re.match(r"^\s*(\d+)[\s.:]+(.+)$", title)
                if lead:
                    number, title = int(lead.group(1)), lead.group(2).strip()
        if number is None or number in used_numbers:
            continue
        # skip a level-`top` container (e.g. the book title) whose children are themselves chapters
        if lvl == top and any(_CH_HEADING.match(c[1]) for c in children):
            continue
        used_numbers.add(number)
        levels[number] = lvl
        ch = Chapter(number=number, title=title, page_start=page, page_end=page, part=part)
        child_level = min((c[0] for c in children), default=lvl + 1)
        after_back = False
        seen_numbered = False
        sec_i = 0
        for c_lvl, c_title, c_page in children:
            numbered = bool(_NUM_PREFIX.match(c_title))
            if c_lvl == child_level:
                kind = _section_kind(c_title, numbered, after_back, seen_numbered)
                seen_numbered = seen_numbered or numbered
                if kind in ("questions", "references"):
                    after_back = True
                sec_i += 1
                ch.sections.append(Section(id=f"c{number}-s{sec_i}", title=c_title.strip(), level=c_lvl,
                                           page_start=c_page, page_end=c_page, kind=kind))
            elif ch.sections:
                ch.sections[-1].subsections.append(c_title.strip())
        chapters.append(ch)

    if not chapters:
        return []
    # page ranges: a chapter ends where the next TOC entry at its level or above (chapter, part,
    # back matter) begins
    for ch in chapters:
        nxt = [t[2] for t in toc if t[0] <= levels[ch.number] and t[2] > ch.page_start]
        ch.page_end = (nxt[0] - 1) if nxt else doc.page_count
        ch.page_end = max(ch.page_end, ch.page_start)
        for i, s in enumerate(ch.sections):
            s.page_end = ch.sections[i + 1].page_start if i + 1 < len(ch.sections) else ch.page_end
            s.page_end = max(s.page_end, s.page_start)
    chapters.sort(key=lambda c: c.page_start)
    if len(chapters) < 2 and doc.page_count > 60:
        return []  # a single "chapter" covering a large book means the bookmarks are not chapter bookmarks
    return chapters


def _chapters_from_headings(pages: list[PageContent], page_count: int) -> list[Chapter]:
    """Fallback for PDFs without bookmarks: use 'Chapter N' headings and numbered section headings."""
    ch_starts: dict[int, tuple[int, str]] = {}
    sections: list[tuple[int, str, int]] = []  # chapter number, title, page
    for pc in pages:
        for b in pc.blocks:
            if b.kind not in ("heading", "para") or len(b.text) > 110:
                continue
            m = _CH_HEADING.match(b.text)
            if m and b.kind == "heading":
                g = m.group(1)
                n = int(g) if g.isdigit() else _roman(g)
                ch_starts.setdefault(n, (pc.page, m.group(2).strip() or f"Chapter {n}"))
                continue
            m2 = re.match(r"^\s*(\d+)\.(\d+)\s+([A-Z].{2,})$", b.text)
            if m2 and b.kind == "heading":
                sections.append((int(m2.group(1)), b.text.strip(), pc.page))
    if not ch_starts and sections:
        for n, title, page in sections:
            ch_starts.setdefault(n, (page, f"Chapter {n}"))
    if not ch_starts:
        return []
    ordered = sorted(ch_starts.items(), key=lambda kv: kv[1][0])
    chapters = []
    for i, (n, (page, title)) in enumerate(ordered):
        end = ordered[i + 1][1][0] - 1 if i + 1 < len(ordered) else page_count
        ch = Chapter(number=n, title=title, page_start=page, page_end=max(page, end))
        seen = set()
        for k, (sn, st, sp) in enumerate([s for s in sections if s[0] == n and page <= s[2] <= end]):
            if st in seen:
                continue
            seen.add(st)
            ch.sections.append(Section(id=f"c{n}-s{len(ch.sections) + 1}", title=st, level=2,
                                       page_start=sp, page_end=sp))
        for j, s in enumerate(ch.sections):
            s.page_end = ch.sections[j + 1].page_start if j + 1 < len(ch.sections) else ch.page_end
        chapters.append(ch)
    return chapters


def _chapters_by_pages(page_count: int, size: int = 15) -> list[Chapter]:
    chapters = []
    for i, start in enumerate(range(1, page_count + 1, size), start=1):
        end = min(page_count, start + size - 1)
        chapters.append(Chapter(number=i, title=f"Pages {start}–{end}", page_start=start, page_end=end))
    return chapters


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    parts, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > limit:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


@dataclass
class _Unit:
    """A run of blocks belonging to one teaching section (level-2 TOC entry)."""
    chapter: Chapter | None
    section: Section | None
    sub_title: str
    blocks: list[Block]


def _assign_sections(chapter: Chapter, pages: dict[int, PageContent]) -> list[_Unit]:
    entries: list[tuple[Section, str]] = []
    for s in chapter.sections:
        entries.append((s, ""))
        for sub in s.subsections:
            entries.append((s, sub))
    units: list[_Unit] = [_Unit(chapter, None, "", [])]
    ptr = 0
    for pno in range(chapter.page_start, chapter.page_end + 1):
        pc = pages.get(pno)
        if not pc:
            continue
        # headings we expected on an earlier page but never matched: switch at the start of this page
        while ptr < len(entries) and entries[ptr][0].page_start < pno and not _heading_on_page(entries[ptr], pc):
            sec, sub = entries[ptr]
            units.append(_Unit(chapter, sec, sub, []))
            ptr += 1
        for b in pc.blocks:
            if ptr < len(entries):
                sec, sub = entries[ptr]
                key = _squash(sub or sec.title)[:28]
                if pno >= sec.page_start and key and _squash(b.text).startswith(key):
                    units.append(_Unit(chapter, sec, sub, []))
                    ptr += 1
                    continue  # heading text is metadata, not content
            units[-1].blocks.append(b)
    return [u for u in units if u.blocks]


def _heading_on_page(entry: tuple[Section, str], pc: PageContent) -> bool:
    key = _squash(entry[1] or entry[0].title)[:28]
    return bool(key) and any(_squash(b.text).startswith(key) for b in pc.blocks)


def _chunks_from_unit(unit: _Unit, book_id: str, source_type: str, source_name: str,
                      start_order: int) -> list[Chunk]:
    ch, sec = unit.chapter, unit.section
    chapter_no = ch.number if ch else -1
    chapter_title = ch.label if ch else source_name
    sec_title = sec.title if sec else (f"{ch.title} (introduction)" if ch else "")
    if unit.sub_title:
        sec_title = f"{sec_title} › {unit.sub_title}"
    sec_id = sec.id if sec else (f"c{chapter_no}-intro" if ch else "notes")
    base_type = "questions" if sec and sec.kind == "questions" else "text"
    if sec and sec.kind == "references":
        base_type = "references"

    chunks: list[Chunk] = []
    order = start_order

    def emit(text: str, pages: list[int], ctype: str, figs: str = "") -> None:
        nonlocal order
        text = text.strip()
        if len(text) < 25:
            return
        chunks.append(Chunk(
            chunk_id=f"{book_id}-{order:06d}", text=text, book_id=book_id, source_type=source_type,
            source_name=source_name, chapter_no=chapter_no, chapter_title=chapter_title,
            section_id=sec_id, section_title=sec_title, page_start=min(pages), page_end=max(pages),
            chunk_type=ctype, figure_labels=figs, order=order))
        order += 1

    buf: list[Block] = []

    def flush(keep_tail: bool) -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n\n".join(b.text for b in buf)
        pages = [b.page for b in buf]
        for part in _split_long(text, settings.chunk_max_chars):
            emit(part, pages, base_type)
        tail = buf[-1]
        buf = [tail] if keep_tail and len(tail.text) < 350 and tail.kind != "equation" else []

    for b in unit.blocks:
        if b.kind == "caption":
            label = _CAPTION_RE.match(b.text).group(1).replace("-", ".")
            emit(b.text, [b.page], "figure_caption", label)
            continue
        cur_len = sum(len(x.text) for x in buf)
        glued = (b.kind == "equation" or (buf and buf[-1].kind == "equation")
                 or b.text[:6].lower().startswith("where") or (buf and buf[-1].text.endswith(":")))
        if buf and cur_len + len(b.text) > settings.chunk_target_chars and not (
                glued and cur_len + len(b.text) < settings.chunk_max_chars):
            flush(keep_tail=True)
        buf.append(b)
    flush(keep_tail=False)
    return chunks


def _cell_text(page: pymupdf.Page, bbox) -> str:
    if not bbox:
        return ""
    d = page.get_text("dict", clip=pymupdf.Rect(bbox))
    lines = [_line_text(ln) for b in d["blocks"] if b["type"] == 0 for ln in b["lines"]]
    return _join_lines(lines).replace("|", "/")


def _table_chunks(doc: pymupdf.Document, table_pages: dict[int, list[str]], book: ProcessedBook,
                  start_order: int) -> list[Chunk]:
    """Extract tables (PyMuPDF table finder) only on pages whose text announces a table.
    Cell text goes through the same symbol-font repair as body text."""
    chunks: list[Chunk] = []
    order = start_order
    for pno, labels in sorted(table_pages.items()):
        page = doc[pno - 1]
        try:
            found = page.find_tables().tables
        except Exception:
            continue  # tables are a bonus; never fail ingestion because of them
        for t_i, tab in enumerate(found):
            rows = [[_cell_text(page, cell) for cell in row.cells] for row in tab.rows]
            rows = [r for r in rows if any(r)]
            if len(rows) < 2 or max(len(r) for r in rows) < 2:
                continue
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            md = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
            md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
            label = labels[t_i] if t_i < len(labels) else ""
            ch = book.chapter_for_page(pno)
            head = f"TABLE {label} (structured, PDF page {pno})" if label else f"Table on PDF page {pno}"
            chunks.append(Chunk(
                chunk_id=f"{book.book_id}-t{order:05d}", text=(head + "\n" + "\n".join(md))[:4000],
                book_id=book.book_id, source_type="textbook", source_name=book.name,
                chapter_no=ch.number if ch else -1, chapter_title=ch.label if ch else book.name,
                section_id=_section_for_page(ch, pno), section_title=_section_title_for_page(ch, pno),
                page_start=pno, page_end=pno, chunk_type="table", figure_labels=label, order=order))
            order += 1
    return chunks


def _section_for_page(ch: Chapter | None, page: int) -> str:
    if not ch:
        return ""
    sec = [s for s in ch.sections if s.page_start <= page]
    return sec[-1].id if sec else f"c{ch.number}-intro"


def _section_title_for_page(ch: Chapter | None, page: int) -> str:
    if not ch:
        return ""
    sec = [s for s in ch.sections if s.page_start <= page]
    return sec[-1].title if sec else f"{ch.title} (introduction)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _clean_name(file_name: str, doc: pymupdf.Document) -> str:
    title = (doc.metadata or {}).get("title", "").strip()
    toc = doc.get_toc(simple=True)
    if not title and toc and toc[0][0] == 1 and not _SKIP_TOP.match(toc[0][1]) \
            and not re.match(r"^\s*(part|chapter)\b", toc[0][1], re.I) and toc[0][2] <= 5:
        title = toc[0][1].strip()
    if not title:
        title = re.sub(r"[_\-]+", " ", Path(file_name).stem).strip()
    return title[:120] or "Textbook"


def _extract_all(doc: pymupdf.Document, progress: ProgressCB | None, lo: float, hi: float) -> list[PageContent]:
    body = _body_font_size(doc)
    pages = []
    n = doc.page_count
    for i in range(n):
        pages.append(extract_page(doc[i], body))
        if progress and (i % 20 == 0 or i == n - 1):
            progress(lo + (hi - lo) * (i + 1) / n, f"Extracting text: page {i + 1} of {n}")
    return pages


def process_pdf(data: bytes, file_name: str, progress: ProgressCB | None = None) -> tuple[ProcessedBook, list[Chunk]]:
    """Validate, extract and chunk a textbook. Embedding/indexing happens in retrieval.py."""
    doc = validate_pdf(data, file_name)
    book_id = book_id_for(data)
    folder = settings.books_dir / book_id
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = folder / "source.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(data)

    pages = _extract_all(doc, progress, 0.0, 0.75)
    total_text = sum(len(b.text) for p in pages for b in p.blocks)
    scanned = [p.page for p in pages if p.is_scanned]
    if total_text < 200:
        shutil.rmtree(folder, ignore_errors=True)
        raise PDFProcessingError(
            "No extractable text was found. This looks like a fully scanned PDF; please upload a text-based "
            "(OCR'd) copy so the textbook can be searched.")

    if progress:
        progress(0.78, "Detecting chapters and sections")
    method = "bookmarks"
    chapters = _chapters_from_toc(doc)
    if not chapters:
        method = "headings"
        chapters = _chapters_from_headings(pages, doc.page_count)
    if not chapters:
        method = "page ranges"
        chapters = _chapters_by_pages(doc.page_count)

    # A label like "Figure 1.1" can also appear in front matter (figure credits, lists of figures); prefer the
    # occurrence inside the chapter with the matching number, then any occurrence inside a chapter.
    def best_page(label: str, candidates: list[int]) -> int:
        major = label.split(".")[0]
        ch = next((c for c in chapters if str(c.number) == major), None)
        inside = [p for p in candidates if ch and ch.page_start <= p <= ch.page_end]
        in_any = [p for p in candidates if any(c.page_start <= p <= c.page_end for c in chapters)]
        return (inside or in_any or candidates)[0]

    fig_cands: dict[str, list[int]] = {}
    tab_cands: dict[str, list[int]] = {}
    for p in pages:
        for lab in p.figure_labels:
            fig_cands.setdefault(lab, []).append(p.page)
        for lab in p.table_labels:
            tab_cands.setdefault(lab, []).append(p.page)
    figure_index = {lab: best_page(lab, c) for lab, c in fig_cands.items()}
    table_index = {lab: best_page(lab, c) for lab, c in tab_cands.items()}
    table_pages: dict[int, list[str]] = {}
    for lab, pg in table_index.items():
        table_pages.setdefault(pg, []).append(lab)

    book = ProcessedBook(
        book_id=book_id, name=_clean_name(file_name, doc), file_name=file_name, page_count=doc.page_count,
        detection_method=method, chapters=chapters, figure_index=figure_index, table_index=table_index,
        image_pages=[p.page for p in pages if p.has_images or p.figure_labels], scanned_pages=scanned)
    if scanned:
        book.notes.append(f"{len(scanned)} page(s) have no text layer (scanned/image-only). Ask about them by page "
                          f"number and the professor will read the page image.")
    if method != "bookmarks":
        book.notes.append(f"No usable PDF bookmarks, so chapters were detected from {method}.")

    if progress:
        progress(0.82, "Building semantic chunks")
    page_map = {p.page: p for p in pages}
    chunks: list[Chunk] = []
    for ch in chapters:
        for unit in _assign_sections(ch, page_map):
            chunks.extend(_chunks_from_unit(unit, book_id, "textbook", book.name, len(chunks)))
    if progress:
        progress(0.9, f"Extracting tables from {len(table_pages)} page(s)")
    in_chapters = {p for ch in chapters for p in range(ch.page_start, ch.page_end + 1)}
    chunks.extend(_table_chunks(doc, {p: l for p, l in table_pages.items() if p in in_chapters},
                                book, len(chunks)))
    book.chunk_count = len(chunks)
    save_book(book, chunks)
    if progress:
        progress(1.0, "Text processing complete")
    return book, chunks


def process_notes_pdf(data: bytes, file_name: str, book_id: str) -> list[Chunk]:
    """Extra notes/handouts: chunked page by page and tagged as notes of the given textbook."""
    doc = validate_pdf(data, file_name)
    pages = _extract_all(doc, None, 0, 1)
    if sum(len(b.text) for p in pages for b in p.blocks) < 50:
        raise PDFProcessingError(f"No extractable text in '{file_name}'.")
    notes_id = f"{book_id}-n{book_id_for(data)[:6]}"
    chunks: list[Chunk] = []
    for p in pages:
        unit = _Unit(None, None, "", p.blocks)
        for c in _chunks_from_unit(unit, notes_id, "notes", f"Your notes: {file_name}", len(chunks)):
            c.section_title = f"Notes page {p.page}"
            chunks.append(c)
    return chunks


def save_book(book: ProcessedBook, chunks: list[Chunk]) -> None:
    folder = settings.books_dir / book.book_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "book.json").write_text(json.dumps(asdict(book), ensure_ascii=False, indent=1), encoding="utf-8")
    with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_book(book_id: str) -> ProcessedBook | None:
    path = settings.books_dir / book_id / "book.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    chapters = []
    for c in raw.pop("chapters"):
        secs = [Section(**s) for s in c.pop("sections")]
        chapters.append(Chapter(**c, sections=secs))
    return ProcessedBook(**raw, chapters=chapters)


def load_chunks(book_id: str) -> list[Chunk]:
    path = settings.books_dir / book_id / "chunks.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def list_books() -> list[ProcessedBook]:
    books = []
    for folder in sorted(settings.books_dir.glob("*/book.json")):
        b = load_book(folder.parent.name)
        if b:
            books.append(b)
    return books


def render_page_png(book: ProcessedBook, page: int, dpi: int | None = None) -> bytes | None:
    """Render one PDF page (1-based) to PNG bytes for vision input and on-screen display."""
    if not book.pdf_path.exists() or not 1 <= page <= book.page_count:
        return None
    with pymupdf.open(str(book.pdf_path)) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi or settings.page_image_dpi)
        return pix.tobytes("png")
