"""TEXTBOOK AI — premium Streamlit front end for the AI Textbook Professor.

Run:  streamlit run app.py

Only the presentation layer lives here. Teaching, retrieval, PDF processing and Gemini access are the
existing modules (agent.py, retrieval.py, pdf_processor.py, prompts.py, config.py), used unchanged.
Icons: Lucide (ISC licence), embedded as inline SVG so no extra dependency or network request is needed.
"""
from __future__ import annotations

import html
import math
import re
import urllib.parse
from datetime import datetime, timedelta

import plotly.graph_objects as pgo
import streamlit as st
import streamlit.components.v1 as components

from agent import (INTENT_LABELS, LEARNING_MODES, GeminiClient, LLMError, QuizSet, TeachingAgent,
                   chapter_sequence, grade_answer, load_progress, new_learning_state, record_quiz_answer,
                   save_progress)
from config import api_key_problem, settings
from pdf_processor import (PDFProcessingError, ProcessedBook, book_id_for, list_books, load_book, load_chunks,
                           process_notes_pdf, process_pdf, render_page_png)
from retrieval import Embedder, RetrievalError, TextbookIndex

APP_VERSION = "2.0.0"
st.set_page_config(page_title="Textbook AI", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

# =============================================================================================================
# 1. DESIGN SYSTEM — tokens, icons, centralised CSS
# =============================================================================================================
THEMES = {
    "dark": {
        "bg": "#090B10", "bg2": "#10141D", "card": "#151B27", "sidebar": "#0D1119", "input": "#10141D",
        "accent": "#8B7CFF", "accent2": "#42D6C8", "accent-deep": "#6A5AF0", "text": "#F5F7FA", "muted": "#929BAA",
        "border": "rgba(255,255,255,0.08)", "border-strong": "rgba(255,255,255,0.16)",
        "success": "#36C991", "warning": "#F5B94C", "error": "#F16B75",
        "glass": "rgba(21,27,39,0.72)", "hover": "rgba(255,255,255,0.045)",
        "accent-soft": "rgba(139,124,255,0.14)", "accent-border": "rgba(139,124,255,0.45)",
        "accent-glow": "rgba(139,124,255,0.55)", "accent2-soft": "rgba(66,214,200,0.10)",
        "success-soft": "rgba(54,201,145,0.12)", "warning-soft": "rgba(245,185,76,0.12)",
        "error-soft": "rgba(241,107,117,0.12)", "eq-bg": "rgba(139,124,255,0.06)",
        "user-bubble": "linear-gradient(135deg, rgba(139,124,255,0.22), rgba(106,90,240,0.16))",
        "glow1": "rgba(139,124,255,0.16)", "glow2": "rgba(66,214,200,0.10)", "track": "rgba(255,255,255,0.07)",
        "shadow-sm": "0 1px 2px rgba(0,0,0,0.35)", "shadow-md": "0 8px 24px -10px rgba(0,0,0,0.55)",
        "shadow-lg": "0 22px 50px -18px rgba(0,0,0,0.7)", "scheme": "dark",
    },
    "light": {
        "bg": "#F7F8FC", "bg2": "#FFFFFF", "card": "#FFFFFF", "sidebar": "#FFFFFF", "input": "#FFFFFF",
        "accent": "#6856E8", "accent2": "#12A99A", "accent-deep": "#5443D6", "text": "#171923", "muted": "#707789",
        "border": "#E8EAF1", "border-strong": "#D5D9E4",
        "success": "#1E9E6E", "warning": "#B97A0E", "error": "#D64552",
        "glass": "rgba(255,255,255,0.82)", "hover": "rgba(23,25,35,0.04)",
        "accent-soft": "rgba(104,86,232,0.10)", "accent-border": "rgba(104,86,232,0.38)",
        "accent-glow": "rgba(104,86,232,0.45)", "accent2-soft": "rgba(18,169,154,0.08)",
        "success-soft": "rgba(30,158,110,0.10)", "warning-soft": "rgba(185,122,14,0.10)",
        "error-soft": "rgba(214,69,82,0.09)", "eq-bg": "rgba(104,86,232,0.045)",
        "user-bubble": "linear-gradient(135deg, rgba(104,86,232,0.12), rgba(104,86,232,0.07))",
        "glow1": "rgba(104,86,232,0.10)", "glow2": "rgba(18,169,154,0.08)", "track": "rgba(23,25,35,0.07)",
        "shadow-sm": "0 1px 2px rgba(23,25,35,0.06)", "shadow-md": "0 8px 24px -12px rgba(23,25,35,0.18)",
        "shadow-lg": "0 22px 50px -20px rgba(23,25,35,0.25)", "scheme": "light",
    },
}

ICONS = {
    'arrow-left': '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>',
    'arrow-right': '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    'award': '<path d="m15.477 12.89 1.515 8.526a.5.5 0 0 1-.81.47l-3.58-2.687a1 1 0 0 0-1.197 0l-3.586 2.686a.5.5 0 0 1-.81-.469l1.514-8.526"/><circle cx="12" cy="8" r="6"/>',
    'book-marked': '<path d="M10 2v7.751a.25.25 0 00.407.195l2.28-1.834a.5.5 0 01.627 0l2.28 1.834A.25.25 0 0016 9.751V2"/><path d="M4 19.5v-15A2.5 2.5 0 016.5 2H19a1 1 0 011 1v18a1 1 0 01-1 1H6.5a1 1 0 010-5H20"/>',
    'book-open': '<path d="M12 5v16"/><path d="M20.001 19A2 2 0 0022 17V5a2 2 0 00-1.999-2L16 3.002A5 5 0 0012 5a5 5 0 00-4-2H4a2 2 0 00-2 2v12a2 2 0 001.999 2H8a5 5 0 014 2 5 5 0 014-2z"/>',
    'bot': '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
    'brain': '<path d="M12 18V5"/><path d="M15 13a4.17 4.17 0 01-3-4 4.17 4.17 0 01-3 4"/><path d="M17.707 6.294A3 3 0 1012 5a3 3 0 10-5.707 1.294"/><path d="M17.996 5.125a4 4 0 012.526 5.77"/><path d="M18 18a4 4 0 002-7.465"/><path d="M19.967 17.482a4 4 0 11-7.967.517 4 4 0 11-7.967-.517"/><path d="M6 18a4 4 0 01-2-7.465"/><path d="M6.004 5.125a4 4 0 00-2.526 5.77"/>',
    'calculator': '<rect width="16" height="20" x="4" y="2" rx="2"/><line x1="8" x2="16" y1="6" y2="6"/><line x1="16" x2="16" y1="14" y2="18"/><path d="M16 10h.01"/><path d="M12 10h.01"/><path d="M8 10h.01"/><path d="M12 14h.01"/><path d="M8 14h.01"/><path d="M12 18h.01"/><path d="M8 18h.01"/>',
    'chart-column': '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    'chart-line': '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="m19 9-5 5-4-4-3 3"/>',
    'check': '<path d="M20 6 9 17l-5-5"/>',
    'chevron-right': '<path d="m9 18 6-6-6-6"/>',
    'circle-check': '<circle cx="12" cy="12" r="10"/><path d="m16 9-5.5 5.5L8 12"/>',
    'circle-help': '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    'circle-x': '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    'clipboard-check': '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="m9 14 2 2 4-4"/>',
    'clock': '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    'copy': '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    'cpu': '<path d="M12 20v2"/><path d="M12 2v2"/><path d="M17 20v2"/><path d="M17 2v2"/><path d="M2 12h2"/><path d="M2 17h2"/><path d="M2 7h2"/><path d="M20 12h2"/><path d="M20 17h2"/><path d="M20 7h2"/><path d="M7 20v2"/><path d="M7 2v2"/><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="8" y="8" width="8" height="8" rx="1"/>',
    'file-text': '<path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    'flame': '<path d="M12 3q1 4 4 6.5t3 5.5a1 1 0 0 1-14 0 5 5 0 0 1 1-3 1 1 0 0 0 5 0c0-2-1.5-3-1.5-5q0-2 2.5-4"/>',
    'flask-conical': '<path d="M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2"/><path d="M6.453 15h11.094"/><path d="M8.5 2h7"/>',
    'gauge': '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    'graduation-cap': '<path d="M21.42 10.922a1 1 0 0 0-.019-1.838L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.832l8.57 3.908a2 2 0 0 0 1.66 0z"/><path d="M22 10v6"/><path d="M6 12.5V16a6 3 0 0 0 12 0v-3.5"/>',
    'history': '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    'image': '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    'key-round': '<path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z"/><circle cx="16.5" cy="7.5" r=".5" fill="currentColor"/>',
    'languages': '<path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/>',
    'layers': '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/><path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"/><path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"/>',
    'layout-dashboard': '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    'layout-grid': '<rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/>',
    'library': '<path d="m16 6 4 14"/><path d="M12 6v14"/><path d="M8 8v12"/><path d="M4 4v16"/>',
    'lightbulb': '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    'list': '<path d="M3 5h.01"/><path d="M3 12h.01"/><path d="M3 19h.01"/><path d="M8 5h13"/><path d="M8 12h13"/><path d="M8 19h13"/>',
    'list-checks': '<path d="M13 5h8"/><path d="M13 12h8"/><path d="M13 19h8"/><path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/>',
    'map': '<path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z"/><path d="M15 5.764v15"/><path d="M9 3.236v15"/>',
    'message-square': '<path d="M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z"/>',
    'moon': '<path d="M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401"/>',
    'notebook-pen': '<path d="M13.4 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-7.4"/><path d="M2 6h4"/><path d="M2 10h4"/><path d="M2 14h4"/><path d="M2 18h4"/><path d="M21.378 5.626a1 1 0 1 0-3.004-3.004l-5.01 5.012a2 2 0 0 0-.506.854l-.837 2.87a.5.5 0 0 0 .62.62l2.87-.837a2 2 0 0 0 .854-.506z"/>',
    'play': '<path d="M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z"/>',
    'plug': '<path d="M12 22v-5"/><path d="M15 8V2"/><path d="M17 8a1 1 0 0 1 1 1v4a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V9a1 1 0 0 1 1-1z"/><path d="M9 8V2"/>',
    'refresh-cw': '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    'rotate-ccw': '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    'scan-eye': '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="1"/><path d="M18.944 12.33a1 1 0 0 0 0-.66 7.5 7.5 0 0 0-13.888 0 1 1 0 0 0 0 .66 7.5 7.5 0 0 0 13.888 0"/>',
    'search': '<path d="m21 21-4.34-4.34"/><circle cx="11" cy="11" r="8"/>',
    'send': '<path d="M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z"/><path d="m21.854 2.147-10.94 10.939"/>',
    'settings': '<path d="M9.671 4.136a2.34 2.34 0 0 1 4.659 0 2.34 2.34 0 0 0 3.319 1.915 2.34 2.34 0 0 1 2.33 4.033 2.34 2.34 0 0 0 0 3.831 2.34 2.34 0 0 1-2.33 4.033 2.34 2.34 0 0 0-3.319 1.915 2.34 2.34 0 0 1-4.659 0 2.34 2.34 0 0 0-3.32-1.915 2.34 2.34 0 0 1-2.33-4.033 2.34 2.34 0 0 0 0-3.831A2.34 2.34 0 0 1 6.35 6.051a2.34 2.34 0 0 0 3.319-1.915"/><circle cx="12" cy="12" r="3"/>',
    'sigma': '<path d="M18 7V5a1 1 0 0 0-1-1H6.5a.5.5 0 0 0-.4.8l4.5 6a2 2 0 0 1 0 2.4l-4.5 6a.5.5 0 0 0 .4.8H17a1 1 0 0 0 1-1v-2"/>',
    'sparkles': '<path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z"/><path d="M20 2v4"/><path d="M22 4h-4"/><circle cx="4" cy="20" r="2"/>',
    'sun': '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    'target': '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    'timer': '<line x1="10" x2="14" y1="2" y2="2"/><line x1="12" x2="15" y1="14" y2="11"/><circle cx="12" cy="14" r="8"/>',
    'trash-2': '<path d="M10 11v6"/><path d="M14 11v6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    'triangle-alert': '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    'trophy': '<path d="M10 14.66V17a1 1 0 0 1-1 1 2 2 0 0 0-2 2v2"/><path d="M14 14.66V17a1 1 0 0 0 1 1 2 2 0 0 1 2 2v2"/><path d="M17.916 10H19.5A2.5 2.5 0 0 0 22 7.5V5a1 1 0 0 0-1-1h-3"/><path d="M4 22h16"/><path d="M6 9a6 6 0 0 0 12 0V3a1 1 0 0 0-1-1H7a1 1 0 0 0-1 1z"/><path d="M6.084 10H4.5A2.5 2.5 0 0 1 2 7.5V5a1 1 0 0 1 1-1h3"/>',
    'upload': '<path d="M12 3v12"/><path d="m17 8-5-5-5 5"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>',
    'wand-sparkles': '<path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72"/><path d="m14 7 3 3"/><path d="M5 6v4"/><path d="M19 14v4"/><path d="M10 2v2"/><path d="M7 8H3"/><path d="M21 16h-4"/><path d="M11 3H9"/>',
    'x': '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    'zap': '<path d="M15.914 4a1.5 1.5 0 00-2.474-1.561l-9 9A1.5 1.5 0 005.5 14h4.002a.5.5 0 01.471.666L8.086 20a1.5 1.5 0 002.475 1.56l9-9A1.5 1.5 0 0018.5 10h-3.997a.5.5 0 01-.472-.667z"/>',
}


def icon(name: str, size: int = 18, cls: str = "ta-ico") -> str:
    return (f'<svg class="{cls}" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{ICONS.get(name, "")}</svg>')


def icon_uri(name: str) -> str:
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
           f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{ICONS.get(name, '')}</svg>")
    return 'url("data:image/svg+xml;utf8,' + urllib.parse.quote(svg.replace('"', "'"), safe="") + '")'


STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root { --font: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
  --font-display: 'Plus Jakarta Sans', 'Inter', -apple-system, 'Segoe UI', sans-serif;
  --r-sm: 10px; --r-md: 14px; --r-lg: 20px; --ease: cubic-bezier(.2,.75,.2,1); }

/* ---------- canvas ---------- */
.stApp { background: var(--bg); color: var(--text); font-family: var(--font); isolation: isolate; color-scheme: var(--scheme); }
.stApp::before { content: ""; position: fixed; inset: -15%; z-index: -1; pointer-events: none;
  background: radial-gradient(38% 32% at 18% 8%, var(--glow1), transparent 70%),
              radial-gradient(30% 28% at 88% 4%, var(--glow2), transparent 70%),
              radial-gradient(30% 30% at 70% 95%, var(--glow1), transparent 75%);
  animation: ta-drift 40s ease-in-out infinite alternate; }
@keyframes ta-drift { to { transform: translate3d(2.5%, 2%, 0) scale(1.04); } }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding: 2rem 2.2rem 5rem; }
section[data-testid="stSidebar"], [data-testid="stMain"], [data-testid="stBottom"] { color: var(--text); }
[data-testid="stMarkdownContainer"] { color: var(--text); font-size: 15px; line-height: 1.65; }
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {
  font-family: var(--font-display); color: var(--text); letter-spacing: -0.01em; }
[data-testid="stMarkdownContainer"] a { color: var(--accent); text-decoration: none; }
[data-testid="stMarkdownContainer"] code { background: var(--bg2); color: var(--accent2); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.1em 0.35em; }
[data-testid="stMarkdownContainer"] hr { border-color: var(--border); }
[data-testid="stWidgetLabel"] p { color: var(--muted) !important; font-size: 13px; font-weight: 500; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }

/* ---------- page transitions ---------- */
[class*="st-key-page_"] { animation: ta-fade-up .45s var(--ease) both; }
@keyframes ta-fade-up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 0.4rem 1rem 1.2rem; }
.ta-brand { display: flex; align-items: center; gap: 12px; padding: 4px 4px 14px; }
.ta-logo { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; color: #fff;
  background: linear-gradient(135deg, var(--accent), var(--accent2)); box-shadow: 0 8px 22px -8px var(--accent-glow);
  animation: ta-logo 6s ease-in-out infinite; }
@keyframes ta-logo { 0%, 100% { transform: rotate(0deg); } 50% { transform: rotate(-8deg) scale(1.04); } }
.ta-brand b { font-family: var(--font-display); font-size: 17px; letter-spacing: .06em; color: var(--text); display: block; }
.ta-brand b { line-height: 1.2; }
.ta-brand small { display: block; white-space: nowrap; line-height: 1.3; margin-top: 3px; font-size: 10px; letter-spacing: .16em; color: var(--muted); font-weight: 600; }
.st-key-navgroup { gap: 0.2rem !important; }
[class*="st-key-nav_"] button { background: transparent !important; border: 1px solid transparent !important;
  color: var(--muted) !important; justify-content: flex-start !important; padding: 0 .9rem 0 2.8rem !important;
  min-height: 42px; position: relative; box-shadow: none !important; transition: background .2s var(--ease), color .2s, transform .2s var(--ease); }
[class*="st-key-nav_"] button > div { justify-content: flex-start; }
[class*="st-key-nav_"] button p { font-size: 14px; font-weight: 500; }
[class*="st-key-nav_"] button::before { content: ""; position: absolute; left: .95rem; top: 50%; width: 18px; height: 18px;
  margin-top: -9px; background-color: currentColor; -webkit-mask: var(--ico) center / contain no-repeat;
  mask: var(--ico) center / contain no-repeat; transition: transform .25s var(--ease); }
[class*="st-key-nav_"] button:hover { background: var(--hover) !important; color: var(--text) !important; transform: translateX(2px); }
[class*="st-key-nav_"] button:hover::before { transform: scale(1.12); }
[class*="st-key-nav_"] button:active { transform: scale(.98); }
.ta-status { border: 1px solid var(--border); border-radius: var(--r-md); padding: 12px 12px 6px; background: var(--glass); margin-top: 10px; }
.ta-status-row { display: flex; align-items: center; gap: 10px; font-size: 12.5px; color: var(--muted); padding: 4px 0 8px; }
.ta-status-row div { color: var(--muted); line-height: 1.35; }
.ta-status-row b { color: var(--text); font-weight: 600; display: block; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 170px; }
.ta-dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 9px; box-shadow: 0 0 0 3px var(--hover); }
.ta-dot.ok { background: var(--success); box-shadow: 0 0 0 3px var(--success-soft); animation: ta-pulse 2.4s infinite; }
.ta-dot.idle { background: var(--accent); } .ta-dot.warn { background: var(--warning); } .ta-dot.bad { background: var(--error); }
@keyframes ta-pulse { 50% { box-shadow: 0 0 0 6px transparent; } }
.ta-version { font-size: 11.5px; color: var(--muted) !important; text-align: center; padding-top: 8px; letter-spacing: .04em; }

/* ---------- buttons ---------- */
.stApp button[data-testid="stBaseButton-secondary"], .stApp button[data-testid="stBaseButton-secondaryFormSubmit"],
.stApp [data-testid="stPopoverButton"], .stApp [data-testid="stDownloadButton"] button {
  background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 12px; min-height: 44px;
  font-weight: 500; box-shadow: var(--shadow-sm);
  transition: transform .18s var(--ease), background .18s, border-color .18s, box-shadow .18s, filter .18s; }
.stApp button[data-testid="stBaseButton-secondary"]:hover, .stApp [data-testid="stPopoverButton"]:hover,
.stApp [data-testid="stDownloadButton"] button:hover, .stApp button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
  border-color: var(--accent-border); background: var(--hover); color: var(--text); transform: translateY(-1px); box-shadow: var(--shadow-md); }
.stApp button[data-testid="stBaseButton-primary"], .stApp button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, var(--accent), var(--accent-deep)); color: #fff; border: 0; border-radius: 12px;
  min-height: 44px; font-weight: 600; letter-spacing: .01em; box-shadow: 0 8px 22px -10px var(--accent-glow);
  transition: transform .18s var(--ease), box-shadow .18s, filter .18s; }
.stApp button[data-testid="stBaseButton-primary"] p, .stApp button[data-testid="stBaseButton-primaryFormSubmit"] p { color: #fff; }
.stApp button[data-testid="stBaseButton-primary"]:hover, .stApp button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  filter: brightness(1.08); transform: translateY(-1px) scale(1.01); box-shadow: 0 12px 28px -10px var(--accent-glow); color: #fff; }
.stApp button[data-testid="stBaseButton-tertiary"] { color: var(--muted); border-radius: 10px; min-height: 40px; padding: 0 .7rem;
  transition: background .18s, color .18s, transform .18s var(--ease); }
.stApp button[data-testid="stBaseButton-tertiary"]:hover { color: var(--text); background: var(--hover); }
.stApp button:active { transform: scale(.97) !important; }
.stApp button:disabled, .stApp button[disabled] { opacity: .42; cursor: not-allowed; transform: none !important;
  box-shadow: none !important; filter: none !important; }
[class*="st-key-ib_"] button { gap: .1rem; justify-content: center; }
[class*="st-key-ib_"] button > div { flex: 0 1 auto !important; width: auto !important; }
[class*="st-key-ib_"] button::before { content: ""; width: 17px; height: 17px; flex: 0 0 17px; margin-right: .45rem;
  background-color: currentColor; -webkit-mask: var(--ico) center / contain no-repeat; mask: var(--ico) center / contain no-repeat;
  transition: transform .25s var(--ease); }
[class*="st-key-ib_"] button:hover::before { transform: translateX(1px) scale(1.08); }

/* ---------- inputs ---------- */
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"], [data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] [data-rac][role="group"], [data-testid="stMultiSelect"] [data-rac][role="group"] {
  background: var(--input) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
  transition: border-color .18s, box-shadow .18s; min-height: 44px; }
[data-testid="stTextInputRootElement"]:hover, [data-testid="stTextAreaRootElement"]:hover,
[data-testid="stSelectbox"] [data-rac][role="group"]:hover { border-color: var(--border-strong) !important; }
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within, [data-testid="stSelectbox"] [data-rac][role="group"]:focus-within {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft); }
[data-testid="stTextInputRootElement"] input, [data-testid="stTextAreaRootElement"] textarea,
[data-testid="stNumberInputContainer"] input, [data-testid="stSelectbox"] input {
  background: transparent !important; color: var(--text) !important; caret-color: var(--accent); font-size: 14.5px; }
.stApp input::placeholder, .stApp textarea::placeholder { color: var(--muted) !important; opacity: .8; }
[data-testid="stSelectbox"] button, [data-testid="stNumberInputContainer"] button { color: var(--muted) !important; background: transparent !important; }
[data-testid="stSelectboxVirtualDropdown"], [data-trigger="ComboBox"] { background: var(--card) !important;
  border: 1px solid var(--border) !important; border-radius: 12px !important; box-shadow: var(--shadow-lg) !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"], [data-trigger="ComboBox"] [role="option"] { color: var(--text) !important; border-radius: 8px; }
[data-trigger="ComboBox"] [role="option"][data-focused="true"], [data-trigger="ComboBox"] [role="option"]:hover { background: var(--accent-soft) !important; }
[data-trigger="ComboBox"] [role="option"][aria-selected="true"] { color: var(--accent) !important; font-weight: 600; }
[data-testid="stButtonGroup"] button { background: var(--card) !important; border: 1px solid var(--border) !important;
  color: var(--muted) !important; min-height: 40px; border-radius: 10px; transition: all .18s var(--ease); }
[data-testid="stButtonGroup"] button:hover { color: var(--text) !important; border-color: var(--border-strong) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--accent-soft) !important; color: var(--text) !important;
  border-color: var(--accent-border) !important; box-shadow: inset 0 0 0 1px var(--accent-border); }
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: var(--text); }
[data-testid="stCheckbox"] label > div:not([data-testid]) { background: var(--track) !important; border: 1px solid var(--border-strong); transition: background .2s; }
[data-testid="stCheckbox"][data-selected="true"] label > div:not([data-testid]) { background: linear-gradient(135deg, var(--accent), var(--accent-deep)) !important; border-color: transparent; }
[data-testid="stCheckbox"] label > div:not([data-testid]) > div { background: #fff !important; }
.ta-sec { display: flex; align-items: center; gap: 8px; font-family: var(--font-display); font-weight: 700; font-size: 16px; color: var(--text); margin: 18px 0 6px; }
.ta-sec svg { color: var(--accent); }
[data-testid="stFileUploaderDropzone"] { background: var(--input) !important; border: 1.5px dashed var(--border-strong) !important;
  border-radius: 16px !important; transition: border-color .2s, background .2s; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; background: var(--accent-soft) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--muted) !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * { color: var(--text); }

/* ---------- containers ---------- */
[data-testid="stExpander"] details { background: var(--card); border: 1px solid var(--border) !important; border-radius: 14px; }
[data-testid="stExpander"] summary { color: var(--text) !important; background: transparent !important; border-radius: 14px; transition: color .18s; }
[data-testid="stExpander"] summary * { color: inherit !important; }
[data-testid="stExpander"] summary:hover { color: var(--accent) !important; }
[data-testid="stExpander"] summary p { font-weight: 500; }
[data-testid="stAlertContainer"] { border-radius: 12px; border: 1px solid var(--border); color: var(--text) !important; background: var(--card) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { background: var(--success-soft) !important; border-color: var(--success); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { background: var(--warning-soft) !important; border-color: var(--warning); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { background: var(--error-soft) !important; border-color: var(--error); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background: var(--accent-soft) !important; border-color: var(--accent-border); }
[data-testid="stCode"] pre { background: var(--bg2) !important; border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stCode"] code { color: var(--text); }
[data-testid="stPopoverBody"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; color: var(--text); }
[data-testid="stProgress"] [data-testid="stProgressBarTrack"] > div { background: var(--track); }
[data-testid="stProgress"] [role="progressbar"] > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--border); }
[data-testid="stForm"] { border: 0; padding: 0; }
.st-key-mobile_js { position: absolute; width: 0; height: 0; overflow: hidden; }

/* ---------- cards & components ---------- */
[class*="st-key-card_"], .st-key-hero, .st-key-panel_left, .st-key-panel_right, [class*="st-key-tile_"] {
  background: var(--glass); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 1.15rem 1.2rem 1.2rem;
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); box-shadow: var(--shadow-sm);
  transition: transform .25s var(--ease), box-shadow .25s var(--ease), border-color .25s; }
[class*="st-key-card_"]:hover { transform: translateY(-3px); border-color: var(--accent-border); box-shadow: var(--shadow-lg); }
[class*="st-key-card_"]:hover .ta-tile-ico { transform: translateY(-2px) scale(1.06); }
.st-key-hero { position: relative; overflow: hidden; padding: 2rem 2rem 1.6rem; border-radius: 24px;
  background: linear-gradient(120deg, var(--accent-soft), transparent 55%), var(--glass); }
.st-key-hero::before, .st-key-hero::after { content: ""; position: absolute; border-radius: 50%; filter: blur(40px); pointer-events: none; }
.st-key-hero::before { width: 320px; height: 320px; right: -60px; top: -120px; background: var(--glow1); animation: ta-orb 14s ease-in-out infinite alternate; }
.st-key-hero::after { width: 240px; height: 240px; right: 180px; bottom: -150px; background: var(--glow2); animation: ta-orb 18s ease-in-out infinite alternate-reverse; }
@keyframes ta-orb { to { transform: translate3d(-30px, 24px, 0) scale(1.1); } }
.st-key-hero > div { position: relative; z-index: 1; }
.ta-kicker { font-size: 12px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); }
.ta-h1 { font-family: var(--font-display); font-size: 36px; font-weight: 800; letter-spacing: -0.025em; line-height: 1.15; margin: 6px 0 6px; color: var(--text); }
.ta-h2 { font-family: var(--font-display); font-size: 23px; font-weight: 700; letter-spacing: -0.015em; margin: 4px 0 2px; color: var(--text); }
.ta-h3 { font-family: var(--font-display); font-size: 17px; font-weight: 700; color: var(--text); margin: 0; }
.ta-sub { color: var(--muted); font-size: 15px; margin: 0; }
.ta-small { color: var(--muted); font-size: 12.5px; }
.ta-page-head { display: flex; align-items: center; gap: 14px; margin: 0 0 14px; }
.ta-page-head .ta-tile-ico { width: 46px; height: 46px; border-radius: 14px; }
.ta-section { display: flex; align-items: center; justify-content: space-between; margin: 26px 0 10px; }
.ta-tile-ico { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; color: var(--accent);
  background: var(--accent-soft); border: 1px solid var(--accent-border); transition: transform .25s var(--ease); flex: 0 0 auto; }
.ta-tile-ico.alt { color: var(--accent2); background: var(--accent2-soft); border-color: rgba(66,214,200,.35); }
.ta-stat { display: flex; flex-direction: column; gap: 6px; }
.ta-stat-top { display: flex; align-items: center; justify-content: space-between; }
.ta-stat-val { font-family: var(--font-display); font-size: 30px; font-weight: 800; letter-spacing: -0.02em; color: var(--text); line-height: 1.1;
  animation: ta-fade-up .6s var(--ease) both; }
.ta-stat-label { font-size: 13.5px; color: var(--text); font-weight: 600; }
.ta-stat-hint { font-size: 12px; color: var(--muted); min-height: 17px; }
[class*="st-key-card_s_"] { height: 100%; }
.ta-clamp { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.ta-trend { font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 999px; background: var(--success-soft); color: var(--success); }
.ta-trend.down { background: var(--error-soft); color: var(--error); }
.ta-trend.flat { background: var(--hover); color: var(--muted); }
.ta-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px; margin: 3px 6px 3px 0; border-radius: 999px;
  font-size: 12.5px; color: var(--text); border: 1px solid var(--border); background: var(--card); }
.ta-chip.accent { border-color: var(--accent-border); background: var(--accent-soft); }
.ta-badge { display: inline-block; font-size: 11.5px; font-weight: 600; padding: 3px 9px; border-radius: 999px; letter-spacing: .02em; }
.ta-badge.done { color: var(--success); background: var(--success-soft); }
.ta-badge.prog { color: var(--accent); background: var(--accent-soft); }
.ta-badge.new { color: var(--muted); background: var(--hover); }
.ta-bar { height: 7px; border-radius: 99px; background: var(--track); overflow: hidden; }
.ta-bar > span { display: block; height: 100%; width: var(--w); border-radius: 99px;
  background: linear-gradient(90deg, var(--accent), var(--accent2)); animation: ta-grow 1s var(--ease) both; }
@keyframes ta-grow { from { width: 0; } to { width: var(--w); } }
.ta-ring { position: relative; display: inline-grid; place-items: center; }
.ta-ring svg { transform: rotate(-90deg); }
.ta-ring .fg { stroke: url(#ta-ring-grad); stroke-dasharray: var(--c); stroke-dashoffset: var(--off); animation: ta-ring 1.2s var(--ease) both; }
.ta-ring .bg { stroke: var(--track); }
@keyframes ta-ring { from { stroke-dashoffset: var(--c); } to { stroke-dashoffset: var(--off); } }
.ta-ring b { position: absolute; font-family: var(--font-display); font-weight: 800; color: var(--text); }
.ta-empty { text-align: center; padding: 2.4rem 1.2rem; border: 1px dashed var(--border-strong); border-radius: var(--r-lg); background: var(--glass); }
.ta-empty .ta-tile-ico { margin: 0 auto 12px; width: 52px; height: 52px; border-radius: 16px; }
.ta-empty h4 { font-family: var(--font-display); font-size: 18px; margin: 0 0 4px; color: var(--text); }
.ta-empty p { color: var(--muted); font-size: 14px; margin: 0 auto; max-width: 460px; }
.ta-act { display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid var(--border); align-items: flex-start; }
.ta-act:last-child { border-bottom: 0; }
.ta-act .ta-tile-ico { width: 32px; height: 32px; border-radius: 10px; }
.ta-act b { font-size: 14px; color: var(--text); font-weight: 500; display: block; }
.ta-kv { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }
.ta-kv div { border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; background: var(--card); }
.ta-kv b { display: block; font-family: var(--font-display); font-size: 20px; color: var(--text); }
.ta-kv span { font-size: 12px; color: var(--muted); }
.ta-callout { border-radius: 14px; padding: 14px 16px; border: 1px solid var(--accent-border); background: var(--accent-soft); color: var(--text); }
.ta-answer-final { border-radius: 16px; padding: 16px 18px; border: 1px solid rgba(54,201,145,.45); background: var(--success-soft); }
.ta-answer-final .ta-kicker { color: var(--success); }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { background: transparent !important; padding: .35rem 0; gap: .8rem; max-width: 880px; margin: 0 auto;
  animation: ta-msg .4s var(--ease) both; }
@keyframes ta-msg { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) [data-testid="stChatMessageContent"] {
  background: var(--glass); border: 1px solid var(--border); border-radius: 6px 20px 20px 20px; padding: 1rem 1.3rem 1.1rem;
  box-shadow: var(--shadow-md); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); min-width: 0; }
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) > div:first-child {
  width: 34px; height: 34px; border-radius: 11px; font-size: 0; position: relative; flex: 0 0 34px;
  background: linear-gradient(135deg, var(--accent), var(--accent2)); box-shadow: 0 6px 16px -8px var(--accent-glow); }
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) > div:first-child > * { display: none; }
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) > div:first-child::after {
  content: ""; position: absolute; inset: 8px; background: #fff; -webkit-mask: var(--ico-spark) center / contain no-repeat; mask: var(--ico-spark) center / contain no-repeat; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  flex: 0 1 auto !important; width: auto !important; max-width: 78%; margin-left: auto; background: var(--user-bubble);
  border: 1px solid var(--accent-border); border-radius: 20px 6px 20px 20px; padding: .7rem 1.05rem; }
[data-testid="stChatMessageContent"] h1, [data-testid="stChatMessageContent"] h2 { font-size: 21px; margin-top: .6em; }
[data-testid="stChatMessageContent"] h3 { font-size: 18px; margin: 1.1em 0 .35em; padding-bottom: .3em; border-bottom: 1px solid var(--border); }
[data-testid="stChatMessageContent"] h4 { font-size: 16px; margin: .9em 0 .3em; }
.stApp [data-testid="stMarkdownContainer"] table { border-collapse: separate; border-spacing: 0; border: 1px solid var(--border);
  border-radius: 12px; overflow: hidden; margin: .6rem 0; font-size: 14px; display: block; overflow-x: auto; width: max-content; max-width: 100%; }
.stApp [data-testid="stMarkdownContainer"] th { background: var(--accent-soft); color: var(--text); font-weight: 600; }
.stApp [data-testid="stMarkdownContainer"] th, .stApp [data-testid="stMarkdownContainer"] td { border: 0; border-bottom: 1px solid var(--border); padding: .5rem .8rem; }
.stApp [data-testid="stMarkdownContainer"] blockquote { border-left: 3px solid var(--accent2); background: var(--accent2-soft);
  border-radius: 0 12px 12px 0; padding: .55rem 1rem; margin: .8rem 0; color: var(--text); }
.stApp [data-testid="stMarkdownContainer"] blockquote p { margin: .2rem 0; }
.stApp .katex-display { background: var(--eq-bg); border: 1px solid var(--border); border-radius: 12px; padding: .9rem 1rem;
  margin: .8rem 0; overflow-x: auto; overflow-y: hidden; }
.stApp .katex { color: var(--text); }
.ta-typing { display: inline-flex; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; }
.ta-typing i { display: inline-flex; gap: 4px; }
.ta-typing i span { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); animation: ta-bounce 1.2s infinite ease-in-out; }
.ta-typing i span:nth-child(2) { animation-delay: .15s; background: color-mix(in srgb, var(--accent) 60%, var(--accent2)); }
.ta-typing i span:nth-child(3) { animation-delay: .3s; background: var(--accent2); }
@keyframes ta-bounce { 0%, 80%, 100% { transform: translateY(0); opacity: .45; } 40% { transform: translateY(-5px); opacity: 1; } }
.ta-steps { margin: 10px 0 0; padding: 0; list-style: none; font-size: 12.5px; color: var(--muted); }
.ta-steps li { padding: 2px 0; animation: ta-fade-up .3s var(--ease) both; }
.ta-steps li::before { content: "✓"; color: var(--success); margin-right: 8px; font-weight: 700; }
[data-testid="stBottom"] > div { background: linear-gradient(to top, var(--bg) 72%, transparent) !important; }
[data-testid="stBottomBlockContainer"] { max-width: 920px; padding-top: .6rem; padding-bottom: 1.2rem; }
[data-testid="stChatInput"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 18px !important;
  box-shadow: var(--shadow-md); transition: border-color .2s, box-shadow .2s; }
[data-testid="stChatInput"]:hover { border-color: var(--border-strong) !important; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 4px var(--accent-soft), var(--shadow-md); }
[data-testid="stChatInput"] > div { background: transparent !important; border-color: transparent !important; box-shadow: none !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; background: transparent !important; font-size: 15px; caret-color: var(--accent); }
[data-testid="stChatInputSubmitButton"] { background: linear-gradient(135deg, var(--accent), var(--accent-deep)) !important; color: #fff !important;
  border-radius: 12px !important; min-width: 38px; min-height: 38px; transition: transform .18s var(--ease), filter .18s; }
[data-testid="stChatInputSubmitButton"]:hover { filter: brightness(1.1); transform: scale(1.05); }
[data-testid="stChatInputSubmitButton"]:disabled { background: var(--hover) !important; color: var(--muted) !important; }
[data-testid="stChatInputFileUploadButton"] button { color: var(--muted) !important; }
.st-key-composer_bar { max-width: 880px; margin: 0 auto; }
.st-key-msg_actions [data-testid="stHorizontalBlock"] { gap: .35rem; flex-wrap: wrap; }
.st-key-msg_actions button { min-height: 36px !important; font-size: 13px; }
.st-key-msg_actions button p { font-size: 13px; }

/* ---------- quiz ---------- */
.st-key-quiz_opts [role="radiogroup"] { display: grid; gap: .6rem; width: 100%; }
.st-key-quiz_opts [role="radiogroup"] > div, .st-key-quiz_opts [role="radiogroup"] > label { width: 100%; }
.st-key-quiz_opts [role="radiogroup"] label { width: 100%; border: 1px solid var(--border); border-radius: 14px; padding: .85rem 1rem;
  background: var(--card); min-height: 50px; transition: border-color .18s, background .18s, transform .18s var(--ease); cursor: pointer; }
.st-key-quiz_opts [role="radiogroup"] label:hover { border-color: var(--accent-border); transform: translateY(-1px); }
.st-key-quiz_opts [role="radiogroup"] label[data-selected="true"] { border-color: var(--accent); background: var(--accent-soft); }
.ta-q { font-family: var(--font-display); font-size: 20px; font-weight: 700; line-height: 1.4; color: var(--text); margin: 6px 0 14px; }
.ta-fb { border-radius: 16px; padding: 16px 18px; border: 1px solid; margin-bottom: 12px; }
.ta-fb.ok { border-color: var(--success); background: var(--success-soft); animation: ta-pop .5s var(--ease) both; }
.ta-fb.bad { border-color: var(--error); background: var(--error-soft); animation: ta-shake .45s ease both; }
.ta-fb.neutral { border-color: var(--warning); background: var(--warning-soft); }
.ta-fb b { font-family: var(--font-display); font-size: 18px; }
@keyframes ta-pop { 0% { transform: scale(.96); opacity: 0; } 60% { transform: scale(1.015); opacity: 1; } 100% { transform: none; } }
@keyframes ta-shake { 0%, 100% { transform: none; } 20% { transform: translateX(-6px); } 40% { transform: translateX(5px); } 60% { transform: translateX(-3px); } 80% { transform: translateX(2px); } }

/* ---------- responsive & accessibility ---------- */
@media (max-width: 1024px) { [data-testid="stMainBlockContainer"] { padding: 1.6rem 1.4rem 5rem; } .ta-h1 { font-size: 31px; } }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 1rem .85rem 5.5rem; }
  .ta-h1 { font-size: 26px; } .ta-h2 { font-size: 21px; } .ta-stat-val { font-size: 26px; }
  .st-key-hero { padding: 1.4rem 1.1rem 1.2rem; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] { max-width: 92%; }
  [data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) [data-testid="stChatMessageContent"] { padding: .85rem .95rem; }
  .ta-q { font-size: 18px; }
  .st-key-msg_actions [data-testid="stHorizontalBlock"] { flex-direction: row !important; flex-wrap: wrap !important; }
  .st-key-msg_actions [data-testid="stColumn"] { flex: 1 1 30% !important; min-width: 30% !important; width: auto !important; }
}
@media (hover: none) {
  [class*="st-key-card_"]:hover, .stApp button:hover { transform: none !important; }
  .stApp button:active, [class*="st-key-card_"]:active { transform: scale(.98) !important; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.001ms !important; animation-iteration-count: 1 !important; transition-duration: 0.001ms !important; }
}
"""


def build_css(theme: str, nav_icons: tuple[tuple[str, str], ...], active: str) -> str:
    tokens = THEMES[theme]
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in tokens.items()) + f"--ico-spark:{icon_uri('sparkles')};}}"
    nav = "".join(f".st-key-nav_{pid}{{--ico:{icon_uri(ico)};}}" for pid, ico in nav_icons)
    act = (f".st-key-nav_{active} button{{background:var(--accent-soft)!important;color:var(--text)!important;"
           f"border-color:var(--accent-border)!important;}}"
           f".st-key-nav_{active} button::after{{content:'';position:absolute;left:5px;top:11px;bottom:11px;width:3px;"
           f"border-radius:3px;background:linear-gradient(var(--accent),var(--accent2));animation:ta-fade-up .3s both;}}")
    return f"<style>{root}{STYLES}{nav}{act}</style>"


# =============================================================================================================
# 2. NAVIGATION & STATE
# =============================================================================================================
PAGES = [
    ("overview", "Overview", "layout-dashboard"), ("professor", "AI Professor", "graduation-cap"),
    ("textbook", "My Textbook", "book-open"), ("explorer", "Chapter Explorer", "library"),
    ("numerical", "Numerical Solver", "calculator"), ("diagram", "Diagram Studio", "scan-eye"),
    ("exam", "Exam Preparation", "clipboard-check"), ("quiz", "Quiz Arena", "trophy"),
    ("progress", "Learning Progress", "chart-line"), ("settings", "Settings", "settings"),
]
PAGE_META = {pid: (label, ico) for pid, label, ico in PAGES}
LANGUAGES = ["Hinglish", "English", "Hindi"]
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
QTYPES = {"MCQ": "mcq", "True / False": "true_false", "Fill in the blank": "fill_blank",
          "Conceptual": "conceptual", "Numerical": "numerical"}
STEP_TEXT = {  # what happens *next*, shown while the agent works (real node progress, no fake percentages)
    None: "Understanding your question…", "classify": "Searching your textbook…",
    "retrieve": "Planning the explanation…", "plan": "Your professor is writing…",
    "teach": "Checking the answer against the textbook…", "verify": "Adding references…",
    "format": "Saving your progress…", "quiz": "Saving your quiz…", "update_state": "Done",
}
DONE_TEXT = {"classify": "Understood the request", "retrieve": "Found relevant textbook passages",
             "plan": "Planned the explanation", "teach": "Wrote the explanation", "verify": "Verified against the textbook",
             "format": "Added references", "quiz": "Wrote quiz questions", "update_state": "Updated your progress"}

ss = st.session_state
for k, v in {
    "p_page": "overview", "p_theme": "dark", "p_language": "Hinglish", "p_difficulty": "Intermediate",
    "p_mode": LEARNING_MODES[1], "p_verify": settings.verify_answers, "p_book": None, "p_chapter": None,
    "p_api_key": "", "p_explorer_ch": None, "p_dia_src": "Textbook figure", "p_num_chapter": None,
    "p_exam_view": "Exam pack", "p_quiz_diff": None, "p_quiz_types": None, "p_quiz_n": 5, "chats": {}, "learning": {}, "pending": None, "done_uploads": set(),
    "done_notes": set(), "quiz_feedback": None, "results": {}, "llm_state": ("idle", ""), "goto": None,
}.items():
    ss.setdefault(k, v)


def go(page: str, **extra) -> None:
    """Navigation callback."""
    ss.p_page = page
    ss.nav_clicked = True
    for k, v in extra.items():
        ss[k] = v


def _sync(pref: str, wkey: str) -> None:
    if ss.get(wkey) is not None:
        ss[pref] = ss[wkey]


def bound(fn, label: str, pref: str, wkey: str, **kw):
    """Render a widget whose value lives in a persistent preference key (widget keys are garbage-collected
    when a page is not shown, preferences are not)."""
    ss[wkey] = ss[pref]
    return fn(label, key=wkey, on_change=_sync, args=(pref, wkey), **kw)


ICON_BUTTONS: dict[str, str] = {}


def ibutton(label: str, key: str, ico: str, **kw) -> bool:
    """st.button with a Lucide icon (drawn by CSS via the button's st-key class)."""
    ICON_BUTTONS[key] = ico
    kw.setdefault("width", "stretch")
    return st.button(label, key=f"ib_{key}", **kw)


# =============================================================================================================
# 3. CACHED RESOURCES & BACKEND ADAPTERS
# =============================================================================================================
@st.cache_resource(show_spinner=False)
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner=False)
def get_index(book_id: str) -> TextbookIndex:
    return TextbookIndex(book_id, get_embedder())


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model: str) -> GeminiClient:
    return GeminiClient(settings.with_api_key(api_key))


@st.cache_data(show_spinner=False)
def chapter_equations(book_id: str, chapter_no: int) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    for c in get_index(book_id).chapter_chunks(chapter_no, ["text"]):
        for m in re.finditer(r"([^\n]{0,70})\((\d+\.\d+[a-z]?)\)", c["text"]):
            lab = m.group(2)
            if lab.split(".")[0] == str(chapter_no) and lab not in found:
                found[lab] = m.group(1).strip()[-60:]
    return sorted(found.items(), key=lambda kv: [int(x) if x.isdigit() else 0 for x in re.split(r"[.a-z]", kv[0])])


@st.cache_data(show_spinner=False)
def chapter_intro(book_id: str, chapter_no: int, title: str = "") -> str:
    """Opening text of a chapter (its 'preview' paragraph in most textbooks), cleaned of heading residue."""
    idx = get_index(book_id)
    chunks = [c for c in idx.section_chunks(f"c{chapter_no}-intro") if c["chunk_type"] == "text"]
    if not chunks:
        chunks = idx.chapter_chunks(chapter_no, ["text"])[:1]
    text = re.sub(r"\s+", " ", " ".join(c["text"] for c in chunks[:2])).strip()
    text = re.sub(r"\bCHAPTER PREVIEW\b", " ", text)
    text = re.sub(r"^\s*\d{1,3}\s+", "", text).strip()
    if title and text.lower().startswith(title.lower()):
        text = text[len(title):]
    return re.sub(r"\s+", " ", text).strip(" .:-")


@st.cache_data(show_spinner=False)
def page_png(book_id: str, page: int) -> bytes | None:
    book = load_book(book_id)
    return render_page_png(book, page) if book else None


class StudioAgent(TeachingAgent):
    """TeachingAgent with two UI-level integrations: a fixed intent for dedicated workspaces, and images the
    student uploads (problem photos, diagrams) sent to Gemini alongside the textbook context."""

    def __init__(self, book, index, llm, force_intent: str | None = None, user_images: list | None = None):
        self.force_intent = force_intent
        self.user_images = user_images or []
        super().__init__(book, index, llm)

    def node_classify(self, state):
        out = super().node_classify(state)
        if self.force_intent:
            out["intent"], out["followup"] = self.force_intent, ""
        return out

    def _images(self, pages):
        parts, ok = super()._images(pages)
        if self.user_images:
            from google.genai import types
            extra = []
            for name, data, mime in self.user_images:
                extra += [f"[Attached image uploaded by the student: {name}]",
                          types.Part.from_bytes(data=data, mime_type=mime)]
            parts = extra + parts
        return parts, ok


class QuizTypeAdapter:
    """Wraps the Gemini client so a quiz uses only the question types the student picked in Quiz Arena."""

    def __init__(self, llm: GeminiClient, qtypes: list[str]):
        self._llm, self._qtypes = llm, qtypes

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def generate(self, system, contents, schema=None, **kw):
        if schema is QuizSet and self._qtypes:
            contents = [contents[0] + "\n\nQUESTION TYPES (this overrides the mix requested above): use ONLY "
                        + ", ".join(self._qtypes) + " questions.", *contents[1:]]
        return self._llm.generate(system, contents, schema=schema, **kw)


def api_key() -> str:
    return settings.gemini_api_key or ss.p_api_key.strip()


def key_problem() -> str | None:
    return api_key_problem(settings.with_api_key(api_key()))


def llm_or_none() -> GeminiClient | None:
    if key_problem():
        return None
    try:
        return get_llm(api_key(), settings.gemini_model)
    except LLMError:
        return None


# =============================================================================================================
# 4. DOMAIN HELPERS (learning state, activity log, metrics)
# =============================================================================================================
def active_book() -> ProcessedBook | None:
    books = list_books()
    if not books:
        return None
    ids = [b.book_id for b in books]
    if ss.p_book not in ids:
        ss.p_book = ids[0]
    return next(b for b in books if b.book_id == ss.p_book)


def learning_for(book: ProcessedBook) -> dict:
    if book.book_id not in ss.learning:
        ss.learning[book.book_id] = load_progress(book)
    return ss.learning[book.book_id]


def set_learning(book: ProcessedBook, learning: dict) -> None:
    ss.learning[book.book_id] = learning
    save_progress(book, learning)


def chat_for(book_id: str) -> list[dict]:
    return ss.chats.setdefault(book_id, [])


def current_chapter(book: ProcessedBook) -> int:
    numbers = [c.number for c in book.chapters]
    if ss.p_chapter not in numbers:
        cur = learning_for(book).get("current_chapter")
        ss.p_chapter = cur if cur in numbers else numbers[0]
    return ss.p_chapter


def log_activity(learning: dict, kind: str, label: str, chapter: int | None) -> None:
    act = list(learning.get("activity", []))
    act.append({"t": datetime.now().isoformat(timespec="seconds"), "kind": kind, "label": label[:100], "chapter": chapter})
    learning["activity"] = act[-400:]


def _times(learning: dict) -> list[datetime]:
    out = []
    for a in learning.get("activity", []):
        try:
            out.append(datetime.fromisoformat(a["t"]))
        except (KeyError, ValueError):
            pass
    return sorted(out)


def study_minutes(learning: dict, since: datetime | None = None) -> float | None:
    """Active study time estimated from the activity log: gaps of up to 30 min between actions count as
    study, longer gaps start a new session (each session gets 3 min for reading its last answer)."""
    ts = [t for t in _times(learning) if since is None or t >= since]
    if not ts:
        return None
    total, sessions = 0.0, 1
    for a, b in zip(ts, ts[1:]):
        gap = (b - a).total_seconds() / 60
        if gap <= 30:
            total += gap
        else:
            sessions += 1
    return total + 3 * sessions


def fmt_minutes(m: float) -> str:
    return f"{m:.0f} min" if m < 60 else f"{m / 60:.1f} h"


def rel_time(iso: str) -> str:
    try:
        d = datetime.now() - datetime.fromisoformat(iso)
    except ValueError:
        return ""
    s = d.total_seconds()
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return f"{d.days} d ago" if d.days < 14 else datetime.fromisoformat(iso).strftime("%d %b")


def chapter_stats(book: ProcessedBook, learning: dict) -> list[dict]:
    done = set(learning.get("completed_sections", []))
    rows = []
    for ch in book.chapters:
        seq = chapter_sequence(book, ch)
        n_done = sum(1 for sid, _ in seq if sid in done)
        quizzes = [q for q in learning.get("quiz_history", []) if q.get("chapter_no") == ch.number and q["total"]]
        best = max((100 * q["score"] / q["total"] for q in quizzes), default=None)
        status = "completed" if seq and n_done == len(seq) else (
            "in progress" if n_done or ch.number in learning.get("chapters_started", []) else "not started")
        rows.append({"ch": ch, "lessons": len(seq), "done": n_done, "pct": round(100 * n_done / max(1, len(seq))),
                     "best_quiz": best, "status": status})
    return rows


def quiz_accuracy(learning: dict) -> tuple[float | None, float | None]:
    hist = [q for q in learning.get("quiz_history", []) if q["total"]]
    if not hist:
        return None, None
    pcts = [100 * q["score"] / q["total"] for q in hist]
    trend = pcts[-1] - pcts[-2] if len(pcts) >= 2 else None
    return sum(pcts) / len(pcts), trend


# =============================================================================================================
# 5. UI COMPONENTS
# =============================================================================================================
def esc(s) -> str:
    return html.escape(str(s))


def md_html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def page_head(pid: str, title: str, subtitle: str, kicker: str = "") -> None:
    ico = PAGE_META[pid][1]
    md_html(f'<div class="ta-page-head"><div class="ta-tile-ico">{icon(ico, 22)}</div><div>'
            f'{f"<div class=ta-kicker>{esc(kicker)}</div>" if kicker else ""}'
            f'<div class="ta-h2">{esc(title)}</div><p class="ta-sub">{esc(subtitle)}</p></div></div>')


def section(title: str, right: str = "") -> None:
    md_html(f'<div class="ta-section"><div class="ta-h3">{esc(title)}</div><span class="ta-small">{right}</span></div>')


def empty_state(ico: str, title: str, text: str) -> None:
    md_html(f'<div class="ta-empty"><div class="ta-tile-ico">{icon(ico, 24)}</div><h4>{esc(title)}</h4><p>{text}</p></div>')


def ring(pct: float, size: int = 92, stroke: int = 9, label: str | None = None) -> str:
    r = (size - stroke) / 2
    c = 2 * math.pi * r
    off = c * (1 - max(0.0, min(100.0, pct)) / 100)
    return (f'<div class="ta-ring" style="width:{size}px;height:{size}px;--c:{c:.1f};--off:{off:.1f}">'
            f'<svg width="{size}" height="{size}"><defs><linearGradient id="ta-ring-grad" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0%" stop-color="var(--accent)"/><stop offset="100%" stop-color="var(--accent2)"/></linearGradient></defs>'
            f'<circle class="bg" cx="{size / 2}" cy="{size / 2}" r="{r:.1f}" fill="none" stroke-width="{stroke}"/>'
            f'<circle class="fg" cx="{size / 2}" cy="{size / 2}" r="{r:.1f}" fill="none" stroke-width="{stroke}" stroke-linecap="round"/>'
            f'</svg><b style="font-size:{size * 0.22:.0f}px">{esc(label if label is not None else pct_label(pct))}</b></div>')


def pct_label(pct: float) -> str:
    return "<1%" if 0 < pct < 1 else f"{pct:.0f}%"


def bar(pct: float) -> str:
    return f'<div class="ta-bar"><span style="--w:{max(0.0, min(100.0, pct)):.1f}%"></span></div>'


def stat_card(key: str, ico: str, value: str, label: str, hint: str = "", trend: str = "", trend_cls: str = "") -> None:
    with st.container(key=f"card_{key}"):
        t = f'<span class="ta-trend {trend_cls}">{esc(trend)}</span>' if trend else ""
        md_html(f'<div class="ta-stat"><div class="ta-stat-top"><div class="ta-tile-ico">{icon(ico, 19)}</div>{t}</div>'
                f'<div class="ta-stat-val">{esc(value)}</div><div class="ta-stat-label">{esc(label)}</div>'
                f'<div class="ta-stat-hint">{esc(hint)}</div></div>')


def typing(text: str, done: list[str]) -> str:
    steps = "".join(f"<li>{esc(s)}</li>" for s in done)
    return (f'<div class="ta-typing"><i><span></span><span></span><span></span></i>{esc(text)}</div>'
            f'{f"<ul class=ta-steps>{steps}</ul>" if steps else ""}')


def no_key_banner() -> None:
    if key_problem():
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.warning(f"**Gemini API key missing.** {key_problem()} The professor can't answer until it is set.")
        c2.button("Open Settings", key="nokey_settings", on_click=go, args=("settings",), width="stretch")


def need_book(pid: str) -> ProcessedBook | None:
    book = active_book()
    if book is None:
        empty_state("book-open", "No textbook yet",
                    "Upload your textbook PDF once — Textbook AI extracts its chapters, equations and figures and "
                    "builds a private knowledge base on your laptop.")
        st.write("")
        c = st.columns([1, 1, 1])[1]
        with c:
            ibutton("Upload a textbook", f"empty_up_{pid}", "upload", type="primary", on_click=go, args=("textbook",))
        return None
    if not book.chapters:
        empty_state("triangle-alert", "No chapters detected", "This PDF has no detectable chapter structure.")
        return None
    return book


def chapter_picker(book: ProcessedBook, wkey: str, label: str = "Chapter", allow_all: bool = False,
                   pref: str = "p_chapter") -> int:
    numbers = [c.number for c in book.chapters]
    options = ([0] if allow_all else []) + numbers
    if ss.get(pref) not in options:
        ss[pref] = current_chapter(book)
    bound(st.selectbox, label, pref, wkey, options=options,
          format_func=lambda n: "Whole textbook" if n == 0 else book.chapter(n).label)
    return ss[pref]


def split_footer(content: str) -> tuple[str, str]:
    """The agent appends sources/recommendations after a final '---'; show them in an expander."""
    if "\n\n---\n" in content:
        body, footer = content.rsplit("\n\n---\n", 1)
        if footer.lstrip().startswith(("**📚", "🖼️", "**🧭")):
            return body, footer
    return content, ""


_CALLOUT = re.compile(r"^(📝|💡|🤔|⚠️|🔁|🗺️)")


def enhance(md: str) -> str:
    """Turn one-line exam points, notes and check-yourself prompts into styled callouts."""
    lines, out = md.split("\n"), []
    for i, line in enumerate(lines):
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
        if _CALLOUT.match(line.strip()) and prev_blank and next_blank:
            out.append("> " + line.strip())
        else:
            out.append(line)
    return "\n".join(out)


_HEAD_A = re.compile(r"^\s{0,3}#{1,6}\s+(?:\*\*)?(?:\d{1,2}[.)]\s*)?(.+?)(?:\*\*)?\s*:?\s*$")
_HEAD_B = re.compile(r"^\s{0,3}(?:\d{1,2}[.)]\s*)?\*\*(?:\d{1,2}[.)]\s*)?([^*\n]{2,70}?)\s*:?\s*\*\*\s*:?\s*(.*)$")
_HEAD_C = re.compile(r"^\s{0,3}\d{1,2}[.)]\s+([A-Z][^:\n]{2,45}?)\s*:\s*(.*)$")


def split_sections(md: str, keywords: dict[str, tuple[str, ...]]) -> tuple[str, list[tuple[str, str, str]]] | None:
    """Split a structured answer into (preamble, [(slot, title, body)]) using the numbered headings the teaching
    modes ask Gemini to produce. Returns None when the answer is not structured that way."""
    def slot_for(title: str) -> str | None:
        t = title.lower()
        return next((slot for slot, kws in keywords.items() if any(k in t for k in kws)), None)

    pre, sections, cur, buf = [], [], None, []
    for line in md.split("\n"):
        title, rest = None, ""
        for rx in (_HEAD_A, _HEAD_B, _HEAD_C):
            m = rx.match(line)
            if m:
                title = m.group(1).strip()
                rest = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
                break
        slot = slot_for(title) if title and len(title) <= 70 else None
        if slot:
            if cur:
                sections.append((cur[0], cur[1], "\n".join(buf).strip()))
            cur, buf = (slot, re.sub(r"^\d{1,2}[.)]\s*", "", title)), ([rest] if rest else [])
        elif cur:
            buf.append(line)
        else:
            pre.append(line)
    if cur:
        sections.append((cur[0], cur[1], "\n".join(buf).strip()))
    return ("\n".join(pre).strip(), sections) if len(sections) >= 3 else None


def plot_theme(fig: pgo.Figure, height: int) -> pgo.Figure:
    t = THEMES[ss.p_theme]
    fig.update_layout(height=height, margin=dict(l=6, r=10, t=10, b=6), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=t["muted"], size=12),
                      showlegend=False, hoverlabel=dict(bgcolor=t["card"], bordercolor=t["accent"],
                                                        font=dict(color=t["text"], family="Inter, sans-serif")))
    fig.update_xaxes(gridcolor=t["track"], zeroline=False, linecolor=t["border"], automargin=True)
    fig.update_yaxes(gridcolor=t["track"], zeroline=False, linecolor=t["border"], automargin=True)
    return fig


# =============================================================================================================
# 6. RUNNING THE AGENT
# =============================================================================================================
def request(target: str, label: str, query: str = "", action: dict | None = None, *, intent: str | None = None,
            images: list | None = None, mode: str | None = None, chapter: int | None = None,
            qtypes: list[str] | None = None, difficulty: str | None = None, switch: bool = True) -> None:
    """Button callback: queue a request for the workspace `target` (processed when that page renders)."""
    ss.pending = {"target": target, "label": label, "query": query, "action": action, "intent": intent,
                  "images": images or [], "mode": mode, "chapter": chapter, "qtypes": qtypes or [],
                  "difficulty": difficulty}
    if chapter:
        ss.p_chapter = chapter
    if switch:
        ss.p_page = target
        ss.nav_clicked = True


def take_pending(target: str) -> dict | None:
    if ss.pending and ss.pending["target"] == target:
        req, ss.pending = ss.pending, None
        return req
    return None


def execute(book: ProcessedBook, req: dict, placeholder, history: list[dict] | None = None) -> dict:
    """Run the LangGraph agent for one request and update learning state. Raises LLMError/RetrievalError."""
    llm = llm_or_none()
    if llm is None:
        raise LLMError(key_problem() or "Gemini is not configured.")
    client = QuizTypeAdapter(llm, req["qtypes"]) if req.get("qtypes") else llm
    agent = StudioAgent(book, get_index(book.book_id), client, force_intent=req.get("intent"),
                        user_images=req.get("images"))
    chapter_no = req.get("chapter") if req.get("chapter") is not None else current_chapter(book)
    state = {"query": req.get("query", ""), "mode": req.get("mode") or ss.p_mode, "language": ss.p_language,
             "difficulty": req.get("difficulty") or ss.p_difficulty, "chapter_no": chapter_no or None,
             "history": history or [], "learning": learning_for(book), "action": req.get("action"),
             "verify": ss.p_verify}
    done: list[str] = []

    def on_step(node: str) -> None:
        if node in DONE_TEXT:
            done.append(DONE_TEXT[node])
        placeholder.markdown(typing(STEP_TEXT.get(node, "Working…"), done), unsafe_allow_html=True)

    placeholder.markdown(typing(STEP_TEXT[None], done), unsafe_allow_html=True)
    try:
        out = agent.run(state, on_step=on_step)
    except (LLMError, RetrievalError) as exc:
        ss.llm_state = ("error", str(exc))
        raise
    ss.llm_state = ("ok", "")
    learning = out["learning"]
    log_activity(learning, out.get("intent", ""), req.get("label") or req.get("query", ""),
                 learning.get("current_chapter"))
    set_learning(book, learning)
    if out.get("quiz"):
        ss.quiz_feedback = None
    return out


def run_inline(book: ProcessedBook, target: str, req: dict) -> None:
    """Execute a workspace request (non-chat pages) with an animated loading card; store the result."""
    holder = st.empty()
    with holder.container(key=f"tile_loading_{target}"):
        ph = st.empty()
        try:
            out = execute(book, req, ph)
            ss.results[target] = {"req": {k: v for k, v in req.items() if k != "images"}, "out": out,
                                  "time": datetime.now().isoformat(timespec="seconds"),
                                  "image": req["images"][0] if req.get("images") else None}
        except (LLMError, RetrievalError) as exc:
            ss.results[target] = {"error": str(exc), "req": {k: v for k, v in req.items() if k != "images"}}
    holder.empty()
    if target == "quiz" or (ss.results.get(target, {}).get("out") or {}).get("quiz"):
        ss.p_page = "quiz"
    st.rerun()


# =============================================================================================================
# 7. SIDEBAR
# =============================================================================================================
def sidebar(book: ProcessedBook | None) -> None:
    with st.sidebar:
        md_html(f'<div class="ta-brand"><div class="ta-logo">{icon("sparkles", 21)}</div>'
                f'<div><b>TEXTBOOK AI</b><small>AI LEARNING WORKSPACE</small></div></div>')
        with st.container(key="navgroup"):
            for pid, label, _ in PAGES:
                st.button(label, key=f"nav_{pid}", on_click=go, args=(pid,), width="stretch")

        state, msg = ss.llm_state
        if key_problem():
            dot, gem, sub = "bad", "Not configured", "Add your API key"
        elif state == "ok":
            dot, gem, sub = "ok", "Connected", settings.gemini_model
        elif state == "error":
            dot, gem, sub = "warn", "Last request failed", msg[:60]
        else:
            dot, gem, sub = "idle", "Key configured", settings.gemini_model
        if book:
            idx_ok = get_index(book.book_id).count() >= book.chunk_count
            tdot, tname, tsub = ("ok" if idx_ok else "warn"), book.name, (
                f"{book.page_count} pages · {len(book.chapters)} chapters" if idx_ok else "Indexing incomplete")
        else:
            tdot, tname, tsub = "bad", "No textbook", "Upload a PDF"
        md_html(f'<div class="ta-status">'
                f'<div class="ta-status-row"><span class="ta-dot {dot}"></span><div><b>Gemini · {esc(gem)}</b>{esc(sub)}</div></div>'
                f'<div class="ta-status-row"><span class="ta-dot {tdot}"></span><div><b>{esc(tname)}</b>{esc(tsub)}</div></div>'
                f'</div>')
        bound(st.selectbox, "Learning mode", "p_mode", "w_mode_sidebar", options=LEARNING_MODES)
        ss["w_theme_toggle"] = ss.p_theme == "dark"
        st.toggle("Dark mode", key="w_theme_toggle",
                  on_change=lambda: ss.update(p_theme="dark" if ss.w_theme_toggle else "light"))
        md_html(f'<div class="ta-version">TEXTBOOK AI · v{APP_VERSION}</div>')


# =============================================================================================================
# 8. PAGES
# =============================================================================================================
def page_overview() -> None:
    book = active_book()
    with st.container(key="hero"):
        md_html('<div class="ta-kicker">Your AI learning workspace</div>'
                '<div class="ta-h1">Welcome back, Student!</div>'
                '<p class="ta-sub">Ready to understand Engineering Materials better?</p>')
        st.write("")
        c1, _ = st.columns([1, 3])
        with c1:
            if book:
                ch = current_chapter(book)
                ibutton("CONTINUE LEARNING", "hero_continue", "arrow-right", type="primary", on_click=request,
                        args=("professor", f"Continue Chapter {ch}"), kwargs={"action": {"type": "teach", "op": "next"},
                                                                         "mode": "Complete Chapter Teaching", "chapter": ch})
            else:
                ibutton("UPLOAD TEXTBOOK", "hero_upload", "upload", type="primary", on_click=go, args=("textbook",))
    if book is None:
        st.write("")
        empty_state("book-open", "Start by adding your textbook",
                    "Your dashboard fills with real progress as you learn — chapters, lessons, quizzes and study time.")
        return

    learning = learning_for(book)
    stats = chapter_stats(book, learning)
    week = datetime.now() - timedelta(days=7)
    acts_week = [a for a in learning.get("activity", []) if a.get("t", "") >= week.isoformat()]
    lessons_week = sum(1 for a in acts_week if a.get("kind") == "chapter_teaching")
    minutes = study_minutes(learning)
    minutes_week = study_minutes(learning, since=week)
    acc, trend = quiz_accuracy(learning)
    total_lessons = sum(r["lessons"] for r in stats)
    st.write("")
    cols = st.columns(4)
    with cols[0]:
        stat_card("s_ch", "library", str(len(book.chapters)), "Chapters available", f"{total_lessons} lessons detected")
    with cols[1]:
        stat_card("s_done", "circle-check", str(len(learning.get("completed_sections", []))), "Topics completed",
                  "lessons finished in Chapter teaching", f"+{lessons_week} this week" if lessons_week else "")
    with cols[2]:
        if minutes is None:
            stat_card("s_time", "timer", "—", "Study time", "Starts with your first question")
        else:
            stat_card("s_time", "timer", fmt_minutes(minutes), "Study time", "Estimated from your activity",
                      f"{fmt_minutes(minutes_week)} this week" if minutes_week else "", "flat")
    with cols[3]:
        if acc is None:
            stat_card("s_acc", "target", "—", "Quiz accuracy", "No quiz attempts yet")
        else:
            t = "" if trend is None else (f"▲ {trend:.0f} pts" if trend > 0 else f"▼ {abs(trend):.0f} pts" if trend < 0 else "± 0")
            stat_card("s_acc", "target", f"{acc:.0f}%", "Quiz accuracy",
                      f"{len(learning.get('quiz_history', []))} quiz(zes) taken", t,
                      "" if (trend or 0) > 0 else ("down" if (trend or 0) < 0 else "flat"))

    left, right = st.columns([3, 2], gap="medium")
    with left:
        section("Continue learning")
        ch_no = current_chapter(book)
        ch = book.chapter(ch_no)
        row = next(r for r in stats if r["ch"].number == ch_no)
        seq = chapter_sequence(book, ch)
        ptr = learning.get("lesson_pointer", {}).get(str(ch_no), {"section_idx": 0, "part": 0})
        nxt = seq[ptr["section_idx"]][1] if ptr["section_idx"] < len(seq) else "Chapter complete — take the quiz"
        with st.container(key="card_continue"):
            c1, c2 = st.columns([1, 3], vertical_alignment="center")
            c1.markdown(ring(row["pct"]), unsafe_allow_html=True)
            c2.markdown(f'<div class="ta-small">{esc(book.name)}</div><div class="ta-h3">{esc(ch.label)}</div>'
                        f'<p class="ta-sub" style="margin-top:6px">Last topic: <b>{esc(learning.get("current_topic") or "—")}</b>'
                        f'<br>Up next: {esc(nxt)}</p>', unsafe_allow_html=True)
            ibutton("Resume lesson", "resume", "play", type="primary", on_click=request,
                    args=("professor", f"Continue Chapter {ch_no}"),
                    kwargs={"action": {"type": "teach", "op": "next"}, "mode": "Complete Chapter Teaching", "chapter": ch_no})
    with right:
        section("Recent activity")
        acts = list(reversed(learning.get("activity", [])))[:6]
        with st.container(key="tile_activity"):
            if acts:
                kind_icon = {"chapter_teaching": "book-open", "quiz": "trophy", "quiz_answer": "trophy",
                             "numerical": "calculator", "diagram": "scan-eye", "derivation": "sigma",
                             "exam_prep": "clipboard-check", "revision": "zap"}
                md_html("".join(
                    f'<div class="ta-act"><div class="ta-tile-ico">{icon(kind_icon.get(a.get("kind"), "message-square"), 15)}</div>'
                    f'<div><b>{esc(a.get("label") or INTENT_LABELS.get(a.get("kind"), "Study"))}</b>'
                    f'<span class="ta-small">{esc(INTENT_LABELS.get(a.get("kind"), "Quiz answer" if a.get("kind") == "quiz_answer" else "Activity"))}'
                    f' · {rel_time(a["t"])}</span></div></div>' for a in acts))
            elif learning.get("recent_topics"):
                md_html("".join(f'<span class="ta-chip">{esc(t)}</span>' for t in learning["recent_topics"][:8]))
            else:
                md_html('<p class="ta-sub">No learning activity yet — your sessions will appear here.</p>')

    section("Quick actions")
    actions = [
        ("ask", "message-square", "Ask AI Professor", "Grounded answers with page references", "professor", False),
        ("explore", "library", "Explore Chapters", "Browse every chapter and section", "explorer", True),
        ("solve", "calculator", "Solve Numerical", "Step-by-step worked solutions", "numerical", False),
        ("mcq", "trophy", "Practice MCQs", "Adaptive quizzes per chapter", "quiz", True),
        ("notes", "zap", "Revision Notes", "One-page formula & concept sheets", "exam", False),
    ]
    cols = st.columns(5)
    for col, (key, ico, title, desc, target, alt) in zip(cols, actions):
        with col, st.container(key=f"card_qa_{key}"):
            md_html(f'<div class="ta-tile-ico{" alt" if alt else ""}">{icon(ico, 19)}</div>'
                    f'<div class="ta-h3" style="margin-top:12px">{esc(title)}</div><p class="ta-small" style="min-height:38px">{esc(desc)}</p>')
            ibutton("Open", f"qa_{key}", "arrow-right", on_click=go, args=(target,))


# ---------------------------------------------------------------------------------------------------------- chat
SUGGESTIONS = [
    ("lightbulb", "Explain a concept from my textbook", "Why do we use electrons instead of visible light in a microscope?", None),
    ("book-open", "Teach me an entire chapter", "", "teach"),
    ("sigma", "Derive an important equation", "Derive the relativistic electron wavelength equation step by step", None),
    ("scan-eye", "Explain a textbook diagram", None, "diagram"),
    ("calculator", "Solve a numerical problem", "Calculate the relativistic wavelength of electrons accelerated through 200 kV", None),
    ("clipboard-check", "Prepare me for examinations", "", "exam"),
]


def chat_welcome(book: ProcessedBook) -> None:
    md_html('<div style="text-align:center;padding:2.2rem 0 1.4rem">'
            f'<div class="ta-logo" style="margin:0 auto 14px;width:52px;height:52px;border-radius:16px">{icon("sparkles", 26)}</div>'
            '<div class="ta-h1" style="font-size:32px">Let\'s understand something new.</div>'
            '<p class="ta-sub">Ask anything about Engineering Materials.</p></div>')
    ch = current_chapter(book)
    for row in (SUGGESTIONS[:3], SUGGESTIONS[3:]):
        cols = st.columns(3)
        for col, (ico, title, prompt, kind) in zip(cols, row):
            with col, st.container(key=f"card_sg_{ico}"):
                md_html(f'<div class="ta-tile-ico">{icon(ico, 18)}</div><div class="ta-h3" style="margin:10px 0 2px;font-size:15.5px">{esc(title)}</div>'
                        f'<p class="ta-small" style="min-height:36px">{esc(prompt or ("Chapter " + str(ch) + " · " + book.chapter(ch).title))}</p>')
                if kind == "teach":
                    st.button("Start", key=f"sg_{ico}", width="stretch", on_click=request,
                              args=("professor", f"Teach me Chapter {ch}"),
                              kwargs={"action": {"type": "teach", "op": "next"}, "mode": "Complete Chapter Teaching", "chapter": ch})
                elif kind == "exam":
                    st.button("Start", key=f"sg_{ico}", width="stretch", on_click=request,
                              args=("professor", f"Exam preparation — Chapter {ch}"), kwargs={"action": {"type": "exam"}, "chapter": ch})
                elif kind == "diagram":
                    st.button("Open Diagram Studio", key=f"sg_{ico}", width="stretch", on_click=go, args=("diagram",))
                else:
                    st.button("Ask", key=f"sg_{ico}", width="stretch", on_click=request, args=("professor", prompt, prompt))


def render_message(book: ProcessedBook, msg: dict, i: int, is_last: bool) -> None:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
            if msg.get("attachment"):
                md_html(f'<span class="ta-chip">{icon("image", 13)} {esc(msg["attachment"])}</span>')
        return
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        body, footer = split_footer(msg["content"])
        st.markdown(enhance(body))
        if msg.get("images"):
            with st.expander(f"PDF page(s) read by the professor · {', '.join(map(str, msg['images']))}", icon=":material/image:"):
                for p in msg["images"]:
                    png = page_png(book.book_id, p)
                    if png:
                        st.image(png, caption=f"PDF page {p}", width="stretch")
        if footer:
            with st.expander("Textbook references & what to study next", icon=":material/menu_book:"):
                st.markdown(footer)
        ver = msg.get("verification") or {}
        if ver.get("checked_by_llm") or msg.get("warnings"):
            with st.expander("Verification report", icon=":material/fact_check:"):
                if ver.get("checked_by_llm"):
                    st.markdown("✅ **Verified against the retrieved passages**" if ver.get("verdict") == "pass"
                                else "✏️ **Corrected by the verifier**")
                    for k, t in (("unsupported_claims", "Unsupported claims"), ("math_or_unit_issues", "Equation / unit issues"),
                                 ("missing_points", "Missing points")):
                        if ver.get(k):
                            st.markdown(f"*{t}:*\n" + "\n".join(f"- {x}" for x in ver[k]))
                for w in msg.get("warnings", []):
                    st.caption("⚠️ " + w)
        if msg.get("quiz"):
            ibutton("Open Quiz Arena", f"goquiz_{i}", "trophy", type="primary", on_click=go, args=("quiz",))
        if is_last and not msg.get("error"):
            message_actions(book, body, i)


def message_actions(book: ProcessedBook, body: str, i: int) -> None:
    chat = chat_for(book.book_id)
    last_user = next((m for m in reversed(chat) if m["role"] == "user"), None)
    topic = (last_user or {}).get("query") or learning_for(book).get("current_topic", "")
    with st.container(key="msg_actions"):
        cols = st.columns([1, 1.25, 1.35, 1.15, 1.5])
        with cols[0].popover("Copy", width="stretch"):
            st.caption("Use the copy icon at the top-right of the box.")
            st.code(body, language="markdown", wrap_lines=True)
        with cols[1]:
            ibutton("Regenerate", f"act_regen_{i}", "refresh-cw", type="tertiary", on_click=regenerate, args=(book.book_id,))
        with cols[2]:
            ibutton("More simply", f"act_simple_{i}", "wand-sparkles", type="tertiary", on_click=request,
                    args=("professor", "Explain that more simply", "Explain that more simply"))
        with cols[3]:
            ibutton("Continue", f"act_cont_{i}", "play", type="tertiary", on_click=request,
                    args=("professor", "Continue", "continue"))
        with cols[4]:
            ibutton("Quiz me on this", f"act_quiz_{i}", "trophy", type="tertiary", on_click=request,
                    args=("quiz", f"Quiz on: {topic[:80]}", f"Quiz on {topic[:160]}"),
                    kwargs={"action": {"type": "quiz", "n": 5}})


def regenerate(book_id: str) -> None:
    chat = chat_for(book_id)
    idx = next((j for j in range(len(chat) - 1, -1, -1) if chat[j]["role"] == "user"), None)
    if idx is None:
        return
    req = chat[idx].get("req")
    del chat[idx:]
    if req:
        ss.pending = dict(req)


def page_professor() -> None:
    book = need_book("professor")
    if book is None:
        return
    chat = chat_for(book.book_id)
    h1, h2 = st.columns([3, 2], vertical_alignment="bottom")
    with h1:
        page_head("professor", "AI Professor", "Grounded in your textbook · cites PDF pages · Hinglish, English or Hindi")
    with h2:
        chapter_picker(book, "w_chapter_chat", "Focus chapter")
    no_key_banner()
    if ss.p_mode == "Complete Chapter Teaching":
        learning = learning_for(book)
        ch = current_chapter(book)
        seq = chapter_sequence(book, book.chapter(ch))
        ptr = learning.get("lesson_pointer", {}).get(str(ch), {"section_idx": 0, "part": 0})
        nxt = seq[ptr["section_idx"]][1] if ptr["section_idx"] < len(seq) else "chapter complete"
        c1, c2 = st.columns([4, 1.2], vertical_alignment="center")
        c1.markdown(f'<div class="ta-callout">{icon("book-open", 15)} &nbsp;Chapter {ch} · next lesson: <b>{esc(nxt)}</b></div>',
                    unsafe_allow_html=True)
        with c2:
            ibutton("Next lesson", "next_lesson", "play", type="primary", disabled=bool(key_problem()), on_click=request,
                    args=("professor", f"Next lesson: {nxt}"), kwargs={"action": {"type": "teach", "op": "next"}})

    if not chat and not (ss.pending and ss.pending["target"] == "professor"):
        chat_welcome(book)
    for i, msg in enumerate(chat):
        render_message(book, msg, i, is_last=(i == len(chat) - 1 and msg["role"] == "assistant"))

    # ---- composer settings (above the pinned input) ----
    with st.container(key="composer_bar"):
        c1, c2, c3 = st.columns([2.2, 2.2, 1], vertical_alignment="bottom")
        with c1:
            bound(st.segmented_control, "Teaching language", "p_language", "w_lang_chat", options=LANGUAGES, required=True)
        with c2:
            bound(st.selectbox, "Teaching mode", "p_mode", "w_mode_chat", options=LEARNING_MODES)
        with c3:
            ibutton("Clear", "clear_chat", "trash-2", disabled=not chat, on_click=lambda: ss.chats.update({book.book_id: []}))

    prompt = st.chat_input("Ask your professor anything… (attach a PDF of notes or a photo of a problem)",
                           key="composer", accept_file=True, file_type=["pdf", "png", "jpg", "jpeg", "webp"],
                           disabled=bool(key_problem()))
    req = take_pending("professor")
    if prompt is not None and (prompt.text or prompt.files):
        images, notes_msg, attach = [], "", ""
        for f in prompt.files:
            if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                try:
                    chunks = process_notes_pdf(f.getvalue(), f.name, book.book_id)
                    get_index(book.book_id).index_chunks(chunks)
                    notes_msg = f"📎 Added **{f.name}** to your knowledge base ({len(chunks)} passages)."
                except (PDFProcessingError, RetrievalError) as exc:
                    notes_msg = f"⚠️ Could not add {f.name}: {exc}"
            else:
                images.append((f.name, f.getvalue(), f.type or "image/png"))
                attach = f.name
        if notes_msg:
            chat.append({"role": "assistant", "content": notes_msg, "error": True})
        if prompt.text or images:
            text = prompt.text or "Explain the attached image."
            req = {"target": "professor", "label": text, "query": text, "action": None,
                   "intent": "general" if images and not prompt.text else None, "images": images, "mode": None,
                   "chapter": None, "qtypes": [], "difficulty": None, "attachment": attach}
        elif notes_msg:
            st.rerun()
    if req:
        history = [{"role": m["role"], "content": m["content"]} for m in chat if not m.get("error")]
        chat.append({"role": "user", "content": req["label"] or req["query"], "query": req["query"],
                     "attachment": req.get("attachment", ""), "req": req})
        render_message(book, chat[-1], len(chat) - 1, False)
        with st.chat_message("assistant", avatar=":material/auto_awesome:"):
            ph = st.empty()
            try:
                out = execute(book, req, ph, history)
                chat.append({"role": "assistant", "content": out.get("answer", ""), "images": out.get("image_pages", []),
                             "verification": out.get("verification"), "warnings": out.get("warnings", []),
                             "intent": out.get("intent", ""), "quiz": bool(out.get("quiz"))})
                if out.get("quiz"):
                    ss.p_page = "quiz"
            except (LLMError, RetrievalError) as exc:
                chat.append({"role": "assistant", "content": f"⚠️ {exc}", "error": True})
        st.rerun()


# ---------------------------------------------------------------------------------------------------- textbook
STAGES = [("Analyzing your textbook…", 0.0), ("Extracting chapter structure…", 0.78),
          ("Building your knowledge base…", 1.01), ("Preparing your AI professor…", 2.0)]


def ingest(uploaded) -> None:
    data = uploaded.getvalue()
    book_id = book_id_for(data)
    with st.status("Analyzing your textbook…", expanded=True) as status:
        bar_ph = st.empty()
        try:
            book = load_book(book_id)
            if book is None:
                def cb(f, m):
                    label = next(t for t, lim in reversed(STAGES) if f >= lim) if f < 1 else STAGES[1][0]
                    status.update(label=label)
                    bar_ph.progress(min(f, 1.0), text=m)
                book, _ = process_pdf(data, uploaded.name, cb)
                st.write(f"✓ {book.page_count} pages · {len(book.chapters)} chapters via {book.detection_method} · "
                         f"{book.chunk_count} passages · {len(book.figure_index)} figures")
            else:
                st.write("✓ Already processed earlier — reusing its structure.")
            status.update(label="Building your knowledge base…")
            if not finish_index(book, bar_ph):
                status.update(label="Knowledge base incomplete", state="error")
                return
            status.update(label="Preparing your AI professor…")
            ss.p_book = book.book_id
            ss.p_chapter = None
            status.update(label="Your textbook is ready", state="complete", expanded=False)
        except PDFProcessingError as exc:
            status.update(label="Could not process this PDF", state="error")
            st.error(str(exc))
    ss.done_uploads.add(uploaded.file_id)


def finish_index(book: ProcessedBook, bar_ph) -> bool:
    index = get_index(book.book_id)
    chunks = load_chunks(book.book_id)
    if index.count() >= len(chunks):
        return True
    bar_ph.progress(0.0, text="Loading the local embedding model (first run downloads ~130 MB)…")
    try:
        index.index_chunks(chunks, progress=lambda f, m: bar_ph.progress(min(f, 1.0), text=m))
    except RetrievalError as exc:
        st.error(str(exc))
        return False
    return True


def page_textbook() -> None:
    page_head("textbook", "My Textbook", "Upload once — the text, chapters, equations and figures are indexed on your laptop")
    left, right = st.columns([2, 3], gap="medium")
    with left, st.container(key="panel_left"):
        md_html(f'<div class="ta-h3">{icon("upload", 17)} Upload textbook PDF</div><p class="ta-small">Text-based PDFs work best. '
                'Processing runs once; afterwards the knowledge base is reused instantly.</p>')
        up = st.file_uploader("Textbook PDF", type=["pdf"], key="book_upl", label_visibility="collapsed")
        if up is not None and up.file_id not in ss.done_uploads:
            ingest(up)
            st.rerun()
        books = list_books()
        if len(books) > 1:
            st.write("")
            ids = [b.book_id for b in books]
            if ss.p_book not in ids:
                ss.p_book = ids[0]
            bound(st.selectbox, "Active textbook", "p_book", "w_book", options=ids,
                  format_func=lambda i: next(b.name for b in books if b.book_id == i))
    book = active_book()
    with right:
        if book is None:
            empty_state("book-open", "No textbook uploaded", "Drop your textbook PDF on the left. You'll see each stage: "
                        "analysing, extracting chapter structure, building your knowledge base.")
            return
        with st.container(key="panel_right"):
            idx = get_index(book.book_id)
            indexed = idx.count()
            md_html(f'<div class="ta-kicker">Active textbook</div><div class="ta-h2">{esc(book.name)}</div>'
                    f'<p class="ta-small">{esc(book.file_name)}</p>')
            md_html('<div class="ta-kv">' + "".join(
                f"<div><b>{v}</b><span>{k}</span></div>" for k, v in [
                    ("PDF pages", book.page_count), ("Chapters", len(book.chapters)),
                    ("Lessons", sum(len(chapter_sequence(book, c)) for c in book.chapters)),
                    ("Figures", len(book.figure_index)), ("Tables", len(book.table_index)),
                    ("Indexed passages", indexed)]) + "</div>")
            st.write("")
            badge = "done" if indexed >= book.chunk_count else "prog"
            md_html(f'<span class="ta-badge {badge}">{"Knowledge base ready" if badge == "done" else "Indexing incomplete"}</span> '
                    f'<span class="ta-badge new">Chapters via {esc(book.detection_method)}</span>')
            for note in book.notes:
                st.caption("ℹ️ " + note)
            if indexed < book.chunk_count:
                if ibutton("Finish building the knowledge base", "finish_idx", "layers", type="primary"):
                    with st.status("Building your knowledge base…", expanded=True) as s:
                        if finish_index(book, st.empty()):
                            s.update(label="Ready", state="complete")
                            st.rerun()
    st.write("")
    with st.container(key="tile_notes"):
        md_html(f'<div class="ta-h3">{icon("notebook-pen", 17)} Your notes & handouts</div>'
                '<p class="ta-small">Lecture notes are searched together with the textbook and always labelled as your notes.</p>')
        notes = st.file_uploader("Notes PDF", type=["pdf"], key="notes_upl", label_visibility="collapsed")
        if notes is not None and notes.file_id not in ss.done_notes:
            with st.spinner("Adding your notes to the knowledge base…"):
                try:
                    chunks = process_notes_pdf(notes.getvalue(), notes.name, book.book_id)
                    get_index(book.book_id).index_chunks(chunks)
                    st.success(f"Added {len(chunks)} passages from {notes.name}.")
                except (PDFProcessingError, RetrievalError) as exc:
                    st.error(str(exc))
            ss.done_notes.add(notes.file_id)
        try:
            srcs = get_index(book.book_id).notes_sources()
        except Exception:
            srcs = []
        md_html("".join(f'<span class="ta-chip">{icon("file-text", 13)} {esc(s.replace("Your notes: ", ""))}</span>' for s in srcs)
                or '<p class="ta-small">No notes added yet.</p>')


# ---------------------------------------------------------------------------------------------------- explorer
def page_explorer() -> None:
    book = need_book("explorer")
    if book is None:
        return
    if ss.p_explorer_ch and book.chapter(ss.p_explorer_ch):
        chapter_detail(book, book.chapter(ss.p_explorer_ch))
        return
    page_head("explorer", "Chapter Explorer", f"{len(book.chapters)} chapters detected in {book.name}")
    explorer_grid(book)


@st.fragment
def explorer_grid(book: ProcessedBook) -> None:
    c1, c2, c3, c4 = st.columns([2.4, 2.6, 1.3, 1.1], vertical_alignment="bottom")
    q = c1.text_input("Search chapters", key="ex_q", placeholder="Search chapters or sections…")
    flt = c2.segmented_control("Status", ["All", "Completed", "In progress", "Not started"], key="ex_f", default="All")
    order = c3.selectbox("Sort", ["Chapter ↑", "Chapter ↓"], key="ex_sort")
    view = c4.segmented_control("View", ["Grid", "List"], key="ex_view", default="Grid")
    learning = learning_for(book)
    rows = chapter_stats(book, learning)
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in r["ch"].title.lower() or ql in str(r["ch"].number)
                or any(ql in s.title.lower() for s in r["ch"].sections)]
    if flt and flt != "All":
        rows = [r for r in rows if r["status"] == flt.lower()]
    rows.sort(key=lambda r: r["ch"].number, reverse=order == "Chapter ↓")
    if not rows:
        empty_state("search", "No chapters match", "Try a different search term or status filter.")
        return
    badge = {"completed": ("done", "Completed"), "in progress": ("prog", "In progress"), "not started": ("new", "Not started")}
    per_row = 3 if view != "List" else 1
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row)
        for col, r in zip(cols, rows[start:start + per_row]):
            ch = r["ch"]
            bcls, btxt = badge[r["status"]]
            desc = chapter_intro(book.book_id, ch.number, ch.title)
            with col, st.container(key=f"card_ch_{ch.number}"):
                if view == "List":
                    a, b, c = st.columns([5, 1.2, 1.2], vertical_alignment="center")
                    a.markdown(f'<div class="ta-small">CHAPTER {ch.number} · {esc(ch.part)}</div><div class="ta-h3">{esc(ch.title)}</div>'
                               f'<div style="margin-top:8px">{bar(r["pct"])}</div>'
                               f'<span class="ta-small">{r["done"]}/{r["lessons"]} lessons · {len(ch.teaching_sections())} sections</span> '
                               f'<span class="ta-badge {bcls}">{btxt}</span>', unsafe_allow_html=True)
                    tgt = b
                else:
                    md_html(f'<div style="display:flex;justify-content:space-between;align-items:center">'
                            f'<span class="ta-tile-ico" style="font-family:var(--font-display);font-weight:800">{ch.number}</span>'
                            f'<span class="ta-badge {bcls}">{btxt}</span></div>'
                            f'<div class="ta-h3" style="margin:12px 0 4px;min-height:44px">{esc(ch.title)}</div>'
                            f'<p class="ta-small ta-clamp" style="min-height:58px">{esc(desc) or esc(ch.part)}</p>'
                            f'{bar(r["pct"])}<p class="ta-small" style="margin:6px 0 10px">{len(ch.teaching_sections())} sections · '
                            f'{r["done"]}/{r["lessons"]} lessons · PDF pp. {ch.page_start}–{ch.page_end}</p>')
                    tgt = None
                if view == "List":
                    with b:
                        if st.button("Details", key=f"chd_{ch.number}", width="stretch"):
                            ss.p_explorer_ch = ch.number
                            st.rerun()
                    with c:
                        if st.button("Learn", key=f"chl_{ch.number}", type="primary", width="stretch"):
                            request("professor", f"Teach me Chapter {ch.number}", action={"type": "teach", "op": "next"},
                                    mode="Complete Chapter Teaching", chapter=ch.number)
                            ss.p_mode = "Complete Chapter Teaching"
                            st.rerun()
                else:
                    b1, b2 = st.columns(2)
                    if b1.button("Details", key=f"chd_{ch.number}", width="stretch"):
                        ss.p_explorer_ch = ch.number
                        st.rerun()
                    if b2.button("Start learning", key=f"chl_{ch.number}", type="primary", width="stretch"):
                        request("professor", f"Teach me Chapter {ch.number}", action={"type": "teach", "op": "next"},
                                mode="Complete Chapter Teaching", chapter=ch.number)
                        ss.p_mode = "Complete Chapter Teaching"
                        st.rerun()


def chapter_detail(book: ProcessedBook, ch) -> None:
    learning = learning_for(book)
    row = next(r for r in chapter_stats(book, learning) if r["ch"].number == ch.number)
    ibutton("All chapters", "back_explorer", "arrow-left", type="tertiary", width="content", on_click=go,
            args=("explorer",), kwargs={"p_explorer_ch": None})
    with st.container(key="hero"):
        a, b = st.columns([4, 1], vertical_alignment="center")
        a.markdown(f'<div class="ta-kicker">Chapter {ch.number} · {esc(ch.part)} · PDF pp. {ch.page_start}–{ch.page_end}</div>'
                   f'<div class="ta-h1" style="font-size:32px">{esc(ch.title)}</div>'
                   f'<p class="ta-sub">{row["done"]} of {row["lessons"]} lessons completed'
                   f'{" · best quiz " + str(round(row["best_quiz"])) + "%" if row["best_quiz"] is not None else ""}</p>',
                   unsafe_allow_html=True)
        b.markdown(ring(row["pct"], 104), unsafe_allow_html=True)
        c1, c2, c3, _ = st.columns([1.3, 1.1, 1.1, 1.5])
        with c1:
            ibutton("Start teaching" if row["done"] == 0 else "Continue teaching", "detail_teach", "play", type="primary",
                    on_click=request, args=("professor", f"Teach me Chapter {ch.number}"),
                    kwargs={"action": {"type": "teach", "op": "next"}, "mode": "Complete Chapter Teaching", "chapter": ch.number})
        with c2:
            ibutton("Take quiz", "detail_quiz", "trophy", on_click=go, args=("quiz",), kwargs={"p_chapter": ch.number})
        with c3:
            ibutton("Exam prep", "detail_exam", "clipboard-check", on_click=go, args=("exam",), kwargs={"p_chapter": ch.number})
    intro = chapter_intro(book.book_id, ch.number, ch.title)
    if intro:
        section("Chapter introduction", "from the textbook")
        md_html(f'<div class="ta-callout" style="background:var(--glass);border-color:var(--border)">{esc(intro[:700])}{"…" if len(intro) > 700 else ""}</div>')
    left, right = st.columns([3, 2], gap="medium")
    done = set(learning.get("completed_sections", []))
    with left:
        section("Sections", f"{len(chapter_sequence(book, ch))} lessons")
        with st.container(key="tile_sections"):
            for sid, title in chapter_sequence(book, ch):
                a, b = st.columns([5, 1.1], vertical_alignment="center")
                a.markdown(f'{"✅" if sid in done else "○"} &nbsp;{esc(title)}', unsafe_allow_html=True)
                with b:
                    st.button("Teach", key=f"sec_{sid}", width="stretch", on_click=request,
                              args=("professor", f"Teach: {title}"),
                              kwargs={"action": {"type": "teach", "op": "jump", "section_id": sid},
                                      "mode": "Complete Chapter Teaching", "chapter": ch.number})
    with right:
        topics = [re.sub(r"^[0-9.]+[A-Z]?\s+", "", t) for s in ch.teaching_sections() for t in s.subsections]
        if not topics:
            topics = [re.sub(r"^[0-9.]+\s+", "", s.title) for s in ch.teaching_sections()]
        section("Important concepts", "textbook headings")
        md_html("".join(f'<span class="ta-chip">{esc(t)}</span>' for t in topics[:24]))
        section("Key equations", "numbered in the text")
        eqs = chapter_equations(book.book_id, ch.number)
        if not eqs:
            md_html('<p class="ta-small">No numbered equations detected in this chapter.</p>')
        for lab, snip in eqs[:8]:
            a, b = st.columns([4, 1.2], vertical_alignment="center")
            a.markdown(f'<span class="ta-chip accent">({esc(lab)})</span> <span class="ta-small">{esc(snip)}</span>', unsafe_allow_html=True)
            with b:
                st.button("Derive", key=f"eq_{lab}", width="stretch", on_click=request,
                          args=("professor", f"Derive equation ({lab})", f"Derive and explain equation ({lab}) step by step"),
                          kwargs={"mode": "Mathematical Derivations", "chapter": ch.number})
        if len(eqs) > 8:
            st.caption(f"+ {len(eqs) - 8} more equations — pick any in Mathematical Derivations mode.")


# --------------------------------------------------------------------------------------------------- numerical
NUM_SLOTS = {"given": ("given",), "required": ("required", "to find", "find"), "formula": ("formula",),
             "explain": ("explanation of", "symbols"), "subst": ("substitut",), "steps": ("step", "calculation"),
             "final": ("final answer",), "interp": ("interpret",), "verify": ("verif", "check")}


def page_numerical() -> None:
    book = need_book("numerical")
    if book is None:
        return
    page_head("numerical", "Numerical Solver", "Every step shown — given data, formula, substitution, units and verification")
    no_key_banner()
    left, right = st.columns([2, 3], gap="medium")
    with left, st.container(key="panel_left"):
        problem = st.text_area("Problem", key="num_text", height=150,
                               placeholder="e.g. Calculate the relativistic wavelength of electrons accelerated through 300 kV, "
                                           "and compare it with the non-relativistic value.")
        ch = chapter_picker(book, "w_num_ch", "Textbook chapter (for formulas)", allow_all=True, pref="p_num_chapter")
        img = st.file_uploader("Photo of the problem (optional)", type=["png", "jpg", "jpeg", "webp"], key="num_img")
        images = [(img.name, img.getvalue(), img.type or "image/png")] if img else []
        text = problem.strip() or ("Solve the numerical problem shown in the attached image." if images else "")
        ibutton("Solve step by step", "num_solve", "zap", type="primary", disabled=not text or bool(key_problem()),
                on_click=request, args=("numerical", (text[:90] or "Numerical problem"), text),
                kwargs={"intent": "numerical", "images": images, "chapter": ch or 0, "mode": "Numerical Problem Solver"})
        ch_label = book.chapter(ch).label if ch else book.chapter(current_chapter(book)).label
        ibutton("Give me a practice problem", "num_practice", "flask-conical", disabled=bool(key_problem()),
                on_click=request, args=("numerical", "Practice numerical",
                                        f"Create one exam-style numerical problem from {ch_label} using the textbook's "
                                        "formulas, then solve it completely."),
                kwargs={"intent": "numerical", "chapter": ch or current_chapter(book), "mode": "Numerical Problem Solver"})
    with right:
        req = take_pending("numerical")
        if req:
            run_inline(book, "numerical", req)
        res = ss.results.get("numerical")
        if not res:
            empty_state("calculator", "Your worked solution appears here",
                        "Type a problem (or attach a photo of it). The solution follows the textbook's formulas and constants, "
                        "with every calculation step.")
            return
        if res.get("error"):
            st.error(res["error"])
            return
        render_numerical(book, res)


def render_numerical(book: ProcessedBook, res: dict) -> None:
    out = res["out"]
    body, footer = split_footer(out.get("answer", ""))
    if res.get("image"):
        with st.expander("Your problem image"):
            st.image(res["image"][1], width="stretch")
    parsed = split_sections(body, NUM_SLOTS)
    with st.container(key="panel_right"):
        md_html(f'<div class="ta-kicker">Solution</div><div class="ta-h3">{esc(res["req"].get("label", ""))}</div>')
        if not parsed:
            st.markdown(enhance(body))
        else:
            pre, secs = parsed
            if pre:
                st.markdown(enhance(pre))
            for slot, title, text in secs:
                if slot == "final":
                    md_html(f'<div class="ta-answer-final"><div class="ta-kicker">{icon("circle-check", 14)} {esc(title)}</div></div>')
                    st.markdown(text)
                elif slot in ("subst", "steps"):
                    with st.expander(title, expanded=True, icon=":material/functions:"):
                        st.markdown(enhance(text))
                else:
                    sec_icon = {"given": "list-checks", "required": "target", "formula": "sigma", "explain": "lightbulb",
                                "interp": "brain", "verify": "circle-check"}.get(slot, "chevron-right")
                    md_html(f'<div class="ta-sec">{icon(sec_icon, 17)} {esc(title)}</div>')
                    st.markdown(enhance(text))
    if footer:
        with st.expander("Textbook references", icon=":material/menu_book:"):
            st.markdown(footer)


# ----------------------------------------------------------------------------------------------------- diagram
DIA_SLOTS = {"which": ("which figure", "which page", "identif"), "represents": ("represent",), "labels": ("label",),
             "axes": ("axes", "curves", "arrows", "symbols"), "relations": ("relationship",), "mechanism": ("mechanism",),
             "draw": ("reproduce", "draw", "exam"), "conclude": ("conclusion",)}


def page_diagram() -> None:
    book = need_book("diagram")
    if book is None:
        return
    page_head("diagram", "Diagram Studio", "Figures read straight from the PDF page — labels, mechanism and how to draw it in the exam")
    no_key_banner()
    left, right = st.columns([2, 3], gap="medium")
    preview, req_args = None, None
    with left, st.container(key="panel_left"):
        src = bound(st.segmented_control, "Diagram source", "p_dia_src", "w_dia_src",
                    options=["Textbook figure", "PDF page", "Upload image"], required=True)
        question = st.text_input("What should the professor focus on? (optional)", key="dia_q",
                                 placeholder="e.g. Why is the objective aperture placed here?")
        extra = f" Focus: {question.strip()}" if question.strip() else ""
        if src == "Textbook figure":
            ch = chapter_picker(book, "w_dia_ch")
            chapter = book.chapter(ch)
            figs = sorted([f for f, p in book.figure_index.items() if chapter.page_start <= p <= chapter.page_end],
                          key=lambda s: [int(x) for x in re.findall(r"\d+", s)])
            if figs:
                fig = st.selectbox("Figure", figs, key=f"w_dia_fig_{ch}",
                                   format_func=lambda f: f"Figure {f} · PDF p. {book.figure_index[f]}")
                preview = ("page", book.figure_index[fig])
                req_args = (f"Explain Figure {fig}", f"Explain Figure {fig} in detail.{extra}",
                            {"intent": "diagram", "chapter": ch, "mode": "Diagram Explanation"})
            else:
                st.caption("No figures detected in this chapter.")
        elif src == "PDF page":
            page = st.number_input("PDF page", 1, book.page_count, value=min(61, book.page_count), key="dia_page")
            preview = ("page", int(page))
            ch = book.chapter_for_page(int(page))
            req_args = (f"Explain the diagram on page {int(page)}", f"Explain the diagram on page {int(page)}.{extra}",
                        {"intent": "diagram", "chapter": ch.number if ch else 0, "mode": "Diagram Explanation"})
        else:
            up = st.file_uploader("Diagram image", type=["png", "jpg", "jpeg", "webp"], key="dia_img")
            if up:
                preview = ("image", up.getvalue())
                req_args = ("Explain my diagram",
                            "Explain the attached diagram image like a professor: what it represents, every visible label, "
                            "the axes/arrows/symbols, how the components relate, the underlying mechanism, how to draw it "
                            f"in an examination, and what conclusions follow. Use the textbook passages where relevant.{extra}",
                            {"intent": "general", "images": [(up.name, up.getvalue(), up.type or "image/png")],
                             "mode": "Diagram Explanation"})
        if req_args:
            ibutton("Explain this diagram", "dia_go", "sparkles", type="primary", disabled=bool(key_problem()),
                    on_click=request, args=("diagram", req_args[0], req_args[1]), kwargs=req_args[2])
    with right:
        if preview:
            with st.container(key="tile_preview"):
                if preview[0] == "page":
                    png = page_png(book.book_id, preview[1])
                    if png:
                        st.image(png, caption=f"PDF page {preview[1]}", width="stretch")
                else:
                    st.image(preview[1], caption="Your diagram", width="stretch")
        req = take_pending("diagram")
        if req:
            run_inline(book, "diagram", req)
        res = ss.results.get("diagram")
        if not res:
            if not preview:
                empty_state("scan-eye", "Pick a figure, a page or upload a diagram", "The preview and the explanation appear here.")
            return
        if res.get("error"):
            st.error(res["error"])
            return
        out = res["out"]
        body, footer = split_footer(out.get("answer", ""))
        st.write("")
        with st.container(key="panel_right"):
            md_html(f'<div class="ta-kicker">Interpretation</div><div class="ta-h3">{esc(res["req"].get("label", ""))}</div>')
            parsed = split_sections(body, DIA_SLOTS)
            if not parsed:
                st.markdown(enhance(body))
            else:
                pre, secs = parsed
                if pre:
                    st.markdown(enhance(pre))
                for n, (slot, title, text) in enumerate(secs):
                    with st.expander(f"{n + 1}. {title}", expanded=n < 3):
                        st.markdown(enhance(text))
        if footer:
            with st.expander("Textbook references", icon=":material/menu_book:"):
                st.markdown(footer)


# -------------------------------------------------------------------------------------------------------- exam
def page_exam() -> None:
    book = need_book("exam")
    if book is None:
        return
    page_head("exam", "Exam Preparation", "University-style questions with model answers, and one-page revision sheets")
    no_key_banner()
    with st.container(key="panel_left"):
        c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
        with c1:
            ch = chapter_picker(book, "w_exam_ch")
        topic = c2.text_input("Focus topic (optional)", key="exam_topic", placeholder="e.g. Rutherford scattering, screening")
        b1, b2, _ = st.columns([1.3, 1.3, 2])
        q = topic.strip()
        with b1:
            ibutton("Generate exam pack", "exam_go", "clipboard-check", type="primary", disabled=bool(key_problem()),
                    on_click=request, args=("exam", f"Exam pack — {q or book.chapter(ch).label}", f"Exam questions on {q}" if q else ""),
                    kwargs={"action": {"type": "exam"}, "chapter": ch})
        with b2:
            ibutton("Quick revision sheet", "rev_go", "zap", disabled=bool(key_problem()),
                    on_click=request, args=("exam", f"Revision sheet — {q or book.chapter(ch).label}", f"Revision notes on {q}" if q else ""),
                    kwargs={"action": {"type": "revision"}, "chapter": ch})
    req = take_pending("exam")
    if req:
        kind = (req.get("action") or {}).get("type", "exam")
        holder = st.empty()
        with holder.container(key="tile_loading_exam"):
            ph = st.empty()
            try:
                out = execute(book, req, ph)
                ss.results[f"exam_{kind}"] = {"req": req, "out": out, "time": datetime.now().isoformat(timespec="seconds")}
            except (LLMError, RetrievalError) as exc:
                ss.results[f"exam_{kind}"] = {"error": str(exc), "req": req}
        holder.empty()
        ss.p_exam_view = "Revision sheet" if kind == "revision" else "Exam pack"
        st.rerun()
    have = [k for k in ("exam_exam", "exam_revision") if ss.results.get(k)]
    if not have:
        st.write("")
        empty_state("clipboard-check", "Nothing generated yet",
                    "Choose a chapter and generate an exam pack (2-, 5- and 10-mark questions with model answers) or a "
                    "compact revision sheet of definitions, formulas and memory tricks.")
        return
    st.write("")
    views = [v for v, k in (("Exam pack", "exam_exam"), ("Revision sheet", "exam_revision")) if ss.results.get(k)]
    if ss.p_exam_view not in views:
        ss.p_exam_view = views[0]
    if len(views) > 1:
        bound(st.segmented_control, "Show", "p_exam_view", "w_exam_view", options=views, required=True,
              label_visibility="collapsed")
    res = ss.results["exam_exam" if ss.p_exam_view == "Exam pack" else "exam_revision"]
    if res.get("error"):
        st.error(res["error"])
        return
    body, footer = split_footer(res["out"].get("answer", ""))
    with st.container(key="panel_right"):
        a, b = st.columns([4, 1.3], vertical_alignment="center")
        a.markdown(f'<div class="ta-kicker">{esc(ss.p_exam_view)} · {rel_time(res["time"])}</div>'
                   f'<div class="ta-h3">{esc(res["req"]["label"])}</div>', unsafe_allow_html=True)
        b.download_button("Download .md", body, file_name=re.sub(r"\W+", "_", res["req"]["label"]).strip("_") + ".md",
                          mime="text/markdown", key="exam_dl", width="stretch", icon=":material/download:")
        st.markdown(enhance(body))
    if footer:
        with st.expander("Textbook references", icon=":material/menu_book:"):
            st.markdown(footer)


# -------------------------------------------------------------------------------------------------------- quiz
def page_quiz() -> None:
    book = need_book("quiz")
    if book is None:
        return
    learning = learning_for(book)
    req = take_pending("quiz")
    if req:
        page_head("quiz", "Quiz Arena", "Preparing your quiz")
        run_inline(book, "quiz", req)
    quiz = learning.get("quiz")
    res = ss.results.get("quiz")
    if res and res.get("error") and not (quiz and not quiz.get("finished")):
        st.error(res["error"])
    if quiz and not quiz.get("finished") or ss.quiz_feedback:
        quiz_play(book, learning, quiz)
    elif quiz and quiz.get("finished"):
        quiz_results(book, learning, quiz)
    else:
        quiz_setup(book)


def quiz_setup(book: ProcessedBook, compact: bool = False) -> None:
    if not compact:
        page_head("quiz", "Quiz Arena", "One question at a time · answers revealed only after you respond")
        no_key_banner()
    with st.container(key="panel_left"):
        c1, c2 = st.columns(2)
        with c1:
            ch = chapter_picker(book, "w_quiz_ch")
        with c2:
            if ss.p_quiz_diff not in DIFFICULTIES:
                ss.p_quiz_diff = ss.p_difficulty
            bound(st.segmented_control, "Difficulty", "p_quiz_diff", "w_quiz_diff", options=DIFFICULTIES, required=True)
        if ss.p_quiz_types is None:
            ss.p_quiz_types = list(QTYPES)
        bound(st.pills, "Question types", "p_quiz_types", "w_quiz_types", options=list(QTYPES), selection_mode="multi")
        bound(st.segmented_control, "Number of questions", "p_quiz_n", "w_quiz_n", options=[3, 5, 8, 10], required=True)
        types = [QTYPES[t] for t in ss.p_quiz_types] if ss.p_quiz_types else []
        ibutton("Start quiz", "quiz_start", "play", type="primary", disabled=bool(key_problem()) or not types,
                on_click=request, args=("quiz", f"Quiz — {book.chapter(ch).label}"),
                kwargs={"action": {"type": "quiz", "n": ss.p_quiz_n}, "chapter": ch, "difficulty": ss.p_quiz_diff,
                        "qtypes": types if len(types) < len(QTYPES) else [], "mode": "Quiz Mode"})
        if not types:
            st.caption("Pick at least one question type.")


def quiz_play(book: ProcessedBook, learning: dict, quiz: dict) -> None:
    total = len(quiz["questions"])
    graded = [r for r in quiz["results"] if not r.get("ungraded")]
    score = sum(r["score"] for r in graded)
    a, b = st.columns([4, 1.2], vertical_alignment="center")
    a.markdown(f'<div class="ta-kicker">Quiz Arena · {esc(quiz["scope"])}</div>'
               f'<div class="ta-h2">Question {min(quiz["index"] + (0 if ss.quiz_feedback else 1), total)} of {total}</div>',
               unsafe_allow_html=True)
    b.markdown(f'<div style="text-align:right"><span class="ta-chip accent">{icon("flame", 13)} Score {score:g} / {len(graded)}</span></div>',
               unsafe_allow_html=True)
    md_html(bar(100 * len(quiz["results"]) / total))
    st.write("")
    fb = ss.quiz_feedback
    with st.container(key="tile_quiz"):
        if fb:
            q, r = fb["question"], fb["result"]
            md_html(f'<div class="ta-q">{esc(q["question"])}</div>')
            if r.get("ungraded"):
                cls, head = "neutral", "Could not be graded automatically"
            elif r["correct"]:
                cls, head = "ok", f"Correct! +{r['score']:g}"
            else:
                cls, head = "bad", f"Not quite · {r['score']:g} point{'s' if r['score'] != 1 else ''}"
            md_html(f'<div class="ta-fb {cls}"><b>{esc(head)}</b><p class="ta-small" style="margin:6px 0 0">Your answer: '
                    f'{esc(fb["student"])}<br>Correct answer: <b>{esc(q["correct_answer"])}</b></p></div>')
            st.markdown(r["feedback"])
            if q.get("source_label"):
                st.caption("📚 " + q["source_label"])
            last = quiz.get("finished")
            ibutton("See results" if last else "Next question", "quiz_next", "arrow-right", type="primary",
                    on_click=lambda: ss.update(quiz_feedback=None))
            return
        q = quiz["questions"][quiz["index"]]
        md_html(f'<span class="ta-chip">{esc(q["qtype"].replace("_", " ").title())}</span>'
                f'<span class="ta-chip">{esc(q["topic"])}</span><div class="ta-q">{esc(q["question"])}</div>')
        with st.form(f"quizform_{quiz['started']}_{quiz['index']}", border=False):
            if q["qtype"] in ("mcq", "true_false"):
                labels = [f"{chr(65 + i)}.  {o}" if q["qtype"] == "mcq" else o for i, o in enumerate(q["options"])]
                with st.container(key="quiz_opts"):
                    choice = st.radio("Your answer", labels, index=None, label_visibility="collapsed")
                answer = q["options"][labels.index(choice)] if choice else ""
            else:
                answer = st.text_area("Your answer", height=110 if q["qtype"] != "fill_blank" else 70,
                                      placeholder="Show your working for numerical questions.")
            submitted = st.form_submit_button("Submit answer", type="primary", width="stretch")
    if submitted:
        if not answer.strip():
            st.warning("Answer first — the correct answer is revealed only after you respond.")
            return
        with st.spinner("Checking your answer…"):
            result = grade_answer(llm_or_none(), q, answer, ss.p_language)
        learning = record_quiz_answer(learning, result, answer)
        log_activity(learning, "quiz_answer", q["topic"], quiz.get("chapter_no"))
        set_learning(book, learning)
        ss.quiz_feedback = {"question": q, "result": result, "student": answer}
        st.rerun()


def quiz_results(book: ProcessedBook, learning: dict, quiz: dict) -> None:
    graded = [r for r in quiz["results"] if not r.get("ungraded")]
    total = len(graded)
    score = sum(r["score"] for r in graded)
    pct = 100 * score / total if total else 0
    correct = sum(1 for r in graded if r["correct"])
    page_head("quiz", "Quiz complete", quiz["scope"], kicker="Results")
    with st.container(key="hero"):
        a, b = st.columns([1, 3], vertical_alignment="center")
        a.markdown(ring(pct, 128, 11), unsafe_allow_html=True)
        verdict = ("Excellent — you're exam-ready on this material." if pct >= 85 else
                   "Good work — revise the weak topics once more." if pct >= 60 else
                   "Revise this chapter, then take the quiz again.")
        b.markdown(f'<div class="ta-h2">{score:g} / {total} points</div><p class="ta-sub">{verdict}</p>'
                   f'<div class="ta-kv" style="margin-top:12px"><div><b>{pct:.0f}%</b><span>Accuracy</span></div>'
                   f'<div><b>{correct}</b><span>Correct</span></div><div><b>{total - correct}</b><span>Incorrect</span></div>'
                   f'<div><b>{len(quiz["questions"])}</b><span>Questions</span></div></div>', unsafe_allow_html=True)
    missed = [quiz["questions"][r["index"]] for r in graded if not r["correct"]]
    left, right = st.columns(2, gap="medium")
    with left:
        section("Weak topics")
        md_html("".join(f'<span class="ta-chip">{esc(t)}</span>' for t in dict.fromkeys(q["topic"] for q in missed))
                or '<p class="ta-sub">None — every answer was correct. 🎯</p>')
    with right:
        section("Recommended revision")
        secs = list(dict.fromkeys(q["source_label"] for q in missed if q.get("source_label")))
        if secs:
            md_html("".join(f'<div class="ta-act"><div class="ta-tile-ico">{icon("book-marked", 15)}</div><div><b>{esc(s)}</b></div></div>' for s in secs))
        elif missed:
            md_html('<p class="ta-sub">Revise the weak topics with the AI Professor.</p>')
        else:
            md_html('<p class="ta-sub">Nothing to revise for this quiz.</p>')
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        ibutton("New quiz", "quiz_new", "refresh-cw", type="primary",
                on_click=lambda: (learning.update(quiz=None), set_learning(book, learning)))
    with c2:
        if missed:
            topics = ", ".join(dict.fromkeys(q["topic"] for q in missed))
            ibutton("Revise weak topics", "quiz_revise", "graduation-cap", on_click=request,
                    args=("professor", f"Revise: {topics[:80]}", f"Re-teach these topics simply, with examples: {topics}"),
                    kwargs={"mode": "Concept Explanation", "chapter": quiz.get("chapter_no")})
    with c3:
        ibutton("View progress", "quiz_progress", "chart-line", on_click=go, args=("progress",))


# ---------------------------------------------------------------------------------------------------- progress
def page_progress() -> None:
    book = need_book("progress")
    if book is None:
        return
    learning = learning_for(book)
    page_head("progress", "Learning Progress", "Real data from your lessons, quizzes and study sessions")
    stats = chapter_stats(book, learning)
    total = sum(r["lessons"] for r in stats)
    done = sum(r["done"] for r in stats)
    acc, _ = quiz_accuracy(learning)
    minutes = study_minutes(learning)
    with st.container(key="hero"):
        a, b = st.columns([1, 4], vertical_alignment="center")
        a.markdown(ring(100 * done / max(1, total), 120, 11), unsafe_allow_html=True)
        b.markdown(f'<div class="ta-kicker">Overall progress</div><div class="ta-h2">{done} of {total} lessons completed</div>'
                   f'<div class="ta-kv" style="margin-top:12px">'
                   f'<div><b>{len(learning.get("chapters_started", []))}</b><span>Chapters started</span></div>'
                   f'<div><b>{sum(1 for r in stats if r["status"] == "completed")}</b><span>Chapters completed</span></div>'
                   f'<div><b>{len(learning.get("quiz_history", []))}</b><span>Quizzes taken</span></div>'
                   f'<div><b>{f"{acc:.0f}%" if acc is not None else "—"}</b><span>Quiz accuracy</span></div>'
                   f'<div><b>{fmt_minutes(minutes) if minutes is not None else "—"}</b><span>Study time (est.)</span></div></div>',
                   unsafe_allow_html=True)
    t = THEMES[ss.p_theme]
    left, right = st.columns(2, gap="medium")
    with left:
        section("Chapter completion")
        shown = [r for r in stats if r["status"] != "not started"]
        with st.container(key="tile_chart1"):
            if not shown:
                md_html('<p class="ta-sub">No chapters started yet. Start a lesson from Chapter Explorer.</p>')
            else:
                fig = pgo.Figure(pgo.Bar(x=[r["pct"] for r in shown], y=[f"Ch {r['ch'].number}" for r in shown], orientation="h",
                                       width=0.55, marker=dict(color=t["accent"], cornerradius=5),
                                       customdata=[[r["ch"].title, r["done"], r["lessons"]] for r in shown],
                                       hovertemplate="<b>%{y}: %{customdata[0]}</b><br>%{x}% · %{customdata[1]}/%{customdata[2]} lessons<extra></extra>"))
                fig.update_xaxes(range=[0, 100], ticksuffix="%")
                fig.update_yaxes(autorange="reversed", showgrid=False)
                st.plotly_chart(plot_theme(fig, max(170, 36 * len(shown) + 60)), config={"displayModeBar": False}, theme=None)
    with right:
        section("Quiz performance")
        hist = [q for q in learning.get("quiz_history", []) if q["total"]]
        with st.container(key="tile_chart2"):
            if not hist:
                md_html('<p class="ta-sub">No quiz attempts yet. Your scores will be charted here.</p>')
            else:
                pcts = [round(100 * q["score"] / q["total"]) for q in hist]
                fig = pgo.Figure(pgo.Bar(x=[f"#{i + 1} · Ch {q['chapter_no']}" for i, q in enumerate(hist)], y=pcts,
                                       marker=dict(color=t["accent2"], cornerradius=5),
                                       customdata=[[q["scope"], q["date"], q["score"], q["total"]] for q in hist],
                                       hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}/%{customdata[3]} = %{y}%<extra></extra>"))
                fig.update_yaxes(range=[0, 100], ticksuffix="%")
                fig.update_xaxes(type="category", range=[-0.5, max(len(hist), 8) - 0.5], showgrid=False)
                st.plotly_chart(plot_theme(fig, 230), config={"displayModeBar": False}, theme=None)
    section("Study activity", "last 14 days")
    with st.container(key="tile_chart3"):
        ts = _times(learning)
        if not ts:
            md_html('<p class="ta-sub">No learning activity recorded yet. Every question, lesson and quiz answer is logged here.</p>')
        else:
            days = [(datetime.now() - timedelta(days=d)).date() for d in range(13, -1, -1)]
            counts = [sum(1 for x in ts if x.date() == d) for d in days]
            fig = pgo.Figure(pgo.Bar(x=[d.strftime("%d %b") for d in days], y=counts, marker=dict(color=t["accent"], cornerradius=4),
                                   hovertemplate="%{x}: %{y} learning actions<extra></extra>"))
            fig.update_xaxes(type="category", showgrid=False)
            fig.update_yaxes(rangemode="tozero", dtick=max(1, max(counts) // 4) if max(counts) else 1)
            st.plotly_chart(plot_theme(fig, 210), config={"displayModeBar": False}, theme=None)
    left, right = st.columns(2, gap="medium")
    titles = {sid: t for c in book.chapters for sid, t in chapter_sequence(book, c)}
    with left:
        section("Topics completed", f"{len(learning.get('completed_sections', []))}")
        comp = [titles.get(s, s) for s in learning.get("completed_sections", [])][-12:]
        md_html("".join(f'<span class="ta-chip">{icon("check", 12)} {esc(c)}</span>' for c in reversed(comp))
                or '<p class="ta-sub">No completed lessons yet.</p>')
    with right:
        section("Topics requiring revision")
        weak = sorted(learning.get("weak_topics", {}).items(), key=lambda kv: -kv[1])[:6]
        low = [r for r in stats if r["best_quiz"] is not None and r["best_quiz"] < 60]
        if not weak and not low:
            md_html('<p class="ta-sub">Nothing flagged yet — quizzes reveal weak topics.</p>')
        for i, (topic, n) in enumerate(weak):
            a, b = st.columns([4, 1.2], vertical_alignment="center")
            a.markdown(f'<span class="ta-chip">{esc(topic)}</span><span class="ta-small">missed {n}×</span>', unsafe_allow_html=True)
            with b:
                st.button("Revise", key=f"rev_{i}", width="stretch", on_click=request,
                          args=("professor", f"Revise: {topic}", f"Re-teach {topic} simply with an example"),
                          kwargs={"mode": "Concept Explanation"})
        for r in low:
            md_html(f'<span class="ta-chip">Ch {r["ch"].number}: {esc(r["ch"].title)} · best quiz {r["best_quiz"]:.0f}%</span>')


# ---------------------------------------------------------------------------------------------------- settings
def page_settings() -> None:
    page_head("settings", "Settings", "Appearance, teaching defaults and your Gemini connection")
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(key="tile_appearance"):
            md_html(f'<div class="ta-h3">{icon("sun", 16)} Appearance</div>')
            ss["w_theme_seg"] = "Dark" if ss.p_theme == "dark" else "Light"
            st.segmented_control("Theme", ["Dark", "Light"], key="w_theme_seg", required=True,
                                 on_change=lambda: ss.update(p_theme=ss.w_theme_seg.lower()))
        st.write("")
        with st.container(key="tile_teaching"):
            md_html(f'<div class="ta-h3">{icon("languages", 16)} Teaching defaults</div>')
            bound(st.segmented_control, "Teaching language", "p_language", "w_lang_set", options=LANGUAGES, required=True)
            bound(st.segmented_control, "Explanation difficulty", "p_difficulty", "w_diff_set", options=DIFFICULTIES, required=True)
            bound(st.selectbox, "Learning mode", "p_mode", "w_mode_set", options=LEARNING_MODES)
            bound(st.toggle, "Double-check answers against the textbook (second Gemini call)", "p_verify", "w_verify_set")
    with right:
        with st.container(key="tile_api"):
            md_html(f'<div class="ta-h3">{icon("key-round", 16)} Gemini API</div>')
            if settings.gemini_api_key and not key_problem():
                st.success(f"Key loaded from `.env` · model `{settings.gemini_model}`")
            else:
                bound(st.text_input, "Gemini API key", "p_api_key", "w_api_key", type="password",
                      help="Preferred: GEMINI_API_KEY in the .env file. A key typed here stays in this browser session only.")
                if key_problem():
                    st.caption("⚠️ " + key_problem())
            llm = llm_or_none()
            state, msg = ss.llm_state
            st.caption(f"Model `{settings.gemini_model}` · Gemini calls this session: {llm.calls if llm else 0}"
                       + (f" · last error: {msg}" if state == "error" else ""))
            st.caption(f"Embeddings run locally: `{settings.embedding_model}`")
        st.write("")
        with st.container(key="tile_data"):
            md_html(f'<div class="ta-h3">{icon("history", 16)} Data</div>')
            book = active_book()
            c1, c2 = st.columns(2)
            with c1:
                ibutton("Clear conversation", "set_clear", "trash-2", disabled=book is None,
                        on_click=lambda: ss.chats.update({book.book_id: []}) if book else None)
            with c2, st.popover("Reset progress", width="stretch", disabled=book is None):
                st.write("Erase lessons, quiz history, weak topics and activity for this textbook?")
                if st.button("Yes, reset", type="primary", key="set_reset"):
                    set_learning(book, new_learning_state())
                    ss.quiz_feedback = None
                    st.rerun()
    md_html(f'<p class="ta-version" style="margin-top:22px">TEXTBOOK AI v{APP_VERSION} · Gemini via Google AI Studio · '
            'your textbook and progress stay on this computer</p>')


# =============================================================================================================
# 9. MAIN
# =============================================================================================================
RENDER = {"overview": page_overview, "professor": page_professor, "textbook": page_textbook, "explorer": page_explorer,
          "numerical": page_numerical, "diagram": page_diagram, "exam": page_exam, "quiz": page_quiz,
          "progress": page_progress, "settings": page_settings}


def main() -> None:
    page = ss.p_page if ss.p_page in RENDER else "overview"
    st.markdown(build_css(ss.p_theme, tuple((pid, ico) for pid, _, ico in PAGES), page), unsafe_allow_html=True)
    book = active_book()
    sidebar(book)
    if page == "professor":
        page_professor()  # not wrapped: st.chat_input must stay at the top level to dock at the bottom
    else:
        with st.container(key=f"page_{page}"):
            RENDER[page]()
    if ss.pop("nav_clicked", False):
        # Phones/tablets: close the overlay sidebar after navigating (no native Streamlit API for this).
        with st.container(key="mobile_js"):
            components.html("""<script>
              const w = window.parent, d = w.document;
              if (w.innerWidth < 768) {
                const sb = d.querySelector('section[data-testid="stSidebar"]');
                const btn = d.querySelector('[data-testid="stSidebarCollapseButton"] button');
                if (sb && btn && sb.getAttribute('aria-expanded') !== 'false') btn.click();
              }
            </script>""", height=0)
    if ICON_BUTTONS:
        st.markdown("<style>" + "".join(f".st-key-ib_{k}{{--ico:{icon_uri(v)};}}" for k, v in ICON_BUTTONS.items())
                    + "</style>", unsafe_allow_html=True)


main()
