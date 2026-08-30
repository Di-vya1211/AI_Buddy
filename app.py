"""AI Study Buddy — main Streamlit application.

Pages (sidebar navigation)
--------------------------
Upload       — file upload + doc indexing
ELI10        — Explain Like I'm 10 (simplified learning)
Quiz         — Kahoot-style timed quiz
Planner      — Smart Revision Planner
Ask AI       — RAG doubt solver chat
Flashcards   — flip-through flashcard deck
Feynman mode — student explains → AI scores gaps/misconceptions
Concept map  — interactive node graph via vis-network
Progress     — streak, score chart, weak topics
Key topics   — local TF-IDF extraction (no API call)
"""

import json
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

import db
from utils import MODEL, SUPPORTED_EXTENSIONS, ask, ask_stream, extract_keywords, extract_text_from_file
import os

load_dotenv()
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError, Exception):
    API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="StudyBuddy AI", page_icon="📘", layout="wide")

conn = db.get_connection()

KAHOOT_MAX_POINTS = 1000
KAHOOT_MIN_POINTS = 200
OPTION_STYLES = [
    ("🔺", "#E21B3C"),
    ("🔷", "#1368CE"),
    ("🟡", "#D89E00"),
    ("🟩", "#26890C"),
]

# ---------------------------------------------------------------------------
# Global CSS — dark theme matching reference screenshots
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0f14 !important;
    color: #e8eaf0 !important;
}
[data-testid="stHeader"] { background: #0d0f14 !important; border-bottom: 1px solid #1e2130; }
[data-testid="stSidebar"] { background: #11131c !important; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #c9cdd8 !important; }
section.main > div { padding-top: 1rem; }

/* ── Sidebar nav item ── */
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 10px; cursor: pointer;
    margin-bottom: 4px; transition: background .15s;
}
.nav-item:hover { background: #1e2235; }
.nav-item.active { background: #1e2235; }
.nav-item .nav-label { font-weight: 600; font-size: 0.95rem; color: #e8eaf0 !important; }
.nav-item .nav-sub  { font-size: 0.72rem; color: #7a7f94 !important; }
.nav-arrow { margin-left: auto; color: #7a7f94 !important; font-size: 0.8rem; }

/* ── Active doc badge ── */
.active-doc-box {
    background: #1a1e2e; border: 1px solid #2a2f45; border-radius: 8px;
    padding: 8px 12px; font-size: 0.78rem; color: #9ba3b8 !important;
    display: flex; align-items: center; gap: 6px; word-break: break-all;
}

/* ── Page header card ── */
.page-header {
    display: flex; align-items: flex-start; gap: 16px;
    margin-bottom: 24px;
}
.page-icon {
    width: 52px; height: 52px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
}
.page-icon.yellow  { background: #2a2410; }
.page-icon.purple  { background: #1e1430; }
.page-icon.green   { background: #0e2118; }
.page-icon.teal    { background: #0b1e20; }
.page-title { font-size: 1.6rem; font-weight: 700; color: #ffffff !important; line-height: 1.2; }
.page-subtitle { font-size: 0.88rem; color: #7a7f94 !important; margin-top: 4px; }

/* ── Content card ── */
.card {
    background: #131620; border: 1px solid #1e2235;
    border-radius: 14px; padding: 24px; margin-bottom: 20px;
}

/* ── Level / option pill buttons ── */
.pill-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
.pill {
    padding: 8px 18px; border-radius: 20px; border: 1px solid #2a2f45;
    background: #1a1e2e; color: #c9cdd8 !important; font-size: 0.85rem;
    cursor: pointer; transition: all .15s; white-space: nowrap;
}
.pill.active { background: #3b4ee8; border-color: #3b4ee8; color: #fff !important; }

/* ── Topic chip ── */
.chip {
    display: inline-block; padding: 6px 14px; border-radius: 20px;
    border: 1px solid #2a2f45; background: #1a1e2e;
    color: #c9cdd8 !important; font-size: 0.8rem; margin: 4px; cursor: pointer;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input {
    background: #1a1e2e !important; border: 1px solid #2a2f45 !important;
    color: #e8eaf0 !important; border-radius: 10px !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder { color: #4a4f65 !important; }

/* ── All buttons: base shape ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    transition: background .15s, border-color .15s !important;
}

/* ── Primary (active / action) button — bright blue ── */
.stButton > button[kind="primary"] {
    background: #3b4ee8 !important;
    color: #fff !important;
    border: 2px solid #3b4ee8 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2d3ec0 !important;
    border-color: #2d3ec0 !important;
}

/* ── Secondary (inactive selector) button — dark bg with border ── */
.stButton > button[kind="secondary"] {
    background: #1a1e2e !important;
    color: #c9cdd8 !important;
    border: 1.5px solid #2a2f45 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #1e2340 !important;
    border-color: #3b4ee8 !important;
    color: #fff !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .stSlider > div { color: #e8eaf0 !important; }
[data-testid="stSlider"] [role="slider"] { background: #3b4ee8 !important; }

/* ── Suggested question row ── */
.sq-row {
    display: flex; align-items: center; justify-content: space-between;
    background: #131620; border: 1px solid #1e2235; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px; cursor: pointer;
    transition: border-color .15s;
}
.sq-row:hover { border-color: #3b4ee8; }
.sq-row .sq-text { display: flex; align-items: center; gap: 10px; font-size: 0.9rem; color: #c9cdd8 !important; }
.sq-row .sq-arrow { color: #4a4f65 !important; }
.sq-row.highlighted { border-color: #3b4ee8; }

/* ── Divider label ── */
.or-divider {
    text-align: center; color: #3a3f55 !important;
    font-size: 0.7rem; letter-spacing: 2px;
    margin: 14px 0; position: relative;
}
.or-divider::before, .or-divider::after {
    content: ""; position: absolute; top: 50%;
    width: calc(50% - 80px); height: 1px; background: #1e2235;
}
.or-divider::before { left: 0; }
.or-divider::after  { right: 0; }

/* ── Doc count badge in header ── */
.doc-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #131620; border: 1px solid #1e2235;
    border-radius: 20px; padding: 5px 14px; font-size: 0.8rem; color: #c9cdd8 !important;
}
.dot-green { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
.dot-gray  { width: 8px; height: 8px; border-radius: 50%; background: #4a4f65; }

/* ── Chat input bar ── */
.chat-hint { font-size: 0.72rem; color: #4a4f65 !important; margin-top: 4px; }

/* ── Section label ── */
.section-label { font-size: 0.72rem; letter-spacing: 1.5px; color: #3a3f55 !important; text-transform: uppercase; margin: 16px 0 8px; }

/* ── Quiz option grid ── */
.opt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px; }
.opt-btn {
    background: #1a1e2e; border: 1.5px solid #2a2f45; border-radius: 10px;
    padding: 12px; text-align: center; cursor: pointer; font-size: 0.9rem;
    color: #c9cdd8 !important; transition: all .15s;
}
.opt-btn.active { background: #3b4ee8; border-color: #3b4ee8; color: #fff !important; }
.opt-btn:hover  { border-color: #3b4ee8; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, [data-testid="stDeployButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "chat": [],
    "quiz": None,
    "quiz_answers": {},
    "quiz_checked": False,
    "fc_cards": None,
    "fc_index": 0,
    "fc_revealed": False,
    "concept_map": None,
    "notes_text": "",
    "last_uploaded_name": None,
    "kahoot_index": 0,
    "kahoot_score": 0,
    "kahoot_answered": False,
    "kahoot_start_time": None,
    "kahoot_last_result": None,
    "kahoot_finished": False,
    "kahoot_results": [],
    # NEW state keys
    "page": "eli10",
    "eli10_level": "ELI5 (Age 5–8)",
    "quiz_num_questions": 5,
    "quiz_difficulty": "Mixed",
    "quiz_time": 30,
    "quiz_topic": "",
    "eli10_topic_stage": "",
    "planner_syllabus": "",
    "planner_topics": "",
    "planner_weak": "",
    "planner_exam_date": None,
    "planner_hours": 2.0,
    "ask_pending_question": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# JSON parse helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _parse_quiz_json(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {exc}") from exc
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("Unexpected response shape — expected a JSON object with a \"questions\" list.")
    for i, q in enumerate(questions):
        for field in ("question", "options", "correct_index"):
            if field not in q:
                raise ValueError(f"Question {i + 1} is missing the \"{field}\" field.")
        if not isinstance(q["options"], list) or len(q["options"]) < 2:
            raise ValueError(f"Question {i + 1} has fewer than 2 options.")
        if not isinstance(q["correct_index"], int) or not (0 <= q["correct_index"] < len(q["options"])):
            raise ValueError(f"Question {i + 1} has an invalid correct_index ({q['correct_index']!r}).")
    return questions


def _parse_flashcard_json(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {exc}") from exc
    cards = data.get("cards")
    if not isinstance(cards, list) or len(cards) == 0:
        raise ValueError("Unexpected response shape — expected a \"cards\" list.")
    for i, c in enumerate(cards):
        for field in ("front", "back"):
            if field not in c:
                raise ValueError(f"Card {i + 1} is missing the \"{field}\" field.")
    return cards


def _parse_concept_map_json(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {exc}") from exc
    if not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
        raise ValueError("Unexpected response shape — expected \"nodes\" and \"edges\" lists.")
    if len(data["nodes"]) == 0:
        raise ValueError("The model returned an empty concept map.")
    return data


def _parse_feynman_json(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {exc}") from exc
    if "understanding_score" not in data:
        raise ValueError("Unexpected response shape — missing \"understanding_score\".")
    try:
        data["understanding_score"] = max(0, min(100, int(data["understanding_score"])))
    except (TypeError, ValueError):
        data["understanding_score"] = 0
    return data


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
notes = st.session_state.get("notes_text", "")

with st.sidebar:
    # App title
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:12px 4px 20px;">
        <div style="background:#3b4ee8;width:36px;height:36px;border-radius:9px;
             display:flex;align-items:center;justify-content:center;font-size:1.1rem;">📘</div>
        <div>
            <span style="font-size:1.1rem;font-weight:700;color:#fff;">StudyBuddy</span>
            <span style="color:#3b4ee8;font-size:0.75rem;margin-left:5px;font-weight:600;">AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav helper
    def nav_btn(icon, label, sub, page_key, color="#7a7f94"):
        active = st.session_state["page"] == page_key
        style = "background:#1e2235;border-radius:10px;" if active else "border-radius:10px;"
        arrow = "›" if active else ""
        st.markdown(f"""
        <div class="nav-item {'active' if active else ''}"
             style="{style}padding:10px 14px;display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="color:{color};font-size:1rem;">{icon}</span>
            <div style="flex:1">
                <div class="nav-label">{label}</div>
                <div class="nav-sub">{sub}</div>
            </div>
            <span class="nav-arrow">{arrow}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(label, key=f"nav_{page_key}", help=sub, use_container_width=True):
            st.session_state["page"] = page_key
            st.rerun()

    # Upload at top
    nav_btn("⬆", "Upload", "Syllabus & notes", "upload", "#7a7f94")
    nav_btn("💡", "ELI10",   "Simplified learning", "eli10",   "#eab308")
    nav_btn("⚡", "Quiz",    "Kahoot-style game",   "quiz",    "#a855f7")
    nav_btn("📅", "Planner", "Revision schedule",   "planner", "#22c55e")
    nav_btn("💬", "Ask AI",  "RAG doubt solver",    "ask_ai",  "#22d3ee")

    # Spacer + extra pages collapsible
    st.markdown("<div style='margin:8px 0;border-top:1px solid #1e2235;'></div>", unsafe_allow_html=True)
    nav_btn("🃏", "Flashcards",   "Flip & learn",         "flashcards",   "#f97316")
    nav_btn("🧠", "Feynman",      "Teach to learn",       "feynman",      "#ec4899")
    nav_btn("🕸", "Concept map",  "Visual connections",   "concept_map",  "#8b5cf6")
    nav_btn("📊", "Progress",     "Stats & streaks",      "progress",     "#3b82f6")
    nav_btn("🔑", "Key topics",   "TF-IDF extraction",    "key_topics",   "#14b8a6")

    st.markdown("<div style='margin:16px 0 8px;border-top:1px solid #1e2235;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label' style='padding:0 4px;'>ACTIVE DOC</div>", unsafe_allow_html=True)
    doc_name = st.session_state.get("last_uploaded_name", None)
    if doc_name:
        short = doc_name[:28] + "…" if len(doc_name) > 30 else doc_name
        st.markdown(f"""
        <div class="active-doc-box">
            <span>📄</span> {short}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="active-doc-box" style="color:#3a3f55 !important;">
            No document uploaded
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Top-right doc-count badge
# ---------------------------------------------------------------------------
doc_count = 1 if st.session_state.get("last_uploaded_name") else 0
dot_class = "dot-green" if doc_count else "dot-gray"
st.markdown(f"""
<div style="position:fixed;top:12px;right:24px;z-index:9999;">
    <div class="doc-badge">
        <div class="{dot_class}"></div>
        {doc_count} doc indexed
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Upload
# ---------------------------------------------------------------------------
current_page = st.session_state["page"]

if current_page == "upload":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon" style="background:#1a1a2e;font-size:1.8rem;">⬆️</div>
        <div>
            <div class="page-title">Upload Documents</div>
            <div class="page-subtitle">Upload your syllabus, notes, or study material — PDF, DOCX, PPTX, XLSX, CSV, images and more.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        # ── Show already-indexed file banner (persists across tab switches) ──
        if st.session_state.get("last_uploaded_name"):
            doc_name_disp = st.session_state["last_uploaded_name"]
            notes_len = len(st.session_state.get("notes_text", ""))
            col_info, col_clear = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f"""<div style="background:#0e2118;border:1px solid #22c55e;border-radius:10px;
                    padding:12px 16px;display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                    <span style="font-size:1.2rem;">✅</span>
                    <div>
                        <div style="font-weight:600;color:#22c55e;">Indexed: {doc_name_disp}</div>
                        <div style="font-size:0.78rem;color:#7a7f94;">{notes_len:,} characters extracted · used by all features</div>
                    </div></div>""",
                    unsafe_allow_html=True,
                )
            with col_clear:
                if st.button("🗑 Remove", key="clear_doc", use_container_width=True):
                    st.session_state["notes_text"] = ""
                    st.session_state["last_uploaded_name"] = None
                    st.rerun()

        # ── File uploader — all supported formats ──
        uploaded = st.file_uploader(
            "Upload a new document",
            type=SUPPORTED_EXTENSIONS,
            help="PDF, DOCX, DOC, TXT, PPTX, PPT, XLSX, XLS, CSV, PNG, JPG, JPEG, JFIF, WEBP, GIF, BMP",
        )
        if uploaded is not None and uploaded.name != st.session_state["last_uploaded_name"]:
            img_exts = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".gif", ".bmp")
            is_image = uploaded.name.lower().endswith(img_exts)
            with st.spinner(f"Extracting text from **{uploaded.name}**…"):
                try:
                    extracted = extract_text_from_file(uploaded)
                    if not extracted:
                        if is_image:
                            st.warning(
                                "⚠️ **No text could be read from this image.**\n\n"
                                "Images are only useful if they contain printed/typed text "
                                "(e.g. a photo of a textbook page). "
                                "For diagrams or photos without text, upload a PDF or DOCX instead, "
                                "or paste your notes in the text box below.\n\n"
                                "_If you want OCR support, install "
                                "[pytesseract](https://github.com/madmaze/pytesseract) + "
                                "[Tesseract](https://github.com/tesseract-ocr/tesseract)._"
                            )
                        else:
                            st.warning("No text could be extracted from this file. Try copy-pasting instead.")
                    else:
                        st.session_state["notes_text"] = extracted
                        st.session_state["last_uploaded_name"] = uploaded.name
                        st.success(f"✅ Indexed **{uploaded.name}** — {len(extracted):,} characters extracted.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not read that file: {e}")

        st.markdown("<div class='or-divider'>OR PASTE NOTES DIRECTLY</div>", unsafe_allow_html=True)
        # Do NOT bind with key="notes_text" — that would overwrite the
        # extracted file content with an empty string every time the
        # Upload page re-renders with a blank textarea.
        # Instead, read the value and write to session state only when
        # the user actually typed something.
        pasted = st.text_area(
            "Notes / syllabus",
            height=220,
            placeholder="Paste your notes here...",
            value=st.session_state.get("notes_text", ""),
        )
        if pasted != st.session_state.get("notes_text", ""):
            st.session_state["notes_text"] = pasted
            # Clear the uploaded-file name so the banner reflects
            # that content now came from manual paste, not a file.
            if pasted.strip() == "":
                st.session_state["last_uploaded_name"] = None


# ---------------------------------------------------------------------------
# PAGE: ELI10 — Explain Like I'm 10
# ---------------------------------------------------------------------------
elif current_page == "eli10":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon yellow">💡</div>
        <div>
            <div class="page-title">Explain Like I'm 10</div>
            <div class="page-subtitle">Get crystal-clear, age-appropriate explanations for any topic with analogies and key points.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Level selector buttons ──
    # Buttons are rendered BEFORE the text_input so they can set state safely.
    levels = ["ELI5 (Age 5–8)", "Beginner", "Intermediate"]
    level_icons = {"ELI5 (Age 5–8)": "🧒", "Beginner": "📗", "Intermediate": "🎯"}
    cols = st.columns(len(levels))
    for i, lv in enumerate(levels):
        with cols[i]:
            is_active = st.session_state["eli10_level"] == lv
            if st.button(f"{level_icons[lv]} {lv}", key=f"eli10_lv_{i}",
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state["eli10_level"] = lv
                st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Chip buttons — rendered BEFORE the text_input they pre-fill ──
    # We store the chosen chip in a staging key; the text_input reads it as
    # its default value so Streamlit never sees a post-instantiation mutation.
    chips = ["Photosynthesis", "Newton's Laws", "Machine Learning",
             "Mitosis vs Meiosis", "The French Revolution", "Recursion in programming", "Gravity"]
    if "eli10_topic_stage" not in st.session_state:
        st.session_state["eli10_topic_stage"] = ""

    chip_cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        with chip_cols[i]:
            if st.button(chip, key=f"chip_{i}"):
                st.session_state["eli10_topic_stage"] = chip
                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Topic input + Explain button ──
    # No key= on the text_input — using key= caches the widget value and
    # ignores value= after the first render, breaking chip pre-fill.
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        topic = st.text_input(
            "Topic input", label_visibility="collapsed",
            placeholder="Enter a topic, concept, or question...",
            value=st.session_state["eli10_topic_stage"],
        )
        # Always sync whatever is in the box back to state
        st.session_state["eli10_topic_stage"] = topic
    with col_btn:
        explain_clicked = st.button("➤ Explain", key="eli10_btn", type="primary", use_container_width=True)

    level_instructions = {
        "ELI5 (Age 5–8)": (
            "Explain it like explaining to a 5-8 year old child. Use very simple words, "
            "fun analogies, and short sentences a young child would understand."
        ),
        "Beginner": (
            "Explain it clearly at a beginner level. Use accurate terminology but keep it "
            "easy to follow, with a relatable example."
        ),
        "Intermediate": (
            "Explain it at an intermediate level. Use correct technical terminology and "
            "provide meaningful depth, examples, and nuance."
        ),
    }

    if explain_clicked:
        notes_text = st.session_state.get("notes_text", "").strip()
        topic_val = st.session_state.get("eli10_topic_stage", "").strip()
        source = notes_text or topic_val
        if not source:
            st.warning("Enter a topic above, or upload/paste your notes first.")
        else:
            try:
                instruction = level_instructions[st.session_state["eli10_level"]]
                if notes_text:
                    prompt = (
                        f"{instruction} Break it into small numbered points.\n\n"
                        f"Notes:\n{notes_text}"
                    )
                    if topic_val:
                        prompt += f"\n\nFocus especially on: {topic_val}"
                else:
                    prompt = (
                        f"{instruction} Break it into small numbered points.\n\n"
                        f"Topic: {topic_val}"
                    )
                st.write_stream(ask_stream(prompt))
                db.log_activity(conn, "explain")
            except Exception as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# PAGE: Quiz
# ---------------------------------------------------------------------------
elif current_page == "quiz":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon purple">⚡</div>
        <div>
            <div class="page-title">Quiz</div>
            <div class="page-subtitle">Test your knowledge with MCQs — choose Kahoot (timed &amp; gamified) or Classic (all at once).</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ensure new state keys exist
    for _k, _v in {"quiz_source": "document", "quiz_style": "kahoot",
                   "quiz_paragraph": ""}.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # ── Quiz Style ──────────────────────────────────────────────────────
        st.markdown("**Quiz Style**")
        style_opts = [("⚡ Kahoot (timed)", "kahoot"), ("📋 Classic (no timer)", "classic")]
        st_cols = st.columns(len(style_opts))
        for i, (label, val) in enumerate(style_opts):
            with st_cols[i]:
                active = st.session_state["quiz_style"] == val
                if st.button(label, key=f"qstyle_{val}",
                             type="primary" if active else "secondary",
                             use_container_width=True):
                    st.session_state["quiz_style"] = val
                    st.rerun()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── Quiz Source ──────────────────────────────────────────────────────
        st.markdown("**Quiz From**")
        has_doc = bool(st.session_state.get("notes_text", "").strip())
        src_opts = [
            ("📄 Uploaded Document", "document"),
            ("🔤 Enter a Topic", "topic"),
            ("📝 Paste a Paragraph", "paragraph"),
        ]
        src_cols = st.columns(len(src_opts))
        for i, (label, val) in enumerate(src_opts):
            with src_cols[i]:
                active = st.session_state["quiz_source"] == val
                btn_label = label if not (val == "document" and not has_doc) else "📄 Upload doc first"
                if st.button(btn_label, key=f"qsrc_{val}",
                             type="primary" if active else "secondary",
                             use_container_width=True):
                    st.session_state["quiz_source"] = val
                    st.rerun()

        # Conditional input based on source
        quiz_source = st.session_state["quiz_source"]
        if quiz_source == "topic":
            quiz_topic_input = st.text_input(
                "Topic", placeholder="e.g. Newton's Laws, Machine Learning, French Revolution...",
                value=st.session_state.get("quiz_topic", ""),
            )
            st.session_state["quiz_topic"] = quiz_topic_input
        elif quiz_source == "paragraph":
            quiz_para_input = st.text_area(
                "Paste your paragraph / notes here",
                height=110,
                placeholder="Paste any text and the quiz will be generated from it...",
                value=st.session_state.get("quiz_paragraph", ""),
            )
            st.session_state["quiz_paragraph"] = quiz_para_input
        else:
            if has_doc:
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#22c55e;margin:4px 0 8px;'>"
                    f"✅ Using: <strong>{st.session_state.get('last_uploaded_name','your document')}</strong>"
                    f" ({len(st.session_state.get('notes_text','').strip()):,} chars)</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No document uploaded yet. Go to **Upload** page first, or choose 'Topic' or 'Paragraph'.")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── Questions count ──────────────────────────────────────────────────
        st.markdown("**Number of Questions**")
        num_opts = [3, 5, 7, 10]
        q_cols = st.columns(len(num_opts))
        for i, n in enumerate(num_opts):
            with q_cols[i]:
                active = st.session_state["quiz_num_questions"] == n
                if st.button(str(n), key=f"qnum_{n}",
                             type="primary" if active else "secondary",
                             use_container_width=True):
                    st.session_state["quiz_num_questions"] = n
                    st.rerun()

        # ── Difficulty ───────────────────────────────────────────────────────
        st.markdown("**Difficulty**")
        diff_opts = [("🟢 Easy", "Easy"), ("🟡 Medium", "Medium"),
                     ("🔴 Hard", "Hard"), ("🎲 Mixed", "Mixed")]
        d_cols = st.columns(len(diff_opts))
        for i, (label, val) in enumerate(diff_opts):
            with d_cols[i]:
                active = st.session_state["quiz_difficulty"] == val
                if st.button(label, key=f"qdiff_{val}",
                             type="primary" if active else "secondary",
                             use_container_width=True):
                    st.session_state["quiz_difficulty"] = val
                    st.rerun()

        # ── Time (only shown for Kahoot style) ───────────────────────────────
        if st.session_state["quiz_style"] == "kahoot":
            st.markdown("**Time per question**")
            t_cols = st.columns(2)
            with t_cols[0]:
                active15 = st.session_state["quiz_time"] == 15
                if st.button("🕐 15s", key="qt_15",
                             type="primary" if active15 else "secondary",
                             use_container_width=True):
                    st.session_state["quiz_time"] = 15
                    st.rerun()
            with t_cols[1]:
                active30 = st.session_state["quiz_time"] == 30
                if st.button("🕐 30s", key="qt_30",
                             type="primary" if active30 else "secondary",
                             use_container_width=True):
                    st.session_state["quiz_time"] = 30
                    st.rerun()

        # ── Start button ─────────────────────────────────────────────────────
        if st.button("▶  Start Quiz", key="quiz_start_btn", type="primary", use_container_width=True):
            notes_text = st.session_state.get("notes_text", "").strip()
            quiz_source = st.session_state["quiz_source"]
            quiz_topic_val = st.session_state.get("quiz_topic", "").strip()
            quiz_para_val = st.session_state.get("quiz_paragraph", "").strip()

            # Determine actual source content
            if quiz_source == "document":
                if not notes_text:
                    st.warning("No document indexed. Go to Upload page first.")
                    st.stop()
                source_line = f"Notes (from uploaded document):\n{notes_text}"
            elif quiz_source == "topic":
                if not quiz_topic_val:
                    st.warning("Enter a topic first.")
                    st.stop()
                source_line = f"Topic: {quiz_topic_val}"
            else:  # paragraph
                if not quiz_para_val:
                    st.warning("Paste a paragraph first.")
                    st.stop()
                source_line = f"Text:\n{quiz_para_val}"

            with st.spinner("Building your quiz..."):
                try:
                    weak_topics = db.get_recent_weak_topics(conn)
                    last_pct = db.get_last_score_percentage(conn)
                    num_q = st.session_state["quiz_num_questions"]
                    difficulty = st.session_state["quiz_difficulty"]
                    time_limit = st.session_state["quiz_time"]

                    adaptive_note = ""
                    if weak_topics:
                        adaptive_note += f" Emphasise these previously weak topics: {', '.join(weak_topics)}."
                    if last_pct is not None:
                        if last_pct >= 80:
                            adaptive_note += " Make questions more challenging."
                        elif last_pct < 50:
                            adaptive_note += " Keep questions more fundamental."

                    diff_instruction = (
                        "" if difficulty == "Mixed"
                        else f" All questions should be {difficulty.lower()} difficulty."
                    )

                    raw = ask(
                        f"Generate exactly {num_q} multiple choice questions strictly based ONLY on the "
                        f"content provided below. Do NOT use outside knowledge.{diff_instruction}"
                        " Each wrong option should be a realistic misconception."
                        + adaptive_note +
                        ' Respond ONLY with JSON: {"questions": [{"question": "...",'
                        ' "options": ["...","...","...","..."], "correct_index": 0,'
                        ' "option_feedback": ["...","...","...","..."], "topic": "..."}]}.'
                        " option_feedback: correct → affirmation; wrong → misconception explanation.\n\n"
                        + source_line,
                        json_mode=True,
                    )
                    questions = _parse_quiz_json(raw)
                    st.session_state["quiz"] = questions
                    st.session_state["quiz_answers"] = {}
                    st.session_state["quiz_checked"] = False
                    st.session_state["kahoot_index"] = 0
                    st.session_state["kahoot_score"] = 0
                    st.session_state["kahoot_answered"] = False
                    st.session_state["kahoot_start_time"] = time.time()
                    st.session_state["kahoot_last_result"] = None
                    st.session_state["kahoot_finished"] = False
                    st.session_state["kahoot_results"] = []
                    st.session_state["kahoot_time_limit"] = time_limit
                    st.session_state["active_quiz_style"] = st.session_state["quiz_style"]
                except Exception as e:
                    st.error(f"Could not generate the quiz: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Active quiz ──────────────────────────────────────────────────────────
    quiz = st.session_state["quiz"]
    KAHOOT_TIME_LIMIT = st.session_state.get("kahoot_time_limit", 30)
    active_style = st.session_state.get("active_quiz_style", "kahoot")

    if quiz:
        st.divider()

        # ================================================================
        # CLASSIC MODE — all questions at once, no timer
        # ================================================================
        if active_style == "classic":
            for i, q in enumerate(quiz):
                st.markdown(f"**{i + 1}. {q['question']}**")
                saved = st.session_state["quiz_answers"].get(i)
                choice = st.radio(
                    "Select an answer",
                    options=list(range(len(q["options"]))),
                    format_func=lambda idx, opts=q["options"]: opts[idx],
                    key=f"classic_q_{i}",
                    index=saved,          # None → no option pre-selected
                    label_visibility="collapsed",
                )
                # Only record once the user has actually picked something
                if choice is not None:
                    st.session_state["quiz_answers"][i] = choice
                if st.session_state["quiz_checked"]:
                    if choice == q["correct_index"]:
                        st.success("✅ Correct!")
                    elif choice is None:
                        st.warning("⚠️ Not answered")
                    else:
                        st.error(f"❌ Correct answer: **{q['options'][q['correct_index']]}**")
                    feedback = q.get("option_feedback")
                    if feedback and choice is not None and choice < len(feedback):
                        st.caption(feedback[choice])
                st.divider()

            if not st.session_state["quiz_checked"]:
                unanswered = [i for i in range(len(quiz))
                              if st.session_state["quiz_answers"].get(i) is None]
                if st.button("✔ Check answers", key="classic_check", type="primary", use_container_width=True):
                    if unanswered:
                        st.warning(f"Please answer all questions first. "
                                   f"({len(unanswered)} unanswered)")
                    else:
                        st.session_state["quiz_checked"] = True
                        score = sum(
                            1 for i, q in enumerate(quiz)
                            if st.session_state["quiz_answers"].get(i) == q["correct_index"]
                        )
                        weak = [
                            q.get("topic", "general")
                            for i, q in enumerate(quiz)
                            if st.session_state["quiz_answers"].get(i) != q["correct_index"]
                        ]
                        db.log_quiz_attempt(conn, score, len(quiz), weak)
                        st.rerun()
            else:
                score = sum(
                    1 for i, q in enumerate(quiz)
                    if st.session_state["quiz_answers"].get(i) == q["correct_index"]
                )
                st.metric("Score", f"{score} / {len(quiz)}")
                if st.button("🔄 New quiz", key="classic_retry", type="primary"):
                    st.session_state["quiz"] = None
                    st.session_state["quiz_checked"] = False
                    st.rerun()

        # ================================================================
        # KAHOOT MODE — one question at a time, timed, gamified
        # ================================================================
        else:
            if st.session_state["kahoot_finished"]:
                st.balloons()
                st.markdown("## 🏁 Quiz complete!")
                st.metric("Total points", st.session_state["kahoot_score"])
                correct_count = sum(1 for r in st.session_state.get("kahoot_results", []) if r["correct"])
                st.write(f"{correct_count} / {len(quiz)} correct")
                if st.button("🔄 Play again", key="kahoot_replay"):
                    st.session_state.update({
                        "kahoot_index": 0, "kahoot_score": 0,
                        "kahoot_answered": False,
                        "kahoot_start_time": time.time(),
                        "kahoot_last_result": None,
                        "kahoot_finished": False,
                        "kahoot_results": [],
                    })
                    st.rerun()
            else:
                idx = st.session_state["kahoot_index"]
                q = quiz[idx]
                st.progress(idx / len(quiz))
                st.caption(f"Question {idx + 1} of {len(quiz)}  •  Score: {st.session_state['kahoot_score']}")
                st.markdown(f"### {q['question']}")

                if not st.session_state["kahoot_answered"]:
                    if st.session_state["kahoot_start_time"] is None:
                        st.session_state["kahoot_start_time"] = time.time()

                    st_autorefresh(interval=500, key=f"timer_{idx}")
                    elapsed = time.time() - st.session_state["kahoot_start_time"]
                    remaining = max(0.0, KAHOOT_TIME_LIMIT - elapsed)
                    st.progress(remaining / KAHOOT_TIME_LIMIT)
                    st.markdown(f"**⏱️ {int(remaining)}s left**")

                    if remaining <= 0:
                        st.session_state["kahoot_answered"] = True
                        st.session_state["kahoot_last_result"] = {"picked": None, "correct": False, "points": 0}
                        st.session_state["kahoot_results"].append({"correct": False, "topic": q.get("topic", "general")})
                        st.rerun()
                    else:
                        cols = st.columns(2)
                        for opt_i, option in enumerate(q["options"]):
                            emoji, _ = OPTION_STYLES[opt_i % len(OPTION_STYLES)]
                            with cols[opt_i % 2]:
                                if st.button(f"{emoji} {option}", key=f"kahoot_opt_{idx}_{opt_i}", use_container_width=True):
                                    time_taken = time.time() - st.session_state["kahoot_start_time"]
                                    is_correct = opt_i == q["correct_index"]
                                    points = 0
                                    if is_correct:
                                        speed_fraction = max(0.0, 1 - (time_taken / KAHOOT_TIME_LIMIT))
                                        points = int(KAHOOT_MIN_POINTS + speed_fraction * (KAHOOT_MAX_POINTS - KAHOOT_MIN_POINTS))
                                    st.session_state["kahoot_score"] += points
                                    st.session_state["kahoot_answered"] = True
                                    st.session_state["kahoot_last_result"] = {"picked": opt_i, "correct": is_correct, "points": points}
                                    st.session_state["kahoot_results"].append({"correct": is_correct, "topic": q.get("topic", "general")})
                                    st.rerun()
                else:
                    result = st.session_state["kahoot_last_result"]
                    if result["correct"]:
                        st.success(f"✅ Correct! +{result['points']} points")
                    elif result["picked"] is None:
                        st.error("⏰ Time's up!")
                    else:
                        st.error("❌ Not quite.")

                    st.info(f"Correct answer: **{q['options'][q['correct_index']]}**")
                    feedback = q.get("option_feedback")
                    if feedback and result["picked"] is not None and result["picked"] < len(feedback):
                        st.caption(feedback[result["picked"]])
                    elif feedback:
                        st.caption(feedback[q["correct_index"]])

                    is_last = idx == len(quiz) - 1
                    if st.button("Finish" if is_last else "Next question ›", key=f"kahoot_next_{idx}"):
                        if is_last:
                            results = st.session_state.get("kahoot_results", [])
                            correct_count = sum(1 for r in results if r["correct"])
                            weak = [r["topic"] for r in results if not r["correct"]]
                            db.log_quiz_attempt(conn, correct_count, len(quiz), weak)
                            st.session_state["kahoot_finished"] = True
                        else:
                            st.session_state["kahoot_index"] += 1
                            st.session_state["kahoot_answered"] = False
                            st.session_state["kahoot_start_time"] = time.time()
                            st.session_state["kahoot_last_result"] = None
                        st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Planner — Smart Revision Planner
# ---------------------------------------------------------------------------
elif current_page == "planner":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon green">📅</div>
        <div>
            <div class="page-title">Smart Revision Planner</div>
            <div class="page-subtitle">Paste your syllabus or enter topics — get a personalised day-by-day roadmap with concept sessions, practice quizzes, buffer &amp; rest days.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown("📄 **Syllabus / Notes** *(paste raw text — topics extracted automatically)*")
        syllabus = st.text_area(
            "Syllabus text", label_visibility="collapsed",
            height=130,
            placeholder="Unit 1: Newton's Laws of Motion\nUnit 2: Thermodynamics…\n\nPaste your full syllabus here",
            key="planner_syllabus",
        )

        st.markdown("<div class='or-divider'>OR ENTER TOPICS MANUALLY</div>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Topics to Cover**")
            topics_cover = st.text_input(
                "Topics", label_visibility="collapsed",
                placeholder="Calculus, Thermodynamics, Optics...",
                key="planner_topics",
            )
            st.caption("Comma-separated")
        with col_right:
            st.markdown("**Weak Topics**")
            topics_weak = st.text_input(
                "Weak topics", label_visibility="collapsed",
                placeholder="Topics you struggle with...",
                key="planner_weak",
            )
            st.caption("Comma-separated · scheduled first & more often")

        col_left2, col_right2 = st.columns(2)
        with col_left2:
            st.markdown("**Exam Date** *")
            exam_date = st.date_input("Exam date", label_visibility="collapsed", key="planner_exam_date")
        with col_right2:
            hours_val = st.session_state.get("planner_hours", 2.0)
            st.markdown(f"**Daily Study Hours** <span style='color:#3b4ee8;font-weight:700;'>{hours_val:.0f}h</span>", unsafe_allow_html=True)
            study_hours = st.slider(
                "Daily study hours", min_value=0.5, max_value=8.0, value=hours_val,
                step=0.5, label_visibility="collapsed", key="planner_hours",
                format="%.1fh",
            )

        if st.button("✦  Generate Revision Plan", key="plan_btn", type="primary", use_container_width=True):
            notes_text = st.session_state.get("notes_text", "").strip()
            source_material = syllabus.strip() or notes_text or topics_cover.strip()
            if not source_material:
                st.warning("Paste your syllabus, enter topics, or upload a document first.")
            else:
                try:
                    weak_db = db.get_recent_weak_topics(conn)
                    all_weak = list({*([t.strip() for t in topics_weak.split(",") if t.strip()]), *weak_db})
                    focus_note = (
                        f" Give extra revision time to these topics: {', '.join(all_weak)}."
                        if all_weak else ""
                    )
                    hours_note = f" The student can study {study_hours:.1f} hours per day."
                    date_note = f" The exam is on {exam_date}." if exam_date else ""
                    topics_note = (
                        f" Topics to cover: {topics_cover}." if topics_cover.strip() else ""
                    )
                    prompt = (
                        "Create a focused day-by-day revision plan. Under each day list "
                        "2-4 short actionable tasks. Include concept sessions, mini quizzes, "
                        "buffer days, and rest days. Keep it concise."
                        + focus_note + hours_note + date_note + topics_note
                        + f"\n\nSyllabus / Notes:\n{source_material}"
                    )
                    st.write_stream(ask_stream(prompt))
                    db.log_activity(conn, "plan")
                except Exception as e:
                    st.error(str(e))

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Ask AI — RAG Doubt Solver
# ---------------------------------------------------------------------------
elif current_page == "ask_ai":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon teal">💬</div>
        <div>
            <div class="page-title">Ask AI</div>
            <div class="page-subtitle">RAG-powered doubt solver — answers grounded in your uploaded documents.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "ask_ai_mode" not in st.session_state:
        st.session_state["ask_ai_mode"] = "Standard (detailed)"

    # ── Document status banner ──────────────────────────────────────────────
    _doc_name = st.session_state.get("last_uploaded_name")
    _doc_text = st.session_state.get("notes_text", "").strip()
    if _doc_name and _doc_text:
        st.markdown(
            f"<div style='background:#0e2118;border:1px solid #22c55e;border-radius:8px;"
            f"padding:8px 14px;font-size:0.8rem;color:#22c55e;margin-bottom:10px;'>"
            f"📄 Context: <strong>{_doc_name}</strong> · {len(_doc_text):,} chars · "
            f"all answers grounded in this document</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#2a1a0e;border:1px solid #f97316;border-radius:8px;"
            "padding:8px 14px;font-size:0.8rem;color:#f97316;margin-bottom:10px;'>"
            "⚠️ No document indexed — go to <strong>Upload</strong> to add one. "
            "You can still ask general questions.</div>",
            unsafe_allow_html=True,
        )

    # ── Mode selector ────────────────────────────────────────────────────────
    mode_opts = ["Standard (detailed)", "Socratic (guide me)"]
    m_cols = st.columns(len(mode_opts))
    for i, m in enumerate(mode_opts):
        with m_cols[i]:
            if st.button(m, key=f"ask_mode_{i}",
                         type="primary" if st.session_state["ask_ai_mode"] == m else "secondary",
                         use_container_width=True):
                st.session_state["ask_ai_mode"] = m
                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Empty state / suggested questions (only when no chat yet) ────────────
    if not st.session_state["chat"]:
        st.markdown(
            f"""<div style="text-align:center;padding:24px 0 16px;">
            <div style="background:#0d2e2e;width:64px;height:64px;border-radius:16px;
                 display:inline-flex;align-items:center;justify-content:center;
                 font-size:1.8rem;margin-bottom:12px;">🤖</div>
            <div style="color:#c9cdd8;font-size:1rem;">
                {"Ask me anything about your document!" if _doc_text else "Ask me anything!"}
            </div>
            <div style="color:#3b4ee8;font-size:0.82rem;margin-top:4px;">
                Mode: <strong>{st.session_state["ask_ai_mode"]}</strong>
            </div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-label'>SUGGESTED QUESTIONS</div>", unsafe_allow_html=True)
        suggested = [
            "What is the central idea of this document?",
            "Explain the main concepts in simple terms",
            "What are the most important formulas?",
            "Compare and contrast the key theories",
            "What should I focus on for exams?",
        ]
        for i, sq in enumerate(suggested):
            # Use only Streamlit buttons — no duplicate HTML rows
            is_hl = i == 2
            btn_style = (
                "border:1px solid #3b4ee8;" if is_hl else "border:1px solid #1e2235;"
            )
            st.markdown(
                f"<div style='background:#131620;{btn_style}border-radius:10px;"
                f"padding:14px 18px;margin-bottom:6px;display:flex;"
                f"align-items:center;gap:10px;font-size:0.9rem;color:#c9cdd8;'>"
                f"<span>💡</span> {sq} "
                f"<span style='margin-left:auto;color:#4a4f65;'>›</span></div>",
                unsafe_allow_html=True,
            )
            if st.button(sq, key=f"sq_{i}"):
                st.session_state["ask_pending_question"] = sq
                st.rerun()

    # ── Chat history ─────────────────────────────────────────────────────────
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    # ── Resolve pending suggested question ───────────────────────────────────
    if st.session_state.get("ask_pending_question"):
        doubt = st.session_state["ask_pending_question"]
        st.session_state["ask_pending_question"] = ""
    else:
        doubt = None

    # ── Chat input ───────────────────────────────────────────────────────────
    chat_input = st.chat_input("Ask a question about your document or any topic...")
    if chat_input:
        doubt = chat_input.strip()

    st.caption("Enter to send · Shift+Enter for new line")

    if doubt:
        st.session_state["chat"].append({"role": "user", "text": doubt})
        with st.chat_message("user"):
            st.markdown(doubt)
        with st.chat_message("assistant"):
            try:
                # Always re-read notes_text fresh so we get the latest uploaded doc
                fresh_notes = st.session_state.get("notes_text", "").strip()
                fresh_doc_name = st.session_state.get("last_uploaded_name", "")

                if st.session_state["ask_ai_mode"].startswith("Socratic"):
                    style_instruction = (
                        "Use the Socratic method. Ask a guiding question or give a small hint "
                        "that pushes the student to think. Only give the full answer if the student "
                        "explicitly asks or has genuinely tried."
                    )
                else:
                    style_instruction = "Answer directly and clearly."

                if fresh_notes:
                    context = (
                        f"You are a helpful study assistant. The student has uploaded a document "
                        f"called '{fresh_doc_name}'. Use the following document content as your "
                        f"PRIMARY source of information to answer all questions. Always base your "
                        f"answers on this content when relevant.\n\n"
                        f"=== DOCUMENT CONTENT START ===\n{fresh_notes}\n=== DOCUMENT CONTENT END ===\n\n"
                    )
                else:
                    context = (
                        "You are a helpful study assistant. No document has been uploaded yet. "
                        "Answer based on your general knowledge.\n\n"
                    )

                history = "\n".join(
                    f"{m['role'].upper()}: {m['text']}"
                    for m in st.session_state["chat"]
                )
                prompt = (
                    f"{context}"
                    f"{style_instruction}\n\n"
                    f"Conversation history:\n{history}\n\n"
                    f"Reply to the student's latest message."
                )
                reply = st.write_stream(ask_stream(prompt))
                st.session_state["chat"].append({"role": "assistant", "text": reply})
                db.log_activity(conn, "chat")
            except Exception as e:
                st.error(str(e))

    # ── Clear chat button ────────────────────────────────────────────────────
    if st.session_state["chat"]:
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state["chat"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Flashcards
# ---------------------------------------------------------------------------
elif current_page == "flashcards":
    st.title("🃏 Flashcards")
    notes_text = st.session_state.get("notes_text", "").strip()
    if st.button("Generate flashcards", key="fc_btn"):
        if not notes_text:
            st.warning("Paste your notes first (Upload page).")
        else:
            with st.spinner("Building your flashcards..."):
                try:
                    raw = ask(
                        "Based on the following notes, create 8 flashcards. "
                        'Respond ONLY with JSON: {"cards": [{"front": "...", "back": "..."}]}\n\n'
                        f"Notes:\n{notes_text}",
                        json_mode=True,
                    )
                    cards = _parse_flashcard_json(raw)
                    st.session_state["fc_cards"] = cards
                    st.session_state["fc_index"] = 0
                    st.session_state["fc_revealed"] = False
                    db.log_activity(conn, "flashcards")
                except Exception as e:
                    st.error(f"Could not generate flashcards: {e}")

    cards = st.session_state["fc_cards"]
    if cards:
        idx = st.session_state["fc_index"]
        card = cards[idx]
        st.caption(f"Card {idx + 1} of {len(cards)}")
        content = card["back"] if st.session_state["fc_revealed"] else card["front"]
        st.markdown(
            f"<div style='border:1px solid #2a2f45;border-radius:10px;padding:2rem;"
            f"text-align:center;font-size:1.2rem;min-height:100px;background:#131620;'>{content}</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Previous", disabled=idx == 0):
                st.session_state["fc_index"] -= 1; st.session_state["fc_revealed"] = False; st.rerun()
        with c2:
            if st.button("Flip"):
                st.session_state["fc_revealed"] = not st.session_state["fc_revealed"]; st.rerun()
        with c3:
            if st.button("Next", disabled=idx == len(cards) - 1):
                st.session_state["fc_index"] += 1; st.session_state["fc_revealed"] = False; st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Feynman mode
# ---------------------------------------------------------------------------
elif current_page == "feynman":
    st.title("🧠 Feynman Mode")
    st.caption("Explain a concept in your own words — gaps in your explanation reveal gaps in your understanding.")
    notes_text = st.session_state.get("notes_text", "").strip()
    fy_topic = st.text_input("What are you explaining?", placeholder="e.g. Photosynthesis, Newton's second law...", key="feynman_topic")
    fy_explanation = st.text_area("Your explanation, in your own words", height=150,
                                  placeholder="Explain it like you're teaching a friend...", key="feynman_text")
    if st.button("Get feedback", key="feynman_btn"):
        if not fy_explanation.strip():
            st.warning("Write your explanation first.")
        else:
            with st.spinner("Reviewing your explanation..."):
                try:
                    context = f"Reference notes:\n{notes_text}\n\n" if notes_text else ""
                    topic_line = f"Topic: {fy_topic}\n\n" if fy_topic.strip() else ""
                    raw = ask(
                        f"{context}{topic_line}Evaluate this Feynman-technique explanation. "
                        'Respond ONLY with JSON: {"understanding_score": 0-100, '
                        '"whats_correct": ["..."], "whats_missing": ["..."], '
                        '"misconceptions": ["..."], "simple_analogy": "..."}\n\n'
                        f"Student's explanation:\n{fy_explanation}",
                        json_mode=True,
                    )
                    data = _parse_feynman_json(raw)
                    st.metric("Understanding score", f"{data['understanding_score']}/100")
                    if data.get("whats_correct"):
                        st.markdown("**What you got right**")
                        for item in data["whats_correct"]: st.markdown(f"- {item}")
                    if data.get("whats_missing"):
                        st.markdown("**What's missing**")
                        for item in data["whats_missing"]: st.markdown(f"- {item}")
                    if data.get("misconceptions"):
                        st.markdown("**Misconceptions to fix**")
                        for item in data["misconceptions"]: st.markdown(f"- {item}")
                    if data.get("simple_analogy"):
                        st.info(data["simple_analogy"])
                    db.log_activity(conn, "feynman")
                except Exception as e:
                    st.error(str(e))


# ---------------------------------------------------------------------------
# PAGE: Concept map
# ---------------------------------------------------------------------------
elif current_page == "concept_map":
    st.title("🕸 Concept Map")
    st.caption("Interactive concept map — drag nodes to explore how ideas connect.")
    notes_text = st.session_state.get("notes_text", "").strip()
    if st.button("Generate concept map", key="map_btn"):
        if not notes_text:
            st.warning("Paste your notes first (Upload page).")
        else:
            with st.spinner("Mapping the concepts..."):
                try:
                    raw = ask(
                        "Extract key concepts and relationships from these notes. "
                        'Respond ONLY with JSON: {"nodes": [{"id": 1, "label": "..."}], '
                        '"edges": [{"from": 1, "to": 2, "label": "..."}]}. '
                        "Include 6-12 nodes.\n\n"
                        f"Notes:\n{notes_text}",
                        json_mode=True,
                    )
                    data = _parse_concept_map_json(raw)
                    st.session_state["concept_map"] = data
                    db.log_activity(conn, "concept_map")
                except Exception as e:
                    st.error(f"Could not build the concept map: {e}")

    graph = st.session_state["concept_map"]
    if graph:
        nodes_js = json.dumps(graph["nodes"])
        edges_js = json.dumps(graph["edges"])
        html = f"""
        <div id="network" style="width:100%;height:480px;background:#111;border-radius:8px;"></div>
        <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
        <script>
          const nodes = new vis.DataSet({nodes_js});
          const edges = new vis.DataSet({edges_js});
          const container = document.getElementById('network');
          const options = {{
            nodes: {{ shape:'box', color:{{ background:'#4F46E5', border:'#3730A3',
                      highlight:{{ background:'#6366F1' }} }},
                      font:{{ color:'#ffffff', size:14 }}, margin:10 }},
            edges: {{ arrows:'to', font:{{ color:'#cccccc', size:11, strokeWidth:0 }},
                      color:{{ color:'#888888' }}, smooth:{{ type:'continuous' }} }},
            physics:{{ stabilization:true, barnesHut:{{ springLength:140 }} }},
            interaction:{{ dragNodes:true, zoomView:true }}
          }};
          new vis.Network(container, {{ nodes, edges }}, options);
        </script>
        """
        components.html(html, height=500, scrolling=False)


# ---------------------------------------------------------------------------
# PAGE: Progress
# ---------------------------------------------------------------------------
elif current_page == "progress":
    st.title("📊 Progress")
    streak = db.get_streak(conn)
    quiz_count, avg_score = db.get_quiz_stats(conn)
    c1, c2, c3 = st.columns(3)
    c1.metric("Current streak", f"{streak} day{'s' if streak != 1 else ''}")
    c2.metric("Quizzes taken", quiz_count)
    c3.metric("Average score", f"{avg_score}%" if avg_score is not None else "—")

    history = db.get_score_history(conn)
    if history:
        df = pd.DataFrame(history, columns=["timestamp", "score", "total"])
        df = df[df["total"] > 0].copy()
        if not df.empty:
            df["percent"] = (df["score"] / df["total"] * 100).round(0)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.line_chart(df.set_index("timestamp")["percent"])
    else:
        st.caption("Take a quiz to start tracking your progress here.")

    weak_topics = db.get_recent_weak_topics(conn)
    if weak_topics:
        st.markdown("**Topics to review**")
        st.write(", ".join(weak_topics))

    with st.expander("Debug: activity log"):
        active_days = db.get_active_days(conn)
        st.write(active_days if active_days else "No activity recorded yet.")


# ---------------------------------------------------------------------------
# PAGE: Key topics
# ---------------------------------------------------------------------------
elif current_page == "key_topics":
    st.title("🔑 Key Topics (ML)")
    st.caption("Uses TF-IDF (scikit-learn) to extract keywords locally — no API call needed.")
    notes_text = st.session_state.get("notes_text", "").strip()
    if st.button("Extract key topics", key="ml_btn"):
        if not notes_text:
            st.warning("Paste your notes first (Upload page).")
        else:
            keywords = extract_keywords(notes_text)
            if keywords:
                st.write(", ".join(keywords))
            else:
                st.warning("Could not extract keywords — text may be too short or consist only of common words.")
