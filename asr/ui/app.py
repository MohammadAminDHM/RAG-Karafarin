"""
Karafarin Bank – Voice & Text Chatbot
Streamlit frontend that ties together:
  • Persian ASR  (POST /transcribe  @ ASR_SERVICE_URL)
  • RAG chatbot  (POST /api/v1/query @ RAG_SERVICE_URL)
"""

import hashlib
import os

import requests
import streamlit as st

ASR_URL = os.getenv("ASR_SERVICE_URL", "http://localhost:8080")
RAG_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")

# ── Sample questions shown on empty state ─────────────────────────────────────
SAMPLE_QUESTIONS = [
    "چطور می‌توانم حساب باز کنم؟",
    "شرایط دریافت وام چیست؟",
    "نحوه انتقال وجه بین بانکی چگونه است؟",
    "ساعت کاری شعب بانک کارآفرین؟",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="دستیار هوشمند بانک کارآفرین",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap');

/* ── Targeted font: only content elements, never Streamlit chrome/icons ── */
/* Broad selectors like * or [data-testid] would break Material Symbols icon fonts */
html, body,
p, h1, h2, h3, h4, h5, h6,
input, textarea, select, label, a, li, ul, ol,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] *,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stChatInput"] textarea,
[data-testid="stNotification"] *,
[data-testid="stAlert"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondary"] *,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"] * {
    font-family: 'Vazirmatn', Tahoma, Arial, sans-serif !important;
}

.main .block-container {
    direction: rtl;
    padding-top: 1rem;
    max-width: 840px;
}

/* ── header ── */
.kf-header { text-align:center; padding:.5rem 0 .3rem; direction:rtl; }
.kf-header h1 { color:#1e3a5f; font-size:1.65rem; font-weight:700; margin-bottom:.15rem; }
.kf-header p  { color:#64748b; font-size:.88rem; }

/* ── chat RTL ── */
[data-testid="stChatMessage"],
[data-testid="stChatMessageContent"],
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] div { direction:rtl; text-align:right; }

/* ── chat input ── */
[data-testid="stChatInput"] textarea {
    direction:rtl; text-align:right;
    font-family:'Vazirmatn',Tahoma,sans-serif !important;
}

/* ── audio widget ── */
.stAudioInput label, .stAudioInput > div { direction:rtl; }

/* ── sidebar ── */
[data-testid="stSidebar"] { direction:rtl; }
[data-testid="stSidebar"] * { direction:rtl; text-align:right; }

/* ── status dot ── */
.sdot {
    display:inline-block; width:9px; height:9px;
    border-radius:50%; margin-left:6px; vertical-align:middle;
}
.sdot-ok  { background:#22c55e; }
.sdot-err { background:#ef4444; }

/* ── source card ── */
.src-card {
    direction:rtl; text-align:right;
    border-right:4px solid #e2e8f0;
    padding:8px 12px 6px 8px;
    margin-bottom:8px;
    border-radius:0 6px 6px 0;
    background:#f8fafc;
}
.src-card.hi  { border-right-color:#22c55e; }
.src-card.mid { border-right-color:#f59e0b; }
.src-card.lo  { border-right-color:#94a3b8; }
.src-q { font-weight:600; color:#1e3a5f; font-size:.87rem; line-height:1.5; }
.src-a { color:#475569; font-size:.82rem; margin-top:4px; line-height:1.5; }
.src-score-bar-wrap {
    background:#e2e8f0; border-radius:999px;
    height:4px; margin-top:6px; overflow:hidden;
}
.src-score-bar { height:4px; border-radius:999px; }
.src-score-txt { font-size:.72rem; color:#94a3b8; margin-top:2px; }

/* ── voice badge ── */
.voice-badge {
    font-size:.7rem; color:#7c3aed; background:#ede9fe;
    border-radius:999px; padding:1px 8px; margin-left:6px;
}

/* ── suggestion chips ── */
.chip-row { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:1rem; }
.chip {
    background:#f1f5f9; border:1px solid #cbd5e1; border-radius:999px;
    padding:5px 14px; font-size:.82rem; color:#1e3a5f; cursor:pointer;
    direction:rtl;
}
.chip:hover { background:#e2e8f0; }

/* ── empty state ── */
.empty-state {
    text-align:center; padding:2.5rem 1rem;
    color:#94a3b8; direction:rtl;
}
.empty-state .icon { font-size:2.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _health(base_url: str, path: str) -> bool:
    try:
        return requests.get(f"{base_url}{path}", timeout=3).status_code == 200
    except Exception:
        return False


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str | None:
    try:
        resp = requests.post(
            f"{ASR_URL}/transcribe",
            files={"file": (filename, audio_bytes)},
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json().get("transcript", "")
        st.error(f"خطا در سرویس صدا — کد {resp.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("اتصال به سرویس تبدیل صدا برقرار نشد.")
        return None
    except Exception as exc:
        st.error(f"خطا: {exc}")
        return None


def query_rag(text: str, top_k: int = 5) -> dict:
    try:
        resp = requests.post(
            f"{RAG_URL}/api/v1/query",
            json={"query": text, "top_k": top_k},
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"answer": f"خطا در سرویس — کد {resp.status_code}", "sources": []}
    except requests.exceptions.ConnectionError:
        return {"answer": "اتصال به سرویس پاسخ‌گویی برقرار نشد.", "sources": []}
    except Exception as exc:
        return {"answer": f"خطا: {exc}", "sources": []}


def add_message(role: str, content: str, mode: str = "text", sources: list = None):
    st.session_state.messages.append(
        {"role": role, "content": content, "mode": mode, "sources": sources or []}
    )


def _handle_query(text: str, mode: str = "text"):
    add_message("user", text, mode=mode)
    with st.spinner("💭 در حال دریافت پاسخ…"):
        result = query_rag(text)
    add_message(
        "assistant",
        result.get("answer", "پاسخی دریافت نشد."),
        mode=mode,
        sources=result.get("sources", []),
    )
    st.rerun()


def _render_sources(sources: list):
    """Render source cards with confidence colour-bar and clean Q/A layout."""
    for idx, src in enumerate(sources, 1):
        raw = src.get("text_preview") or src.get("text") or ""

        # Parse "Q: … | A: …" into clean parts
        if " | A: " in raw:
            q_part, a_part = raw.split(" | A: ", 1)
            question = q_part.removeprefix("Q: ").strip()
            answer   = a_part.strip()
        elif raw.startswith("Q: "):
            question = raw.removeprefix("Q: ").strip()
            answer   = ""
        else:
            question = raw.strip()
            answer   = ""

        score = float(src.get("score") or src.get("combined_score") or 0)
        pct   = min(int(score * 100), 100)

        # Colour tier
        if score >= 0.70:
            tier, bar_color = "hi",  "#22c55e"
        elif score >= 0.50:
            tier, bar_color = "mid", "#f59e0b"
        else:
            tier, bar_color = "lo",  "#94a3b8"

        q_html = f'<div class="src-q">{question}</div>'        if question else ""
        a_html = f'<div class="src-a">{answer}</div>'          if answer   else ""

        st.markdown(
            f'<div class="src-card {tier}">'
            f'<span style="font-size:.72rem;color:#94a3b8;">منبع {idx}</span>'
            f"{q_html}{a_html}"
            f'<div class="src-score-bar-wrap">'
            f'  <div class="src-score-bar" style="width:{pct}%;background:{bar_color};"></div>'
            f"</div>"
            f'<div class="src-score-txt">تطابق: {pct}٪</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ وضعیت سرویس‌ها")

    asr_ok = _health(ASR_URL, "/health")
    rag_ok = _health(RAG_URL, "/api/v1/health")

    st.markdown(
        f'<span class="sdot {"sdot-ok" if asr_ok else "sdot-err"}"></span>'
        f' تبدیل صدا — <b>{"آنلاین" if asr_ok else "آفلاین"}</b>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="sdot {"sdot-ok" if rag_ok else "sdot-err"}"></span>'
        f' پاسخ‌گویی — <b>{"آنلاین" if rag_ok else "آفلاین"}</b>',
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button("🗑️ پاک کردن تاریخچه", use_container_width=True):
        st.session_state.messages.clear()
        st.session_state.last_audio_hash = None
        st.rerun()

    st.divider()
    st.markdown(
        """
**راهنما:**
- **💬 متن** — سوال را تایپ کنید و Enter بزنید
- **🎙️ صدا** — دکمه ضبط را بزنید، فارسی صحبت کنید و دوباره بزنید
- صدا به متن تبدیل شده و پاسخ بانکی دریافت می‌شود
"""
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="kf-header">
  <h1>🏦 دستیار هوشمند بانک کارآفرین</h1>
  <p>سوالات خود را به صورت متنی یا صوتی بپرسید</p>
</div>
""",
    unsafe_allow_html=True,
)
st.divider()

# ── Empty state with suggestion chips ─────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
<div class="empty-state">
  <div class="icon">🏦</div>
  <p style="font-size:1rem; font-weight:600; color:#1e3a5f; margin-top:.5rem;">سلام! چطور می‌توانم کمک کنم؟</p>
  <p style="font-size:.85rem;">یک سوال بپرسید یا از نمونه‌های زیر انتخاب کنید:</p>
</div>
""",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    is_user = msg["role"] == "user"
    avatar  = ("🎙️" if msg.get("mode") == "voice" else "👤") if is_user else "🏦"

    with st.chat_message(msg["role"], avatar=avatar):
        badge = (
            '<span class="voice-badge">🎙️ صوتی</span>'
            if (is_user and msg.get("mode") == "voice")
            else ""
        )
        st.markdown(
            f'<div dir="rtl" style="text-align:right; line-height:1.7; font-size:.95rem;">'
            f"{badge}{msg['content']}</div>",
            unsafe_allow_html=True,
        )

        # Copy button for assistant answers
        if not is_user:
            if st.button("📋 کپی پاسخ", key=f"copy_{idx}", help="کپی متن پاسخ"):
                escaped = msg["content"].replace("`", "'")
                st.markdown(
                    f'<script>navigator.clipboard.writeText(`{escaped}`)</script>',
                    unsafe_allow_html=True,
                )

        sources = msg.get("sources") or []
        if not is_user and sources:
            toggle_key = f"src_open_{idx}"
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False
            label = (
                f"▲ بستن منابع ({len(sources)} مورد)"
                if st.session_state[toggle_key]
                else f"📚 منابع مرتبط ({len(sources)} مورد)"
            )
            if st.button(label, key=f"src_btn_{idx}", use_container_width=False):
                st.session_state[toggle_key] = not st.session_state[toggle_key]
                st.rerun()
            if st.session_state[toggle_key]:
                _render_sources(sources)

# ── Pending chip query ────────────────────────────────────────────────────────
if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None
    _handle_query(q, mode="text")

# ── Voice input ───────────────────────────────────────────────────────────────
st.markdown(
    '<p style="direction:rtl; margin-bottom:2px; font-size:.9rem; color:#475569;">🎙️ ضبط صدا (فارسی):</p>',
    unsafe_allow_html=True,
)
audio_input = st.audio_input(label=" ", label_visibility="collapsed", key="voice_recorder")

if audio_input is not None:
    audio_bytes = audio_input.read()
    if audio_bytes:
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash

            with st.spinner("🎙️ در حال تبدیل صدا به متن…"):
                transcript = transcribe_audio(
                    audio_bytes,
                    filename=getattr(audio_input, "name", "audio.wav"),
                )

            if transcript and transcript.strip():
                st.info(f"🎙️ متن تشخیص داده شده: **{transcript}**")
                _handle_query(transcript, mode="voice")
            elif transcript is not None:
                st.warning("متنی تشخیص داده نشد. لطفاً واضح‌تر و نزدیک‌تر به میکروفون صحبت کنید.")

# ── Text input ────────────────────────────────────────────────────────────────
user_text = st.chat_input("سوال خود را اینجا بنویسید…")
if user_text and user_text.strip():
    _handle_query(user_text, mode="text")
