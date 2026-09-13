import io
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Tuple

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# APP CONFIG
# ============================================================

APP_NAME = "Digital Justice"
TAGLINE = "Know Your Rights. Navigate the Digital World."
NCCIA_URL = "https://complaint.nccia.gov.pk/"
DEFAULT_MODEL = "openai/gpt-oss-120b"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS ONLY
# IMPORTANT: Visible UI below is built with native Streamlit
# widgets/markdown. No visible HTML is used, so HTML source
# cannot accidentally appear as text in the interface.
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }

    .stApp {
        position: relative;
        background:
            radial-gradient(circle at 10% 5%, rgba(200,164,93,.08), transparent 25%),
            radial-gradient(circle at 90% 15%, rgba(11,31,51,.06), transparent 30%),
            linear-gradient(180deg, #faf9f6 0%, #f5f5f2 52%, #eef1f0 100%);
        color: #252525;
    }

    /* Subtle fixed wave texture — decorative only, never blocks the app. */
    .stApp::before {
        content: "";
        position: fixed;
        left: -10%;
        right: -10%;
        bottom: -6%;
        height: 52vh;
        min-height: 320px;
        pointer-events: none;
        z-index: 0;
        opacity: .55;
        background:
            radial-gradient(ellipse 90% 42% at 50% 100%,
                transparent 0 42%,
                rgba(11,31,51,.035) 42.4% 42.8%,
                transparent 43.2% 50%,
                rgba(200,164,93,.05) 50.4% 50.8%,
                transparent 51.2% 58%,
                rgba(11,31,51,.03) 58.4% 58.8%,
                transparent 59.2% 66%,
                rgba(200,164,93,.04) 66.4% 66.8%,
                transparent 67.2% 100%);
        transform: rotate(-2deg) scale(1.12);
    }

    .main .block-container {
        position: relative;
        z-index: 2;
    }

    /* Fixed, medium-size Digital Justice watermark. */
    .dj-watermark {
        position: fixed;
        left: 50%;
        top: 54%;
        width: min(430px, 48vw);
        height: min(430px, 48vw);
        min-width: 300px;
        min-height: 300px;
        transform: translate(-50%, -50%);
        pointer-events: none;
        z-index: 1;
        opacity: .075;
    }

    .dj-watermark svg {
        width: 100%;
        height: 100%;
        overflow: visible;
    }

    .dj-watermark .outer-ring {
        fill: none;
        stroke: #c8a45d;
        stroke-width: 2.4;
    }

    .dj-watermark .inner-ring {
        fill: none;
        stroke: #c8a45d;
        stroke-width: 1.2;
    }

    .dj-watermark .phrase {
        fill: #b7924f;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.9px;
    }

    .dj-watermark .scale-circle {
        fill: rgba(200,164,93,.025);
        stroke: #c8a45d;
        stroke-width: 1.4;
    }

    .dj-watermark .scale {
        fill: #c8a45d;
        font-size: 68px;
        font-family: "Segoe UI Symbol", "Noto Sans Symbols 2", sans-serif;
        text-anchor: middle;
        dominant-baseline: middle;
    }

    .dj-watermark .title {
        fill: #0b1f33;
        font-size: 17px;
        font-weight: 900;
        letter-spacing: 3.5px;
        text-anchor: middle;
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081b2d 0%, #0b1f33 55%, #102b43 100%);
        border-right: 1px solid rgba(200,164,93,.22);
    }

    section[data-testid="stSidebar"] * {
        color: #f7f6f2;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #f7f6f2;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #c8a45d !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.12);
    }

    /* Sidebar radio navigation */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 9px;
        padding: 6px 8px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,.06);
    }

    /* Sidebar buttons: plain white, black text, no hover recolor */
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stLinkButton > a,
    section[data-testid="stSidebar"] .stButton > button *,
    section[data-testid="stSidebar"] .stLinkButton > a * {
        background: #ffffff !important;
        color: #111111 !important;
        border-color: #ffffff !important;
        box-shadow: none !important;
        text-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] .stButton > button:focus,
    section[data-testid="stSidebar"] .stButton > button:focus-visible,
    section[data-testid="stSidebar"] .stButton > button:active,
    section[data-testid="stSidebar"] .stLinkButton > a:hover,
    section[data-testid="stSidebar"] .stLinkButton > a:focus,
    section[data-testid="stSidebar"] .stLinkButton > a:focus-visible,
    section[data-testid="stSidebar"] .stLinkButton > a:active,
    section[data-testid="stSidebar"] .stButton > button:hover *,
    section[data-testid="stSidebar"] .stButton > button:focus *,
    section[data-testid="stSidebar"] .stButton > button:focus-visible *,
    section[data-testid="stSidebar"] .stButton > button:active *,
    section[data-testid="stSidebar"] .stLinkButton > a:hover *,
    section[data-testid="stSidebar"] .stLinkButton > a:focus *,
    section[data-testid="stSidebar"] .stLinkButton > a:focus-visible *,
    section[data-testid="stSidebar"] .stLinkButton > a:active * {
        background: #ffffff !important;
        color: #111111 !important;
        border-color: #ffffff !important;
        box-shadow: none !important;
        transform: none !important;
        text-shadow: none !important;
    }

    /* ---------- All ordinary buttons ---------- */
    .stButton > button,
    .stLinkButton > a,
    .stDownloadButton > button {
        background: #ffffff !important;
        color: #252525 !important;
        border: 1px solid #252525 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 42px;
        box-shadow: none !important;
        transition: none !important;
    }

    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:focus-visible,
    .stButton > button:active,
    .stLinkButton > a:hover,
    .stLinkButton > a:focus,
    .stLinkButton > a:focus-visible,
    .stLinkButton > a:active,
    .stDownloadButton > button:hover,
    .stDownloadButton > button:focus,
    .stDownloadButton > button:focus-visible,
    .stDownloadButton > button:active {
        background: #ffffff !important;
        color: #252525 !important;
        border-color: #252525 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* Suggested questions */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: #68737d !important;
        color: #ffffff !important;
        border-color: #68737d !important;
        text-align: left !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button:hover,
    div[data-testid="stHorizontalBlock"] .stButton > button:focus,
    div[data-testid="stHorizontalBlock"] .stButton > button:focus-visible,
    div[data-testid="stHorizontalBlock"] .stButton > button:active {
        background: #68737d !important;
        color: #ffffff !important;
        border-color: #68737d !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* ---------- Native content styling ---------- */
    /* ---------- Main headings ---------- */
    h1, h2, h3 {
        color: #0b1f33;
    }

    [data-testid="stCaptionContainer"] p {
        color: #68737d;
    }

    /* ---------- Inputs ---------- */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        border-radius: 10px !important;
        color: #252525 !important;
        background: #ffffff !important;
        border: 1px solid #252525 !important;
    }

    [data-baseweb="select"] {
        border-radius: 10px !important;
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #252525 !important;
        border-radius: 15px;
        margin-bottom: 11px;
        box-shadow: 0 4px 14px rgba(11,31,51,.06);
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: #252525 !important;
    }

    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #252525 !important;
        border-radius: 13px !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #252525 !important;
        background: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #68737d !important;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid rgba(11,31,51,.10);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 7px 22px rgba(11,31,51,.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #68737d !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0b1f33 !important;
    }

    /* ---------- Expanders / cards ---------- */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(11,31,51,.12);
        border-radius: 13px;
        background: rgba(255,255,255,.82);
    }

    .footer {
        text-align: center;
        color: #87919a;
        font-size: .72rem;
        padding: 30px 0 4px;
    }

    @media (max-width: 700px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .dj-watermark {
            width: 300px;
            height: 300px;
            min-width: 300px;
            min-height: 300px;
            top: 56%;
        }
        .dj-watermark .phrase {
            font-size: 9.5px;
            letter-spacing: 1.2px;
        }
        .dj-watermark .scale {
            font-size: 55px;
        }
        .dj-watermark .title {
            font-size: 13px;
            letter-spacing: 2.7px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Decorative watermark only; application logic remains untouched.
st.markdown(
    """
    <div class="dj-watermark" aria-hidden="true">
      <svg viewBox="0 0 500 500" role="presentation">
        <defs>
          <path id="djPhrasePath" d="M 250,250 m -210,0 a 210,210 0 1,1 420,0 a 210,210 0 1,1 -420,0" />
        </defs>
        <circle class="outer-ring" cx="250" cy="250" r="220" />
        <circle class="inner-ring" cx="250" cy="250" r="184" />
        <text class="phrase">
          <textPath href="#djPhrasePath" startOffset="50%" text-anchor="middle">
            LEGAL-TECH • RAG • DIGITAL RIGHTS • PRIVACY • SECURITY • JUSTICE •
          </textPath>
        </text>
        <circle class="scale-circle" cx="250" cy="250" r="72" />
        <text class="scale" x="250" y="250">⚖</text>
        <text class="title" x="250" y="355">DIGITAL JUSTICE</text>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Chunk:
    chunk_id: int
    source: str
    text: str
    start: int
    end: int


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "document_meta" not in st.session_state:
    st.session_state.document_meta = {}

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = []

if "settings" not in st.session_state:
    st.session_state.settings = {
        "chunk_size": 900,
        "chunk_overlap": 150,
        "top_k": 5,
        "min_score": 0.08,
        "temperature": 0.2,
        "max_tokens": 1800,
        "model": DEFAULT_MODEL,
        "show_sources": True,
        "strict_grounding": True,
    }


# ============================================================
# HELPERS
# ============================================================

def get_groq_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(file_name: str, data: bytes) -> str:
    ext = os.path.splitext(file_name.lower())[1]

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return clean_text("\n\n".join(parts))

    if ext == ".docx":
        doc = Document(io.BytesIO(data))
        return clean_text("\n\n".join(p.text for p in doc.paragraphs))

    return clean_text(data.decode("utf-8", errors="ignore"))


def make_chunks(text: str, chunk_size: int, overlap: int):
    if not text:
        return []

    chunk_size = max(100, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        target_end = min(start + chunk_size, text_len)

        if target_end < text_len:
            candidates = [
                text.rfind("\n\n", start, target_end),
                text.rfind(". ", start, target_end),
                text.rfind("! ", start, target_end),
                text.rfind("? ", start, target_end),
                text.rfind(" ", start, target_end),
            ]
            best = max(candidates)

            if best > start + int(chunk_size * 0.55):
                if text[best:best + 2] == ". ":
                    target_end = best + 2
                else:
                    target_end = best + 1

        piece = text[start:target_end].strip()

        if piece:
            chunks.append((piece, start, target_end))

        if target_end >= text_len:
            break

        start = max(target_end - overlap, start + 1)

    return chunks


def rebuild_index():
    all_chunks = []
    cid = 0

    for source, text in st.session_state.documents.items():
        generated = make_chunks(
            text,
            st.session_state.settings["chunk_size"],
            st.session_state.settings["chunk_overlap"],
        )

        for piece, start, end in generated:
            all_chunks.append(
                Chunk(
                    chunk_id=cid,
                    source=source,
                    text=piece,
                    start=start,
                    end=end,
                )
            )
            cid += 1

    st.session_state.chunks = all_chunks


def calculate_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@st.cache_data(show_spinner=False)
def create_tfidf_matrix(texts: Tuple[str, ...]):
    if not texts:
        return None, None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_context(query: str, top_k: int, minimum_score: float):
    if not st.session_state.chunks:
        return []

    texts = tuple(chunk.text for chunk in st.session_state.chunks)
    vectorizer, matrix = create_tfidf_matrix(texts)

    if vectorizer is None:
        return []

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    ranked = scores.argsort()[::-1][:max(1, int(top_k))]

    results = []
    for index in ranked:
        score = float(scores[index])
        if score >= minimum_score:
            results.append((st.session_state.chunks[index], score))

    return results


def format_context(results) -> str:
    if not results:
        return "No relevant document context was retrieved."

    blocks = []
    for i, (chunk, score) in enumerate(results, start=1):
        blocks.append(
            f"[Source {i}: {chunk.source} | similarity={score:.3f}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def detect_cybercrime(text: str) -> bool:
    patterns = [
        r"\bblackmail\b",
        r"\bcyberbullying\b",
        r"\bonline harassment\b",
        r"\bonline abuse\b",
        r"\bidentity theft\b",
        r"\bphishing\b",
        r"\bfraud\b",
        r"\bscam\b",
        r"\bextortion\b",
        r"\bimpersonat(?:e|ion)\b",
        r"\bprivacy violation\b",
        r"\bprivate (?:photos|pictures|videos)\b",
        r"\baccount hacked\b",
        r"\bcybercrime\b",
        r"\bonline threat\b",
        r"\bthreatening me online\b",
    ]
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def generate_answer(question: str, retrieved_results):
    api_key = get_groq_key()

    if not api_key:
        return (
            "⚠️ Groq API key is not configured.\n\n"
            "Please add `GROQ_API_KEY` to Streamlit Secrets "
            "or your environment."
        )

    context = format_context(retrieved_results)

    if st.session_state.settings["strict_grounding"]:
        grounding_rule = """
Answer primarily from the retrieved context.

If the uploaded knowledge base does not contain enough
information to answer the question, clearly say so.

Never invent legal facts.
"""
    else:
        grounding_rule = """
Use the retrieved context as the primary source.

You may provide clearly-labelled general information
when the uploaded documents are insufficient.
"""

    system_prompt = f"""
You are Digital Justice, a legal-information RAG assistant
focused on digital rights, cybercrime, privacy and digital
safety, especially for users in Pakistan.

IMPORTANT SAFETY RULES:

You provide general legal information.
You are not a lawyer.
You do not provide legal representation.

Do not invent laws.
Do not invent PECA sections.
Do not invent penalties.
Do not fabricate cases or citations.
Do not claim a complaint was registered.
Do not claim an investigation was started.

Never request passwords, OTPs, PINs, CNIC numbers,
bank credentials, or private photos/videos.

For serious matters, recommend consulting a qualified lawyer.
For immediate physical danger, advise contacting appropriate
emergency services.

ANSWER STYLE:
Use simple language.
Use short paragraphs.
Use headings when useful.
Use bullet points for practical steps.

GROUNDING:
{grounding_rule}

RETRIEVED KNOWLEDGE BASE:
---
{context}
---
"""

    if detect_cybercrime(question):
        system_prompt += f"""
When appropriate, include a short "Reporting option" section.
Mention the official NCCIA complaint portal:
{NCCIA_URL}

Never say that Digital Justice submitted or registered a complaint.
"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=st.session_state.settings["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=float(st.session_state.settings["temperature"]),
            max_completion_tokens=int(st.session_state.settings["max_tokens"]),
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        return (
            "⚠️ Unable to generate an answer right now.\n\n"
            f"Technical detail: {exc}"
        )


def process_question(question: str):
    question = question.strip()
    if not question:
        return

    results = retrieve_context(
        question,
        st.session_state.settings["top_k"],
        st.session_state.settings["min_score"],
    )

    answer = generate_answer(question, results)
    st.session_state.last_retrieval = results

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": results}
    )


def clear_conversation():
    st.session_state.messages = []
    st.session_state.last_retrieval = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚖️ Digital Justice")
    st.caption("LEGAL-TECH • RAG • DIGITAL RIGHTS")
    st.divider()

    page = st.radio(
        "Workspace",
        [
            "⚖️ Legal Chat",
            "📚 Knowledge Base",
            "🧩 Chunk Manager",
            "⚙️ RAG Settings",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("CYBERCRIME")

    st.markdown("**Need to report a cybercrime?**")
    st.caption("Use the official NCCIA complaint portal.")

    st.link_button(
        "Open NCCIA Complaint Portal ↗",
        NCCIA_URL,
        use_container_width=True,
    )

    st.divider()
    st.caption("RAG STATUS")

    status_box = st.container(border=True)
    with status_box:
        st.write(f"**Documents:** {len(st.session_state.documents)}")
        st.write(f"**Indexed chunks:** {len(st.session_state.chunks)}")
        st.write(f"**Model:** {st.session_state.settings['model']}")

    st.write("")

    if st.button("Clear Conversation", use_container_width=True):
        clear_conversation()
        st.rerun()


# ============================================================
# TOP BRAND
# ============================================================

st.markdown("## ⚖️  Digital Justice")
st.caption("Know Your Rights. Navigate the Digital World.")


# ============================================================
# LEGAL CHAT
# ============================================================

if page == "⚖️ Legal Chat":
    with st.container(border=True):
        st.caption("DIGITAL RIGHTS • LEGAL INFORMATION • RAG")
        st.markdown("## Know Your Rights.")
        st.write(
            "Ask questions about digital rights, cybercrime, online harassment, "
            "privacy, digital safety and the legal information contained in your "
            "knowledge base. Digital Justice retrieves relevant sources before "
            "generating an answer."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", len(st.session_state.documents))
    m2.metric("Indexed Chunks", len(st.session_state.chunks))
    m3.metric("Top-K", st.session_state.settings["top_k"])
    m4.metric("Retrieval", "TF-IDF")

    st.markdown("### Suggested questions")

    suggestions = [
        "What is PECA and what does it cover?",
        "What should I do if someone is blackmailing me online?",
        "What are my digital privacy rights in Pakistan?",
        "Someone hacked my account. What should I do?",
        "How can I report a cybercrime?",
        "What evidence should I preserve after online harassment?",
    ]

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(
                suggestion,
                key=f"suggestion_{i}",
                use_container_width=True,
            ):
                process_question(suggestion)
                st.rerun()
    if not st.session_state.documents:
        st.info(
            "Knowledge Base is currently empty. "
            "Upload legal documents from the Knowledge Base section "
            "for grounded RAG answers."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and st.session_state.settings["show_sources"]
                and message.get("sources")
            ):
                with st.expander("View retrieved sources"):
                    for chunk, score in message["sources"]:
                        st.markdown(f"**{chunk.source}**")
                        st.caption(f"Similarity: {score:.3f}")
                        st.write(chunk.text)

    question = st.chat_input(
        "Ask Digital Justice about your digital rights..."
    )
    if question:
        process_question(question)
        st.rerun()

    st.caption(
        "Digital Justice • General legal information only • "
        "Official cybercrime reporting portal: NCCIA"
    )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

elif page == "📚 Knowledge Base":
    st.markdown("## Knowledge Base")
    st.caption("Upload trusted legal documents to ground the assistant.")

    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, Markdown and CSV.",
    )

    if uploaded:
        added = 0
        skipped = 0

        for file in uploaded:
            data = file.getvalue()
            file_hash = calculate_file_hash(data)

            already_exists = any(
                meta.get("hash") == file_hash
                for meta in st.session_state.document_meta.values()
            )

            if already_exists:
                skipped += 1
                continue

            try:
                text = extract_text(file.name, data)

                if not text:
                    st.warning(f"Could not extract readable text from {file.name}.")
                    continue

                # Avoid overwriting a document with the same filename.
                source_name = file.name
                if source_name in st.session_state.documents:
                    stem, ext = os.path.splitext(file.name)
                    source_name = f"{stem} ({file_hash[:8]}){ext}"

                st.session_state.documents[source_name] = text
                st.session_state.document_meta[source_name] = {
                    "hash": file_hash,
                    "characters": len(text),
                }
                added += 1

            except Exception as exc:
                st.error(f"Failed to process {file.name}: {exc}")

        if added:
            rebuild_index()
            create_tfidf_matrix.clear()
            st.success(f"{added} document(s) added and indexed.")

        if skipped:
            st.info(f"{skipped} duplicate document(s) skipped.")

    st.divider()
    st.markdown("### Current documents")

    if not st.session_state.documents:
        st.warning("No documents uploaded yet.")
    else:
        for source, text in list(st.session_state.documents.items()):
            c1, c2, c3 = st.columns([5, 2, 1])

            with c1:
                st.markdown(f"**{source}**")

            with c2:
                st.caption(f"{len(text):,} characters")

            with c3:
                if st.button("Remove", key=f"remove_{source}"):
                    st.session_state.documents.pop(source, None)
                    st.session_state.document_meta.pop(source, None)
                    rebuild_index()
                    create_tfidf_matrix.clear()
                    st.rerun()

    st.divider()
    st.markdown("### RAG Pipeline")

    pipeline = [
        ("01", "Upload", "Add trusted legal documents."),
        ("02", "Extract", "Extract readable text."),
        ("03", "Chunk", "Split documents into searchable chunks."),
        ("04", "Index", "Build TF-IDF representations."),
        ("05", "Retrieve", "Find the most relevant chunks."),
        ("06", "Generate", "Ask the Groq model using retrieved context."),
        ("07", "Answer", "Return a grounded legal-information response."),
    ]

    for number, title, description in pipeline:
        with st.expander(f"{number}  {title}"):
            st.write(description)


# ============================================================
# CHUNK MANAGER
# ============================================================

elif page == "🧩 Chunk Manager":
    st.markdown("## Chunk Manager")
    st.caption("Inspect exactly what the retrieval layer can search.")

    average_size = (
        sum(len(chunk.text) for chunk in st.session_state.chunks)
        / len(st.session_state.chunks)
        if st.session_state.chunks
        else 0
    )

    a, b, c = st.columns(3)
    a.metric("Documents", len(st.session_state.documents))
    b.metric("Chunks", len(st.session_state.chunks))
    c.metric("Average Chunk", f"{average_size:.0f} chars")

    if not st.session_state.chunks:
        st.info("No chunks available. Upload documents first.")
    else:
        sources = ["All sources"] + sorted(
            {chunk.source for chunk in st.session_state.chunks}
        )

        selected_source = st.selectbox("Filter by source", sources)

        visible_chunks = st.session_state.chunks
        if selected_source != "All sources":
            visible_chunks = [
                chunk
                for chunk in visible_chunks
                if chunk.source == selected_source
            ]

        st.caption(f"Showing {len(visible_chunks)} chunk(s)")

        for chunk in visible_chunks:
            with st.expander(
                f"Chunk {chunk.chunk_id} • {chunk.source} • {len(chunk.text)} chars"
            ):
                st.caption(
                    f"Character range: {chunk.start} → {chunk.end}"
                )
                st.write(chunk.text)


# ============================================================
# RAG SETTINGS
# ============================================================

elif page == "⚙️ RAG Settings":
    st.markdown("## RAG Settings")
    st.caption("Tune chunking, retrieval, generation and transparency.")

    st.markdown("### Chunking")
    c1, c2 = st.columns(2)

    with c1:
        chunk_size = st.slider(
            "Chunk size",
            300,
            2500,
            int(st.session_state.settings["chunk_size"]),
            50,
        )

    with c2:
        max_overlap = min(800, chunk_size // 2)
        current_overlap = min(
            int(st.session_state.settings["chunk_overlap"]),
            max_overlap,
        )
        chunk_overlap = st.slider(
            "Chunk overlap",
            0,
            max_overlap,
            current_overlap,
            25,
        )

    st.markdown("### Retrieval")
    r1, r2 = st.columns(2)

    with r1:
        top_k = st.slider(
            "Top-K retrieved chunks",
            1,
            10,
            int(st.session_state.settings["top_k"]),
        )

    with r2:
        min_score = st.slider(
            "Minimum similarity score",
            0.0,
            0.60,
            float(st.session_state.settings["min_score"]),
            0.01,
        )

    st.markdown("### Generation")
    g1, g2 = st.columns(2)

    with g1:
        model = st.selectbox(
            "Groq model",
            [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
            ],
            index=(
                0
                if st.session_state.settings["model"] == "openai/gpt-oss-120b"
                else 1
            ),
        )

    with g2:
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            float(st.session_state.settings["temperature"]),
            0.05,
        )

    max_tokens = st.slider(
        "Maximum output tokens",
        400,
        6000,
        int(st.session_state.settings["max_tokens"]),
        100,
    )

    st.markdown("### Safety & Transparency")
    s1, s2 = st.columns(2)

    with s1:
        show_sources = st.toggle(
            "Show retrieved sources",
            value=bool(st.session_state.settings["show_sources"]),
        )

    with s2:
        strict_grounding = st.toggle(
            "Strict RAG grounding",
            value=bool(st.session_state.settings["strict_grounding"]),
            help="Prefer answers supported by uploaded documents.",
        )

    st.divider()

    if st.button(
        "Apply Settings & Rebuild Index",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.settings.update(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "min_score": min_score,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": model,
                "show_sources": show_sources,
                "strict_grounding": strict_grounding,
            }
        )

        rebuild_index()
        create_tfidf_matrix.clear()

        st.success("RAG settings applied and index rebuilt successfully.")
        st.rerun()


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":
    st.markdown("## About Digital Justice")
    st.caption("A legal-information RAG workspace for navigating the digital world.")

    st.markdown(
        """
### What is Digital Justice?

Digital Justice is a Retrieval-Augmented Generation (RAG)
application designed to help users understand digital rights,
cybercrime, privacy and online safety.

Users can upload trusted legal documents and the retrieval
system finds relevant sections before the Groq-powered AI
generates an explanation.

### Core Technologies

| Layer | Technology |
|---|---|
| Interface | Streamlit |
| LLM | Groq |
| Retrieval | TF-IDF + Cosine Similarity |
| Documents | PDF, DOCX, TXT, MD, CSV |
| Architecture | Retrieval-Augmented Generation |

### RAG Architecture

Documents → Extraction → Chunking → TF-IDF Index →
Retrieval → Groq LLM → Answer

### Important Limitation

Digital Justice provides general legal information.

It is not a replacement for a qualified lawyer, court,
police authority or official government advice.

The quality of a grounded response depends on the quality
and coverage of the uploaded knowledge base.
"""
    )

    st.link_button(
        "Open Official NCCIA Complaint Portal ↗",
        NCCIA_URL,
        use_container_width=True,
    )

    st.caption("Digital Justice • Know Your Rights. Navigate the Digital World.")
