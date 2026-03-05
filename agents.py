"""
agents.py  —  NexusRAG complete agent stack (single file, no subpackage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Everything that was in backend/ is here:
  ApiKeyPool · get_llm · get_embeddings
  RAGAgent · VideoRAGAgent · DataAnalysisAgent
  CodeGeneratorAgent · DeepResearcherAgent · GeneralChatbotAgent
  MultiAgentOrchestrator

Import in app.py:
    from agents import MultiAgentOrchestrator, key_pool
"""

from __future__ import annotations

# ── stdlib ─────────────────────────────────────────────────────────────────────
import io
import base64
import json
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

# ── third-party: plotting ──────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── LangChain ──────────────────────────────────────────────────────────────────
from langchain.agents import create_agent
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain.tools import StructuredTool
except ImportError:
    from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled,
    )
    TRANSCRIPT_API_OK = True
except ImportError:
    TRANSCRIPT_API_OK = False

try:
    import yt_dlp as _yt_dlp
    YTDLP_OK = True
except ImportError:
    YTDLP_OK = False

try:
    import requests as _requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from duckduckgo_search import DDGS
    DDGS_OK = True
except ImportError:
    DDGS_OK = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — API KEY POOL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_MODEL       = "gemini-2.5-flash"
EMBED_MODEL         = "models/embedding-001"
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200
TOKEN_LIMIT_PER_KEY = 900_000


@dataclass
class _KeySlot:
    key:         str
    tokens_used: int   = 0
    errors:      int   = 0
    last_error:  float = 0.0
    exhausted:   bool  = False

    @property
    def is_available(self) -> bool:
        if self.exhausted:
            return False
        if self.errors >= 3 and (time.time() - self.last_error) < 60:
            return False
        return True

    def record_tokens(self, n: int) -> None:
        self.tokens_used += n
        if self.tokens_used >= TOKEN_LIMIT_PER_KEY:
            self.exhausted = True

    def record_error(self) -> None:
        self.errors    += 1
        self.last_error = time.time()

    def reset(self) -> None:
        self.tokens_used = 0
        self.errors      = 0
        self.exhausted   = False
        self.last_error  = 0.0


class ApiKeyPool:
    """Random-choice API key pool. set_keys([...]) once at startup."""

    def __init__(self) -> None:
        self._slots:      List[_KeySlot] = []
        self._active_idx: int            = 0
        self._lock:       threading.Lock = threading.Lock()

    def set_keys(self, keys: List[str]) -> None:
        with self._lock:
            self._slots = [_KeySlot(key=k.strip()) for k in keys if k.strip()]
            if not self._slots:
                raise ValueError("Provide at least one API key.")
            self._active_idx = random.randrange(len(self._slots))

    def current_key(self) -> str:
        with self._lock:
            if not self._slots:
                raise RuntimeError("Call key_pool.set_keys([...]) first.")
            if not self._slots[self._active_idx].is_available:
                self._rotate()
            return self._slots[self._active_idx].key

    def _rotate(self) -> None:
        available = [i for i, s in enumerate(self._slots)
                     if s.is_available and i != self._active_idx]
        if available:
            self._active_idx = random.choice(available)
        else:
            for s in self._slots:
                s.reset()
            self._active_idx = random.randrange(len(self._slots))

    def report_usage(self, tokens: int) -> None:
        with self._lock:
            if not self._slots:
                return
            self._slots[self._active_idx].record_tokens(tokens)
            if not self._slots[self._active_idx].is_available:
                self._rotate()

    def report_error(self) -> None:
        with self._lock:
            if not self._slots:
                return
            self._slots[self._active_idx].record_error()
            if not self._slots[self._active_idx].is_available:
                self._rotate()

    def status(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "index":       i,
                    "active":      i == self._active_idx,
                    "tokens_used": s.tokens_used,
                    "token_limit": TOKEN_LIMIT_PER_KEY,
                    "pct_used":    round(s.tokens_used / TOKEN_LIMIT_PER_KEY * 100, 1),
                    "exhausted":   s.exhausted,
                    "errors":      s.errors,
                    "available":   s.is_available,
                    "key_preview": s.key[:8] + "..." if len(s.key) > 8 else s.key,
                }
                for i, s in enumerate(self._slots)
            ]

    def key_count(self) -> int:
        return len(self._slots)


# global singleton
key_pool = ApiKeyPool()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — LLM / EMBEDDINGS FACTORIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_llm(temperature: float = 0.1, model: str = DEFAULT_MODEL) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key_pool.current_key(),
        temperature=temperature,
        convert_system_message_to_human=True,
        max_retries=2,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=key_pool.current_key(),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — DOCUMENT / VECTOR STORE HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_documents(file_paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for fp in file_paths:
        ext = Path(fp).suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(fp)
            elif ext == ".txt":
                loader = TextLoader(fp, encoding="utf-8")
            elif ext == ".csv":
                loader = CSVLoader(fp)
            elif ext in (".doc", ".docx"):
                loader = UnstructuredWordDocumentLoader(fp)
            else:
                loader = TextLoader(fp, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as exc:
            docs.append(Document(
                page_content=f"[Error loading {fp}]: {exc}",
                metadata={"source": fp},
            ))
    return docs


def build_vectorstore(docs: List[Document]) -> FAISS:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — CHART HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_dark_theme(ax, fig) -> None:
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — RAG AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _RagState:
    vectorstore = None
    sources: List[str] = []

_rag_state = _RagState()


class _IngestInput(BaseModel):
    file_paths: List[str] = Field(description="Paths to PDF/DOCX/TXT/CSV files.")

class _QueryInput(BaseModel):
    question: str = Field(description="Question to answer from documents.")

class _ListSourcesInput(BaseModel):
    pass


def _rag_ingest(file_paths: List[str]) -> str:
    docs = load_documents(file_paths)
    if not docs:
        return "⚠️ No documents loaded."
    _rag_state.vectorstore = build_vectorstore(docs)
    _rag_state.sources = list({d.metadata.get("source", "?") for d in docs})
    return f"✅ Ingested {len(docs)} chunks from {len(file_paths)} file(s)."


def _rag_query(question: str) -> str:
    if _rag_state.vectorstore is None:
        return json.dumps({"answer": "⚠️ No documents ingested yet.", "sources": []})
    docs    = _rag_state.vectorstore.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata.get("source", "?") for d in docs})
    prompt  = (
        "You are a precise document assistant. Answer ONLY from the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = get_llm(0.1).invoke([HumanMessage(content=prompt)]).content
    return json.dumps({"answer": answer, "sources": sources})


def _rag_list_sources() -> str:
    return ("No docs." if not _rag_state.sources
            else "Sources:\n" + "\n".join(f"  • {s}" for s in _rag_state.sources))


_rag_tools = [
    StructuredTool.from_function(func=_rag_ingest, name="ingest_documents",
        description="Load and index PDF/DOCX/TXT/CSV files.", args_schema=_IngestInput),
    StructuredTool.from_function(func=_rag_query, name="query_documents",
        description="Answer a question from indexed documents.", args_schema=_QueryInput),
    StructuredTool.from_function(func=_rag_list_sources, name="list_sources",
        description="List loaded document sources.", args_schema=_ListSourcesInput),
]


def _make_rag_agent():
    return create_agent(llm=get_llm(0.1), tools=_rag_tools, verbose=True, handle_parsing_errors=True, max_iterations=6)


class RAGAgent:
    name = "RAG Agent"
    def __init__(self): self._ex = None

    @property
    def executor(self):
        if self._ex is None: self._ex = _make_rag_agent()
        return self._ex

    def ingest(self, file_paths): return _rag_ingest(file_paths)

    def query(self, question):
        raw = _rag_query(question)
        try: return json.loads(raw)
        except Exception: return {"answer": raw, "sources": _rag_state.sources}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — YOUTUBE RAG AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _YTState:
    vectorstore = None
    video_id = video_url = title = channel = description = duration = thumbnail = ""
    transcript_chunks: List[Dict] = []
    full_transcript: str = ""

_yt_state = _YTState()


def _yt_extract_id(url: str) -> str:
    url = url.strip()
    if re.match(r'^[A-Za-z0-9_\-]{11}$', url):
        return url
    m = re.search(r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_\-]{11})', url)
    if m:
        return m.group(1)
    qs = parse_qs(urlparse(url).query)
    if 'v' in qs:
        return qs['v'][0]
    raise ValueError(f"Cannot extract YouTube video ID from: {url!r}")


def _yt_secs(s: float) -> str:
    s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class _FetchYTInput(BaseModel):
    youtube_url: str = Field(description="YouTube URL or bare video ID.")

class _FetchTransInput(BaseModel):
    language: str = Field(default="en", description="Transcript language code.")

class _IndexTransInput(BaseModel):
    chunk_size_seconds: int = Field(default=60)

class _QueryYTInput(BaseModel):
    question: str = Field(description="Question about the video.")

class _SummarizeYTInput(BaseModel):
    style: str = Field(default="detailed", description="brief | detailed | bullets")

class _VideoInfoInput(BaseModel):
    pass


def _yt_fetch(youtube_url: str) -> str:
    try:
        vid = _yt_extract_id(youtube_url)
    except ValueError as e:
        return f"❌ {e}"
    _yt_state.video_id  = vid
    _yt_state.video_url = f"https://www.youtube.com/watch?v={vid}"
    _yt_state.thumbnail = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
    if YTDLP_OK:
        try:
            with _yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(_yt_state.video_url, download=False)
            _yt_state.title    = info.get("title", "Unknown")
            _yt_state.channel  = info.get("uploader", info.get("channel", "Unknown"))
            _yt_state.description = (info.get("description", "") or "")[:1000]
            _yt_state.duration = _yt_secs(info.get("duration", 0) or 0)
            return f"✅ {_yt_state.title} | {_yt_state.channel} | {_yt_state.duration}"
        except Exception:
            pass
    if REQUESTS_OK:
        try:
            r = _requests.get(
                f"https://www.youtube.com/oembed?url={_yt_state.video_url}&format=json",
                timeout=10)
            if r.status_code == 200:
                d = r.json()
                _yt_state.title   = d.get("title", "Unknown")
                _yt_state.channel = d.get("author_name", "Unknown")
                return f"✅ {_yt_state.title} | {_yt_state.channel}"
        except Exception:
            pass
    _yt_state.title = f"YouTube Video ({vid})"
    return f"✅ Video ID: {vid}"


def _yt_transcript(language: str = "en") -> str:
    if not _yt_state.video_id:
        return "❌ Fetch video metadata first."
    if not TRANSCRIPT_API_OK:
        return "❌ Install: pip install youtube-transcript-api"
    try:
        tlist = YouTubeTranscriptApi.list_transcripts(_yt_state.video_id)
        t = None
        for getter in [
            lambda: tlist.find_manually_created_transcript([language]),
            lambda: tlist.find_generated_transcript([language]),
            lambda: list(tlist)[0] if list(tlist) else None,
        ]:
            try: t = getter()
            except Exception: pass
            if t: break
        if t is None:
            return f"❌ No transcript for {_yt_state.video_id}"
        chunks = t.fetch()
        _yt_state.transcript_chunks = chunks
        lines = [f"[{_yt_secs(c['start'])}] {c['text']}" for c in chunks]
        _yt_state.full_transcript = "\n".join(lines)
        return f"✅ {len(chunks)} segments fetched."
    except TranscriptsDisabled:
        return "❌ Transcripts disabled for this video."
    except NoTranscriptFound:
        return f"❌ No transcript in '{language}'."
    except Exception as e:
        return f"❌ {e}"


def _yt_index(chunk_size_seconds: int = 60) -> str:
    if not _yt_state.transcript_chunks:
        return "❌ Fetch transcript first."
    docs, cur_text, cur_start, cur_end = [], [], None, 0.0
    for seg in _yt_state.transcript_chunks:
        s, d, txt = seg.get("start", 0), seg.get("duration", 0), seg.get("text", "").strip()
        if cur_start is None: cur_start = s
        cur_text.append(txt); cur_end = s + d
        if (cur_end - cur_start) >= chunk_size_seconds:
            docs.append(Document(
                page_content=f"[{_yt_secs(cur_start)} → {_yt_secs(cur_end)}] {' '.join(cur_text)}",
                metadata={"source": _yt_state.video_url, "title": _yt_state.title,
                          "start_sec": cur_start, "timestamp": _yt_secs(cur_start)}))
            cur_text, cur_start = [], None
    if cur_text and cur_start is not None:
        docs.append(Document(
            page_content=f"[{_yt_secs(cur_start)} → {_yt_secs(cur_end)}] {' '.join(cur_text)}",
            metadata={"source": _yt_state.video_url, "title": _yt_state.title,
                      "start_sec": cur_start, "timestamp": _yt_secs(cur_start)}))
    if not docs:
        return "❌ No chunks created."
    _yt_state.vectorstore = build_vectorstore(docs)
    return f"✅ Indexed {len(docs)} chunks."


def _yt_query(question: str) -> str:
    if _yt_state.vectorstore is None:
        return json.dumps({"answer": "❌ Load and index a video first.", "timestamps": []})
    docs = _yt_state.vectorstore.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(question)
    if not docs:
        return json.dumps({"answer": "No relevant content found.", "timestamps": []})
    context = "\n\n".join(d.page_content for d in docs)
    timestamps = [{"timestamp": d.metadata.get("timestamp", ""),
                   "yt_link": f"{_yt_state.video_url}&t={int(d.metadata.get('start_sec',0))}s",
                   "snippet": d.page_content[:100]} for d in docs]
    resp = get_llm(0.1).invoke([HumanMessage(content=(
        f'Video: "{_yt_state.title}" by {_yt_state.channel}\n\n'
        f"Transcript excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ))])
    return json.dumps({"answer": resp.content, "timestamps": timestamps,
                       "video_url": _yt_state.video_url, "title": _yt_state.title})


def _yt_summarize(style: str = "detailed") -> str:
    if not _yt_state.full_transcript:
        return "❌ Fetch transcript first."
    excerpt = _yt_state.full_transcript[:12000]
    instr = {"brief": "Write a brief 3-5 sentence summary.",
             "bullets": "Extract 8-12 key bullet points with timestamps."
             }.get(style, "Write a comprehensive summary with sections: Overview, Key Topics, Insights, Conclusion.")
    resp = get_llm(0.2).invoke([HumanMessage(content=(
        f'Video: "{_yt_state.title}" by {_yt_state.channel}\n'
        f"Duration: {_yt_state.duration}\n\nTranscript:\n{excerpt}\n\n{instr}"
    ))])
    return json.dumps({"summary": resp.content, "title": _yt_state.title,
                       "channel": _yt_state.channel, "video_url": _yt_state.video_url,
                       "thumbnail": _yt_state.thumbnail})


def _yt_info() -> str:
    if not _yt_state.video_id:
        return json.dumps({"status": "No video loaded."})
    return json.dumps({"video_id": _yt_state.video_id, "title": _yt_state.title,
                       "channel": _yt_state.channel, "duration": _yt_state.duration,
                       "video_url": _yt_state.video_url, "thumbnail": _yt_state.thumbnail,
                       "transcript_segments": len(_yt_state.transcript_chunks),
                       "indexed": _yt_state.vectorstore is not None})


_yt_tools = [
    StructuredTool.from_function(func=_yt_fetch, name="fetch_youtube_data",
        description="Fetch YouTube video metadata. Call first.", args_schema=_FetchYTInput),
    StructuredTool.from_function(func=_yt_transcript, name="fetch_transcript",
        description="Fetch timestamped transcript.", args_schema=_FetchTransInput),
    StructuredTool.from_function(func=_yt_index, name="index_transcript",
        description="Index transcript into FAISS.", args_schema=_IndexTransInput),
    StructuredTool.from_function(func=_yt_query, name="query_youtube",
        description="Answer questions with timestamp citations.", args_schema=_QueryYTInput),
    StructuredTool.from_function(func=_yt_summarize, name="summarize_video",
        description="Summarize the video.", args_schema=_SummarizeYTInput),
    StructuredTool.from_function(func=_yt_info, name="get_video_info",
        description="Return video metadata and index status.", args_schema=_VideoInfoInput),
]


def _make_yt_agent():
    return create_agent(llm=get_llm(0.1), tools=_yt_tools, verbose=True, handle_parsing_errors=True, max_iterations=10)


class VideoRAGAgent:
    name = "YouTube RAG Agent"
    def __init__(self): self._ex = None

    @property
    def executor(self):
        if self._ex is None: self._ex = _make_yt_agent()
        return self._ex

    def ingest(self, youtube_url: str, language: str = "en") -> str:
        r1 = _yt_fetch(youtube_url)
        if r1.startswith("❌"): return r1
        r2 = _yt_transcript(language)
        if r2.startswith("❌"): return r1 + "\n" + r2
        r3 = _yt_index(60)
        return "\n\n".join([r1, r2, r3])

    def query(self, question: str) -> Dict:
        raw = _yt_query(question)
        try: return json.loads(raw)
        except Exception: return {"answer": raw, "timestamps": []}

    def summarize(self, style: str = "detailed") -> Dict:
        raw = _yt_summarize(style)
        try: return json.loads(raw)
        except Exception: return {"summary": raw, "title": _yt_state.title}

    def get_info(self) -> Dict:
        raw = _yt_info()
        try: return json.loads(raw)
        except Exception: return {}

    @staticmethod
    def is_youtube_url(text: str) -> bool:
        return bool(re.search(r'(youtube\.com|youtu\.be)', text, re.IGNORECASE))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 — DATA ANALYSIS AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _DataState:
    df: Optional[pd.DataFrame] = None
    file_name: str = ""

_data_state = _DataState()


class _LoadDataInput(BaseModel):
    file_path: str = Field(description="Path to CSV/Excel/JSON file.")

class _DataSummaryInput(BaseModel): pass

class _AnalyzeInput(BaseModel):
    question: str = Field(description="Data analysis question.")

class _ChartInput(BaseModel):
    chart_type: str           = Field(description="bar|line|scatter|histogram|pie|heatmap|box")
    x_col: Optional[str]      = Field(default=None)
    y_col: Optional[str]      = Field(default=None)
    title: str                = Field(default="Chart")

class _ListColsInput(BaseModel): pass


def _data_load(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    try:
        if ext == ".csv":    _data_state.df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"): _data_state.df = pd.read_excel(file_path)
        elif ext == ".json": _data_state.df = pd.read_json(file_path)
        else: return f"❌ Unsupported: {ext}"
    except Exception as e:
        return f"❌ {e}"
    _data_state.file_name = Path(file_path).name
    return (f"✅ Loaded '{_data_state.file_name}': "
            f"{_data_state.df.shape[0]:,} rows × {_data_state.df.shape[1]} cols. "
            f"Columns: {', '.join(_data_state.df.columns.tolist())}")


def _data_summary() -> str:
    if _data_state.df is None: return "⚠️ No data loaded."
    buf = io.StringIO(); _data_state.df.info(buf=buf)
    num = _data_state.df.select_dtypes(include=np.number).columns.tolist()
    corr = _data_state.df[num].corr().round(2).to_string() if len(num) >= 2 else "N/A"
    return (f"Shape: {_data_state.df.shape}\n\n"
            f"Describe:\n{_data_state.df.describe(include='all').to_string()}\n\n"
            f"Nulls:\n{_data_state.df.isnull().sum().to_string()}\n\nCorrelation:\n{corr}")


def _data_analyze(question: str) -> str:
    if _data_state.df is None: return json.dumps({"analysis": "⚠️ No data loaded."})
    llm = get_llm(0.2)
    prompt = (
        f"You are a senior data analyst.\nColumns:\n{_data_state.df.dtypes.to_string()}\n\n"
        f"Sample:\n{_data_state.df.head(5).to_string()}\n\nStats:\n"
        f"{_data_state.df.describe(include='all').to_string()}\n\nQuestion: {question}\n\n"
        'Return ONLY JSON: {"analysis":"...","chart_type":"bar|line|scatter|histogram|pie|heatmap|box",'
        '"x_col":"col or null","y_col":"col or null","title":"chart title"}'
    )
    resp  = llm.invoke([HumanMessage(content=prompt)])
    match = re.search(r"\{.*\}", resp.content.strip(), re.DOTALL)
    return match.group() if match else json.dumps({"analysis": resp.content})


def _data_chart(chart_type: str, x_col=None, y_col=None, title: str = "Chart") -> str:
    if _data_state.df is None: return "⚠️ No data loaded."
    df = _data_state.df
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if x_col not in df.columns: x_col = cat[0] if cat else (num[0] if num else None)
    if y_col not in df.columns: y_col = num[0] if num else None
    pal = sns.color_palette("viridis", 12)
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_dark_theme(ax, fig)
    try:
        if chart_type == "bar" and x_col and y_col:
            d = df.groupby(x_col)[y_col].mean().reset_index().head(15)
            ax.bar(d[x_col].astype(str), d[y_col], color=pal)
            plt.xticks(rotation=45, ha="right", color="white")
        elif chart_type == "line" and y_col:
            s = df[y_col].dropna().head(150)
            ax.plot(range(len(s)), s.values, color="#7c6df2", linewidth=2.5)
        elif chart_type == "scatter" and x_col and y_col:
            ax.scatter(df[x_col], df[y_col], alpha=0.55, c=pal[3], s=40)
        elif chart_type == "histogram" and y_col:
            ax.hist(df[y_col].dropna(), bins=30, color=pal[4])
        elif chart_type == "pie" and x_col:
            counts = df[x_col].value_counts().head(8)
            ax.pie(counts.values, labels=counts.index.astype(str),
                   autopct="%1.1f%%", colors=pal, textprops={"color": "white"})
        elif chart_type == "heatmap" and len(num) >= 2:
            sns.heatmap(df[num].corr(), ax=ax, cmap="viridis", annot=True, fmt=".2f")
        elif chart_type == "box" and y_col:
            ax.boxplot(df[y_col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor=pal[1]))
        else:
            if num: ax.bar(df[num].mean().index, df[num].mean().values, color=pal)
        ax.set_title(title, color="white", fontsize=13)
    except Exception as e:
        ax.text(0.5, 0.5, f"Chart error:\n{e}", ha="center", va="center",
                transform=ax.transAxes, color="red")
    return fig_to_base64(fig)


def _data_list_cols() -> str:
    if _data_state.df is None: return "No data loaded."
    return "Columns:\n" + "\n".join(f"  {c:30s} {str(t)}"
                                     for c, t in _data_state.df.dtypes.items())


_data_tools = [
    StructuredTool.from_function(func=_data_load,     name="load_data",
        description="Load CSV/Excel/JSON into memory.", args_schema=_LoadDataInput),
    StructuredTool.from_function(func=_data_summary,  name="get_summary",
        description="Statistical summary of the dataset.", args_schema=_DataSummaryInput),
    StructuredTool.from_function(func=_data_analyze,  name="analyze_data",
        description="LLM-driven data analysis + chart plan.", args_schema=_AnalyzeInput),
    StructuredTool.from_function(func=_data_chart,    name="render_chart",
        description="Render a chart as base64 PNG.", args_schema=_ChartInput),
    StructuredTool.from_function(func=_data_list_cols, name="list_columns",
        description="List columns and types.", args_schema=_ListColsInput),
]


def _make_data_agent():
    return create_agent(llm=get_llm(0.1), tools=_data_tools, verbose=True, handle_parsing_errors=True, max_iterations=8)


class DataAnalysisAgent:
    name = "Data Analysis Agent"
    def __init__(self): self._ex = None

    @property
    def executor(self):
        if self._ex is None: self._ex = _make_data_agent()
        return self._ex

    @property
    def df(self): return _data_state.df
    @property
    def file_name(self): return _data_state.file_name

    def load_data(self, path): return _data_load(path)
    def get_summary(self): return _data_summary()

    def analyze(self, question):
        raw = _data_analyze(question)
        try: plan = json.loads(raw)
        except Exception: return {"answer": raw, "chart": None}
        chart = None
        if plan.get("chart_type"):
            r = _data_chart(plan.get("chart_type","bar"), plan.get("x_col"),
                            plan.get("y_col"), plan.get("title","Chart"))
            if r and len(r) > 100: chart = r
        return {"answer": plan.get("analysis", ""), "chart": chart}

    def custom_chart(self, chart_type, x_col, y_col, title=""):
        r = _data_chart(chart_type, x_col, y_col, title or f"{chart_type} Chart")
        return r if r and len(r) > 100 else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 — CODE GENERATOR AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _code_extract(text, lang=""):
    lang_pat = re.escape(lang.lower()) if lang else r"\w*"
    m = re.search(r"```(?:" + lang_pat + r")?[\n\r]?(.*?)```",
                  text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()

def _code_strip(text):
    return re.sub(r"```(?:\w+)?[\n\r]?.*?```", "", text, flags=re.DOTALL).strip()


class _GenInput(BaseModel):
    request: str  = Field(description="What the code should do.")
    language: str = Field(default="Python")
    context: str  = Field(default="")

class _ExplainInput(BaseModel):
    code: str = Field(description="Code to explain.")

class _DebugInput(BaseModel):
    code: str  = Field(description="Buggy code.")
    error: str = Field(default="")

class _RunInput(BaseModel):
    python_code: str = Field(description="Python to execute.")

class _ConvertInput(BaseModel):
    code: str        = Field(description="Source code.")
    source_lang: str = Field(description="From language.")
    target_lang: str = Field(description="To language.")


def _code_generate(request, language="Python", context="") -> str:
    body = (f"You are an expert {language} engineer. Write clean, well-commented, "
            f"production-ready code with error handling and docstrings.\n\n"
            f"{'Context: ' + context + chr(10) + chr(10) if context else ''}"
            f"Request: {request}")
    resp = get_llm(0.2).invoke([HumanMessage(content=body)])
    return json.dumps({"code": _code_extract(resp.content, language),
                       "explanation": _code_strip(resp.content), "language": language})

def _code_explain(code) -> str:
    return get_llm(0.1).invoke([HumanMessage(content=(
        "Explain this code:\n1. Overview\n2. Block walkthrough\n3. Key concepts\n"
        f"4. Issues & improvements\n\n```\n{code}\n```"))]).content

def _code_debug(code, error="") -> str:
    resp = get_llm(0.1).invoke([HumanMessage(content=(
        f"Debug and fix this code.\nError: {error}\n\nCode:\n```\n{code}\n```\n\n"
        "1. Root cause\n2. Fixed code in fenced block\n3. Changes made"))]).content
    return json.dumps({"fixed_code": _code_extract(resp), "explanation": resp})

def _code_run(python_code) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(python_code); tmp = f.name
    try:
        p = subprocess.run([sys.executable, tmp],
                           capture_output=True, text=True, timeout=10)
        return json.dumps({"stdout": p.stdout[:3000], "stderr": p.stderr[:1000],
                           "success": p.returncode == 0})
    except subprocess.TimeoutExpired:
        return json.dumps({"stdout": "", "stderr": "Timed out (10s).", "success": False})
    except Exception as e:
        return json.dumps({"stdout": "", "stderr": str(e), "success": False})

def _code_convert(code, source_lang, target_lang) -> str:
    resp = get_llm(0.1).invoke([HumanMessage(content=(
        f"Translate from {source_lang} to idiomatic {target_lang}. "
        f"Return code in a fenced block.\n\n```{source_lang.lower()}\n{code}\n```"))]).content
    return json.dumps({"converted_code": _code_extract(resp, target_lang),
                       "notes": _code_strip(resp)})


_code_tools = [
    StructuredTool.from_function(func=_code_generate, name="generate_code",
        description="Generate production-ready code.", args_schema=_GenInput),
    StructuredTool.from_function(func=_code_explain,  name="explain_code",
        description="Explain source code.", args_schema=_ExplainInput),
    StructuredTool.from_function(func=_code_debug,    name="debug_code",
        description="Debug and fix code.", args_schema=_DebugInput),
    StructuredTool.from_function(func=_code_run,      name="run_python",
        description="Execute Python code safely.", args_schema=_RunInput),
    StructuredTool.from_function(func=_code_convert,  name="convert_code",
        description="Translate code between languages.", args_schema=_ConvertInput),
]


def _make_code_agent():
    return create_agent(llm=get_llm(0.1), tools=_code_tools, verbose=True, handle_parsing_errors=True, max_iterations=8)


class CodeGeneratorAgent:
    name = "Code Generator Agent"
    def __init__(self): self._ex = None

    @property
    def executor(self):
        if self._ex is None: self._ex = _make_code_agent()
        return self._ex

    def generate(self, request, language="Python", context=""):
        raw = _code_generate(request, language, context)
        try: return json.loads(raw)
        except Exception: return {"code": raw, "explanation": "", "language": language}

    def explain(self, code): return _code_explain(code)

    def debug(self, code, error=""):
        raw = _code_debug(code, error)
        try: return json.loads(raw)
        except Exception: return {"fixed_code": "", "explanation": raw}

    def run(self, python_code):
        raw = _code_run(python_code)
        try: return json.loads(raw)
        except Exception: return {"stdout": "", "stderr": raw, "success": False}

    def convert(self, code, source_lang, target_lang):
        raw = _code_convert(code, source_lang, target_lang)
        try: return json.loads(raw)
        except Exception: return {"converted_code": raw, "notes": ""}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9 — DEEP RESEARCHER AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ResState:
    last_results: List[Dict] = []
    last_queries: List[str]  = []

_res_state = _ResState()


class _PlanInput(BaseModel):
    topic: str = Field(description="Research topic.")
    depth: str = Field(default="standard", description="quick|standard|deep")

class _SearchInput(BaseModel):
    query: str       = Field(description="Search query.")
    max_results: int = Field(default=5)

class _SynthInput(BaseModel):
    topic: str        = Field(description="Research topic.")
    results: List[Dict] = Field(default_factory=list)

class _FactsInput(BaseModel):
    topic: str        = Field(description="Topic.")
    results: List[Dict] = Field(default_factory=list)
    max_facts: int    = Field(default=10)

class _CompareInput(BaseModel):
    topic: str        = Field(description="Topic.")
    results: List[Dict] = Field(default_factory=list)


def _res_plan(topic, depth="standard") -> str:
    n = {"quick": 3, "standard": 5, "deep": 8}.get(depth, 5)
    resp = get_llm(0.4).invoke([HumanMessage(content=(
        f"Generate {n} diverse search queries for: \"{topic}\"\n"
        f"Return ONLY a JSON array of {n} query strings."))])
    m = re.search(r"\[.*?\]", resp.content.strip(), re.DOTALL)
    if m:
        try:
            q = json.loads(m.group()); _res_state.last_queries = q; return json.dumps(q)
        except Exception: pass
    fb = [topic, f"{topic} overview", f"{topic} 2024", f"{topic} analysis", f"{topic} future"][:n]
    _res_state.last_queries = fb; return json.dumps(fb)


def _res_search(query, max_results=5) -> str:
    if not DDGS_OK:
        return json.dumps([{"title": "DuckDuckGo unavailable", "url": "",
                            "snippet": "pip install duckduckgo-search"}])
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=min(max_results, 10)))
        results = [{"title": r.get("title",""), "url": r.get("href",""),
                    "snippet": r.get("body","")} for r in raw]
        _res_state.last_results.extend(results)
        return json.dumps(results)
    except Exception as e:
        return json.dumps([{"title": "Error", "url": "", "snippet": str(e)}])


def _res_synthesize(topic, results) -> str:
    data = results or _res_state.last_results
    if not data: return "⚠️ No results to synthesize."
    ctx = "\n\n".join(f"[{i+1}] {r.get('title','')}\n{r.get('snippet','')}\n{r.get('url','')}"
                      for i, r in enumerate(data[:20]))
    return get_llm(0.2).invoke([HumanMessage(content=(
        f"Write a comprehensive markdown research report on: **{topic}**\n\n"
        "Sections: # Executive Summary, # Key Findings, # Detailed Analysis, "
        "# Trends, # Challenges, # Recommendations, # Sources\n\n"
        f"Cite sources as [1],[2]. Research data:\n{ctx}"))]).content


def _res_facts(topic, results, max_facts=10) -> str:
    data = results or _res_state.last_results
    if not data: return "No results."
    snippets = "\n".join(r.get("snippet","") for r in data[:15])
    return get_llm(0.1).invoke([HumanMessage(content=(
        f"Extract {max_facts} key facts about '{topic}' as bullet points (• ).\n\n"
        f"Snippets:\n{snippets}"))]).content


def _res_compare(topic, results) -> str:
    data = (results or _res_state.last_results)[:8]
    if not data: return "No results."
    text = "\n\n".join(f"Source {i+1} ({r.get('title','')}):\n{r.get('snippet','')}"
                       for i, r in enumerate(data))
    return get_llm(0.2).invoke([HumanMessage(content=(
        f"Compare sources about '{topic}':\n\n{text}\n\n"
        "1. Table: Source|Claim|Sentiment|Key Stat\n"
        "2. Agreements\n3. Disagreements\n4. Credibility notes"))]).content


_res_tools = [
    StructuredTool.from_function(func=_res_plan,      name="plan_queries",
        description="Generate strategic search queries.", args_schema=_PlanInput),
    StructuredTool.from_function(func=_res_search,    name="web_search",
        description="DuckDuckGo web search.", args_schema=_SearchInput),
    StructuredTool.from_function(func=_res_synthesize, name="synthesize_report",
        description="Synthesize results into a structured report.", args_schema=_SynthInput),
    StructuredTool.from_function(func=_res_facts,     name="extract_facts",
        description="Extract key facts as bullet points.", args_schema=_FactsInput),
    StructuredTool.from_function(func=_res_compare,   name="compare_sources",
        description="Compare and contrast sources.", args_schema=_CompareInput),
]


def _make_research_agent():
    return create_agent(llm=get_llm(0.1), tools=_res_tools, verbose=True, handle_parsing_errors=True, max_iterations=15)


class DeepResearcherAgent:
    name = "Deep Researcher Agent"
    def __init__(self): self._ex = None

    @property
    def executor(self):
        if self._ex is None: self._ex = _make_research_agent()
        return self._ex

    def research(self, topic, depth="standard"):
        queries = json.loads(_res_plan(topic, depth))
        all_results = []
        for q in queries:
            try: all_results.extend(json.loads(_res_search(q, 5)))
            except Exception: pass
        report  = _res_synthesize(topic, all_results)
        sources = [{"title": r.get("title",""), "url": r.get("url","")}
                   for r in all_results[:12] if r.get("url")]
        return {"report": report, "queries_used": queries,
                "sources_found": len(all_results), "sources": sources}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10 — GENERAL CHATBOT AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_NEXUS_PERSONA = """You are NEXUS, the central AI assistant of a multi-agent intelligence platform.
Specialists: 📄 RAG (docs) · 🎬 YouTube RAG · 📊 Data Analyst · 💻 Code Generator · 🔬 Deep Researcher.
Be warm, concise, and intelligent. Suggest a specialist when relevant. Never fabricate facts."""


class _ChatState:
    history: List[Dict] = []
    context: Dict       = {}

_chat_state = _ChatState()


class _ChatInput(BaseModel):
    message: str      = Field(description="User message.")
    context_info: Dict = Field(default_factory=dict)

class _SumHistInput(BaseModel):
    max_turns: int = Field(default=10)

class _IntentInput(BaseModel):
    message: str       = Field(description="User message.")
    context_info: Dict = Field(default_factory=dict)

class _StatusInput(BaseModel): pass
class _HelpInput(BaseModel):
    agent_name: str = Field(default="")


def _chat_reply(message, context_info=None) -> str:
    ctx = context_info or {}; _chat_state.context = ctx
    parts = []
    if ctx.get("rag_ingested"):   parts.append("📄 Docs loaded")
    if ctx.get("video_ingested"): parts.append("🎬 Video loaded")
    if ctx.get("data_loaded"):    parts.append(f"📊 {ctx.get('data_filename','')}")
    note = (" [System: " + " | ".join(parts) + "]") if parts else ""
    msgs = [HumanMessage(content=_NEXUS_PERSONA)]
    for t in _chat_state.history[-20:]:
        msgs.append(HumanMessage(content=t["content"]) if t["role"] == "user"
                    else AIMessage(content=t["content"]))
    msgs.append(HumanMessage(content=message + note))
    reply = get_llm(0.7).invoke(msgs).content
    _chat_state.history.append({"role": "user",      "content": message})
    _chat_state.history.append({"role": "assistant", "content": reply})
    return json.dumps({"reply": reply, "turn": len(_chat_state.history) // 2})


def _chat_summarize(max_turns=10) -> str:
    if not _chat_state.history: return "No history yet."
    excerpt = "\n".join(f"{t['role'].upper()}: {t['content'][:250]}"
                        for t in _chat_state.history[-(max_turns * 2):])
    return get_llm(0.1).invoke([HumanMessage(
        content=f"Summarize in 3-5 sentences:\n\n{excerpt}")]).content


def _chat_intent(message, context_info=None) -> str:
    ctx = context_info or {}
    resp = get_llm(0.0).invoke([HumanMessage(content=(
        f'User: "{message}"\nState: {json.dumps(ctx)}\n\n'
        'Classify: "direct"|"rag"|"video"|"data"|"code"|"research"\n'
        "• rag/video/data only if their data is loaded\n"
        '• code = writing/debugging code\n• research = needs web search\n'
        'JSON: {"intent":"...","reason":"..."}'))])
    m = re.search(r"\{.*?\}", resp.content.strip(), re.DOTALL)
    return m.group() if m else '{"intent":"direct","reason":"fallback"}'


def _chat_status() -> str:
    ctx = _chat_state.context
    return "\n".join([
        f"📄 Docs  : {'✅' if ctx.get('rag_ingested') else '❌'}",
        f"🎬 Video : {'✅' if ctx.get('video_ingested') else '❌'}",
        f"📊 Data  : {'✅ ' + ctx.get('data_filename','') if ctx.get('data_loaded') else '❌'}",
        f"💬 Turns : {len(_chat_state.history) // 2}",
    ])


def _chat_help(agent_name="") -> str:
    caps = {
        "rag":      "📄 RAG Agent — PDF/DOCX/TXT semantic Q&A",
        "video":    "🎬 YouTube RAG — transcript Q&A with timestamps",
        "data":     "📊 Data Analyst — CSV/Excel analysis + charts",
        "code":     "💻 Code Generator — write/debug/explain/run/translate",
        "research": "🔬 Deep Researcher — multi-step web research reports",
    }
    k = agent_name.lower().strip()
    return caps.get(k, "All agents:\n\n" + "\n".join(caps.values()))


_chat_tools = [
    StructuredTool.from_function(func=_chat_reply,     name="chat_with_memory",
        description="Stateful conversation with history.", args_schema=_ChatInput),
    StructuredTool.from_function(func=_chat_summarize, name="summarize_history",
        description="Summarize conversation history.", args_schema=_SumHistInput),
    StructuredTool.from_function(func=_chat_intent,    name="detect_intent",
        description="Classify user query to best agent.", args_schema=_IntentInput),
    StructuredTool.from_function(func=_chat_status,    name="get_system_status",
        description="Show which agents have data loaded.", args_schema=_StatusInput),
    StructuredTool.from_function(func=_chat_help,      name="get_agent_help",
        description="Explain what agents can do.", args_schema=_HelpInput),
]


def _make_chat_agent():
    return create_agent(llm=get_llm(0.3), tools=_chat_tools, verbose=True, handle_parsing_errors=True, max_iterations=6)


class GeneralChatbotAgent:
    name = "General Chatbot"
    def __init__(self): self._ex = None

    @property
    def executor(self):
        if self._ex is None: self._ex = _make_chat_agent()
        return self._ex

    def chat(self, message, context_info=None):
        raw = _chat_reply(message, context_info or {})
        try: r = json.loads(raw); return {"answer": r.get("reply", raw), "turn": r.get("turn", 0)}
        except Exception: return {"answer": raw, "turn": 0}

    def clear_history(self): _chat_state.history.clear()

    def get_summary(self): return _chat_summarize(10)

    def smart_reply(self, message, orchestrator, context_info=None):
        ctx = context_info or {}
        try: intent = json.loads(_chat_intent(message, ctx)).get("intent", "direct")
        except Exception: intent = "direct"
        if intent == "direct":
            return self.chat(message, ctx)
        result = {"delegated": True, "intent": intent}
        try:
            if intent == "rag":
                r = orchestrator.rag.query(message)
                result.update({"answer": f"*[→ 📄 RAG]*\n\n{r['answer']}",
                               "sources": r.get("sources", [])})
            elif intent == "video":
                r = orchestrator.video_rag.query(message)
                result["answer"] = f"*[→ 🎬 YouTube RAG]*\n\n{r.get('answer','')}"
            elif intent == "data":
                r = orchestrator.data_analysis.analyze(message)
                result.update({"answer": f"*[→ 📊 Data]*\n\n{r['answer']}", "chart": r.get("chart")})
            elif intent == "code":
                r = orchestrator.code_gen.generate(message)
                result.update({"answer": f"*[→ 💻 Code]*\n\n{r.get('explanation','')}",
                               "code": r.get("code",""), "language": "python"})
            elif intent == "research":
                r = orchestrator.researcher.research(message)
                result.update({"answer": f"*[→ 🔬 Research]*\n\n{r['report']}",
                               "research_sources": r.get("sources",[]),
                               "queries": r.get("queries_used",[])})
        except Exception as e:
            fb = self.chat(message, ctx)
            result.update({"answer": f"Delegation error: {e}\n\n{fb['answer']}", "delegated": False})
        _chat_state.history.append({"role": "user",      "content": message})
        _chat_state.history.append({"role": "assistant", "content": result.get("answer","")})
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 11 — ORCHESTRATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiAgentOrchestrator:
    def __init__(self):
        self.rag           = RAGAgent()
        self.video_rag     = VideoRAGAgent()
        self.data_analysis = DataAnalysisAgent()
        self.code_gen      = CodeGeneratorAgent()
        self.researcher    = DeepResearcherAgent()
        self.chatbot       = GeneralChatbotAgent()

    def route(self, query: str) -> str:
        resp = get_llm(0.0).invoke([HumanMessage(content=(
            f'Classify: "{query}"\nOptions: rag|video|data|code|research|chat\n'
            "Reply with ONE word only."))]).content.strip().lower()
        return resp if resp in {"rag","video","data","code","research","chat"} else "chat"
