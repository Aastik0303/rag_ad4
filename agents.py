"""
agents.py — NexusRAG Agent Stack
No LangChain agents/chains — just direct LLM + tool calls.
Works on any LangChain version.
"""

from __future__ import annotations

import io, base64, json, random, re, subprocess, sys, tempfile, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
    LOADERS_OK = True
except ImportError:
    LOADERS_OK = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    TRANSCRIPT_OK = True
except ImportError:
    TRANSCRIPT_OK = False

try:
    import yt_dlp as _yt_dlp; YTDLP_OK = True
except ImportError:
    YTDLP_OK = False

try:
    import requests as _req; REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from duckduckgo_search import DDGS; DDGS_OK = True
except ImportError:
    DDGS_OK = False


# ── API KEY POOL ───────────────────────────────────────────────────────────────

@dataclass
class _Slot:
    key: str
    tokens: int = 0
    errors: int = 0
    last_err: float = 0.0
    dead: bool = False

    @property
    def ok(self):
        if self.dead: return False
        if self.errors >= 3 and (time.time() - self.last_err) < 60: return False
        return True

    def fail(self):
        self.errors += 1; self.last_err = time.time()

    def reset(self):
        self.tokens = self.errors = 0; self.dead = False; self.last_err = 0.0


class ApiKeyPool:
    def __init__(self):
        self._slots: List[_Slot] = []
        self._idx = 0
        self._lock = threading.Lock()

    def set_keys(self, keys: List[str]):
        with self._lock:
            self._slots = [_Slot(k.strip()) for k in keys if k.strip()]
            if not self._slots: raise ValueError("Need at least one API key.")
            self._idx = random.randrange(len(self._slots))

    def current_key(self) -> str:
        with self._lock:
            if not self._slots: raise RuntimeError("Call set_keys() first.")
            if not self._slots[self._idx].ok: self._rotate()
            return self._slots[self._idx].key

    def _rotate(self):
        avail = [i for i, s in enumerate(self._slots) if s.ok and i != self._idx]
        if avail:
            self._idx = random.choice(avail)
        else:
            for s in self._slots: s.reset()
            self._idx = random.randrange(len(self._slots))

    def report_error(self):
        with self._lock:
            if self._slots:
                self._slots[self._idx].fail()
                if not self._slots[self._idx].ok: self._rotate()

    def status(self) -> List[dict]:
        with self._lock:
            return [{"index": i, "active": i == self._idx,
                     "tokens_used": s.tokens, "token_limit": 900_000,
                     "pct_used": round(s.tokens/900_000*100, 1),
                     "exhausted": s.dead, "errors": s.errors,
                     "available": s.ok,
                     "key_preview": s.key[:8]+"..."} for i, s in enumerate(self._slots)]

    def key_count(self): return len(self._slots)


key_pool = ApiKeyPool()

DEFAULT_MODEL = "gemini-2.5-flash"


def get_llm(temperature=0.1):
    return ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=key_pool.current_key(),
        temperature=temperature,
        convert_system_message_to_human=True,
    )


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def llm_call(messages, temperature=0.1) -> str:
    try:
        return get_llm(temperature).invoke(messages).content
    except Exception as e:
        if any(x in str(e).lower() for x in ("quota", "429", "rate", "exhausted", "invalid", "expired")):
            key_pool.report_error()
            try:
                return get_llm(temperature).invoke(messages).content
            except Exception as e2:
                return f"Error: {e2}"
        return f"Error: {e}"


# ── VECTOR STORE HELPERS ───────────────────────────────────────────────────────

def build_vectorstore(docs: List[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())


def load_documents(paths: List[str]) -> List[Document]:
    docs = []
    for p in paths:
        ext = Path(p).suffix.lower()
        try:
            if ext == ".pdf" and LOADERS_OK:
                docs.extend(PyPDFLoader(p).load())
            elif ext in (".txt", ".md") and LOADERS_OK:
                docs.extend(TextLoader(p, encoding="utf-8").load())
            elif ext == ".csv" and LOADERS_OK:
                docs.extend(CSVLoader(p).load())
            else:
                text = Path(p).read_text(encoding="utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": p}))
        except Exception as e:
            docs.append(Document(page_content=f"Error loading {p}: {e}", metadata={"source": p}))
    return docs


# ── CHART HELPERS ──────────────────────────────────────────────────────────────

def _dark(ax, fig):
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig); return b64


# ── RAG AGENT ─────────────────────────────────────────────────────────────────

class RAGAgent:
    name = "RAG Agent"

    def __init__(self):
        self._vs = None
        self._sources: List[str] = []

    def ingest(self, file_paths: List[str]) -> str:
        docs = load_documents(file_paths)
        if not docs: return "No documents loaded."
        self._vs = build_vectorstore(docs)
        self._sources = list({d.metadata.get("source", "?") for d in docs})
        return f"Ingested {len(docs)} chunks from {len(file_paths)} file(s)."

    def query(self, question: str) -> Dict:
        if self._vs is None:
            return {"answer": "No documents loaded yet. Please upload files first.", "sources": []}
        docs = self._vs.as_retriever(search_kwargs={"k": 5}).invoke(question)
        if not docs:
            return {"answer": "No relevant content found.", "sources": []}
        context = "\n\n".join(d.page_content for d in docs)
        sources = list({d.metadata.get("source", "?") for d in docs})
        answer = llm_call([HumanMessage(content=(
            "Answer based ONLY on the context below. If not in context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ))])
        return {"answer": answer, "sources": sources}


# ── YOUTUBE RAG AGENT ─────────────────────────────────────────────────────────

def _yt_id(url: str) -> str:
    url = url.strip()
    if re.match(r'^[A-Za-z0-9_\-]{11}$', url): return url
    m = re.search(r'(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_\-]{11})', url)
    if m: return m.group(1)
    qs = parse_qs(urlparse(url).query)
    if 'v' in qs: return qs['v'][0]
    raise ValueError(f"Cannot extract video ID from: {url}")


def _secs(s: float) -> str:
    s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class VideoRAGAgent:
    """
    YouTube RAG Agent.
    Transcript fetch tries all available methods silently.
    If no transcript exists, falls back to AI-generated topic segments — never errors out.
    """
    name = "YouTube RAG Agent"

    def __init__(self):
        self._vs          = None
        self._chunks: list = []
        self._transcript   = ""
        self.video_id      = ""
        self.video_url     = ""
        self.title         = "Unknown"
        self.channel       = "Unknown"
        self.duration      = "Unknown"
        self.thumbnail     = ""
        self.source_type   = "unknown"   # "transcript" | "ai_generated"

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def extract_video_id(url: str):
        patterns = [
            r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
            r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
            r"(?:embed/)([0-9A-Za-z_-]{11})",
        ]
        for p in patterns:
            m = re.search(p, url)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _secs_to_ts(seconds: float) -> str:
        s = int(seconds)
        h, r = divmod(s, 3600)
        m, sec = divmod(r, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    @staticmethod
    def _parse_seg(seg) -> tuple:
        """Handle dict (old api) and object (new api) transcript segments."""
        if isinstance(seg, dict):
            return seg.get("start", 0), seg.get("duration", 0), seg.get("text", "").strip()
        return getattr(seg, "start", 0), getattr(seg, "duration", 0), getattr(seg, "text", "").strip()

    def _get_metadata(self, video_id: str):
        """Fetch title/channel via yt-dlp or oEmbed fallback."""
        if YTDLP_OK:
            try:
                opts = {"quiet": True, "no_warnings": True, "skip_download": True}
                with _yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(
                        "https://www.youtube.com/watch?v=" + video_id, download=False
                    )
                self.title    = info.get("title",    "Unknown")
                self.channel  = info.get("uploader", "Unknown")
                dur           = info.get("duration", 0)
                self.duration = self._secs_to_ts(dur) if dur else "Unknown"
                return
            except Exception:
                pass
        # oEmbed fallback (no auth needed)
        if REQUESTS_OK:
            try:
                r = _req.get(
                    "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v="
                    + video_id + "&format=json",
                    timeout=8
                )
                if r.status_code == 200:
                    d = r.json()
                    self.title   = d.get("title",       "Unknown")
                    self.channel = d.get("author_name", "Unknown")
            except Exception:
                pass

    def _fetch_transcript(self, video_id: str, language: str = "en") -> list:
        """
        Try every available method to get a transcript — silently.
        Returns list of {start, duration, text} dicts, or [] if nothing works.
        """
        if not TRANSCRIPT_OK:
            return []

        def _parse_all(result) -> list:
            out = []
            for seg in result:
                s, d, txt = self._parse_seg(seg)
                if str(txt).strip():
                    out.append({"start": float(s), "duration": float(d), "text": str(txt).strip()})
            return out

        # Attempt 1: direct get_transcript with several language variants
        for lang_list in [[language], ["en"], ["en-US"], ["en-GB"], [language, "en"]]:
            try:
                raw = _parse_all(YouTubeTranscriptApi.get_transcript(video_id, languages=lang_list))
                if raw:
                    return raw
            except Exception:
                continue

        # Attempt 2: instance-based API (youtube-transcript-api >= 1.0)
        try:
            api = YouTubeTranscriptApi()
            for lang_list in [[language, "en"], ["en"], []]:
                try:
                    result = api.fetch(video_id, languages=lang_list) if lang_list else api.fetch(video_id)
                    raw = _parse_all(result)
                    if raw:
                        return raw
                except Exception:
                    continue
        except Exception:
            pass

        # Attempt 3: list all transcripts, sorted by preference, try each
        try:
            tlist = YouTubeTranscriptApi.list_transcripts(video_id)
            all_transcripts = list(tlist)

            # Manual transcripts first, then generated; requested language prioritised
            def _priority(t):
                lang_match = t.language_code.startswith(language) or t.language_code.startswith("en")
                return (0 if not t.is_generated else 1, 0 if lang_match else 1)

            all_transcripts.sort(key=_priority)

            for t_obj in all_transcripts:
                try:
                    raw = _parse_all(t_obj.fetch())
                    if raw:
                        return raw
                except Exception:
                    continue

            # Last resort: translate first available to English
            if all_transcripts:
                try:
                    raw = _parse_all(all_transcripts[0].translate("en").fetch())
                    if raw:
                        return raw
                except Exception:
                    pass
        except Exception:
            pass

        return []

    def _ai_fallback_chunks(self) -> List[Document]:
        """
        When no transcript exists, ask the LLM to generate topic segments
        from the video title/channel so the agent is still useful.
        """
        prompt = (
            f"You are analysing a YouTube video.\n"
            f"Title: {self.title}\n"
            f"Channel: {self.channel}\n\n"
            "Generate 12 detailed topic segments that likely appear in this video.\n"
            "Format EACH segment on its own line exactly as:\n"
            "[Topic N] <one or two sentence description of what this segment covers>\n\n"
            "Generate 12 segments now:"
        )
        response = llm_call([HumanMessage(content=prompt)], temperature=0.3)
        lines = [l.strip() for l in response.splitlines() if l.strip().startswith("[")]
        if not lines:
            lines = [f"[Overview] This video titled '{self.title}' by {self.channel} covers various topics."]
        docs = []
        for i, line in enumerate(lines):
            docs.append(Document(
                page_content=line,
                metadata={"start_sec": i * 60, "timestamp": self._secs_to_ts(i * 60),
                          "source": self.video_url}
            ))
        return docs

    def _chunks_from_transcript(self, raw: list, chunk_secs: int = 60) -> List[Document]:
        """Group transcript entries into ~chunk_secs windows."""
        docs = []
        cur_text, cur_start, cur_end = [], None, 0.0
        for seg in raw:
            s, d, txt = seg["start"], seg["duration"], seg["text"]
            if cur_start is None:
                cur_start = s
            cur_text.append(txt)
            cur_end = s + d
            if (cur_end - cur_start) >= chunk_secs:
                ts = self._secs_to_ts(cur_start)
                docs.append(Document(
                    page_content="[" + ts + "] " + " ".join(cur_text),
                    metadata={"start_sec": cur_start, "timestamp": ts, "source": self.video_url}
                ))
                cur_text, cur_start = [], None
        if cur_text and cur_start is not None:
            ts = self._secs_to_ts(cur_start)
            docs.append(Document(
                page_content="[" + ts + "] " + " ".join(cur_text),
                metadata={"start_sec": cur_start, "timestamp": ts, "source": self.video_url}
            ))
        return docs

    # ── public API ────────────────────────────────────────────────────────────

    def ingest(self, youtube_url: str, language: str = "en") -> str:
        # Reset state
        self._vs = None; self._chunks = []; self._transcript = ""
        self.title = self.channel = self.duration = "Unknown"
        self.source_type = "unknown"

        # Extract video ID
        vid = self.extract_video_id(youtube_url)
        if not vid:
            return "Error: Could not extract video ID from URL."

        self.video_id  = vid
        self.video_url = "https://www.youtube.com/watch?v=" + vid
        self.thumbnail = "https://img.youtube.com/vi/" + vid + "/hqdefault.jpg"

        # Metadata (always works — has oEmbed fallback)
        self._get_metadata(vid)

        # Transcript — try all methods silently
        raw = self._fetch_transcript(vid, language)

        if raw:
            # Real transcript path
            self._chunks = raw
            self._transcript = "\n".join(
                "[" + self._secs_to_ts(s["start"]) + "] " + s["text"] for s in raw
            )
            docs = self._chunks_from_transcript(raw)
            self.source_type = "transcript"
            source_note = f"{len(raw)} transcript segments"
        else:
            # AI-generated fallback — no error, just different source
            docs = self._ai_fallback_chunks()
            self._transcript = "\n".join(d.page_content for d in docs)
            self.source_type = "ai_generated"
            source_note = f"{len(docs)} AI-generated topic segments (transcript unavailable)"

        if not docs:
            return "Error: Could not build any content chunks for this video."

        try:
            self._vs = build_vectorstore(docs)
        except Exception as e:
            return "Error building index: " + str(e)

        return (
            "Loaded: " + self.title
            + " | " + source_note
            + " | " + str(len(docs)) + " chunks indexed."
        )

    def is_ready(self) -> bool:
        return self._vs is not None

    def query(self, question: str) -> Dict:
        if not self.is_ready():
            return {
                "answer": "Video not loaded. Please paste a YouTube URL and click Load.",
                "timestamps": []
            }
        docs = self._vs.as_retriever(search_kwargs={"k": 5}).invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        timestamps = [
            {
                "timestamp": d.metadata.get("timestamp", ""),
                "yt_link":  self.video_url + "&t=" + str(int(d.metadata.get("start_sec", 0))) + "s"
            }
            for d in docs
        ]
        source_note = (
            "The context is from the real video transcript with timestamps."
            if self.source_type == "transcript"
            else "Note: No transcript was available — context is AI-generated from the video title/channel."
        )
        answer = llm_call([HumanMessage(content=(
            "You are a helpful YouTube video assistant.\n"
            + source_note + "\n"
            "Answer using ONLY the context below. Cite timestamps like [MM:SS] when relevant.\n"
            "If the answer is not in the context, say so.\n\n"
            "Video: \"" + self.title + "\" by " + self.channel + "\n\n"
            "Relevant segments:\n" + context + "\n\n"
            "Question: " + question + "\n\nAnswer:"
        ))])
        return {"answer": answer, "timestamps": timestamps, "video_url": self.video_url,
                "source_type": self.source_type}

    def summarize(self, style: str = "detailed") -> Dict:
        if not self._transcript:
            return {"summary": "No video loaded.", "title": ""}
        excerpt = self._transcript[:12000]
        instr = {
            "brief":   "Write a brief 3-5 sentence summary.",
            "bullets": "List the 10 most important points as bullet points with timestamps.",
        }.get(style, "Write a structured summary covering: Overview, Key Topics, Main Insights, Conclusion.")
        summary = llm_call([HumanMessage(content=(
            "Video: \"" + self.title + "\" by " + self.channel + "\n\n"
            "Content:\n" + excerpt + "\n\n" + instr
        ))], temperature=0.2)
        return {"summary": summary, "title": self.title, "channel": self.channel,
                "video_url": self.video_url, "thumbnail": self.thumbnail,
                "source_type": self.source_type}

    def get_info(self) -> Dict:
        return {
            "video_id":            self.video_id,
            "title":               self.title,
            "channel":             self.channel,
            "duration":            self.duration,
            "video_url":           self.video_url,
            "thumbnail":           self.thumbnail,
            "transcript_segments": len(self._chunks),
            "indexed":             self._vs is not None,
            "source_type":         self.source_type,
        }

    @staticmethod
    def is_youtube_url(text: str) -> bool:
        return bool(re.search(r"(youtube\.com|youtu\.be)", text, re.IGNORECASE))


# ── DATA ANALYSIS AGENT ───────────────────────────────────────────────────────

class DataAnalysisAgent:
    name = "Data Analysis Agent"

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_name = ""

    def load_data(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".csv":    self.df = pd.read_csv(path)
            elif ext in (".xlsx", ".xls"): self.df = pd.read_excel(path)
            elif ext == ".json": self.df = pd.read_json(path)
            else: return f"Unsupported format: {ext}"
        except Exception as e:
            return f"Error: {e}"
        self.file_name = Path(path).name
        return (f"Loaded '{self.file_name}': "
                f"{self.df.shape[0]:,} rows x {self.df.shape[1]} columns. "
                f"Columns: {', '.join(self.df.columns.tolist())}")

    def get_summary(self) -> str:
        if self.df is None: return "No data loaded."
        buf = io.StringIO(); self.df.info(buf=buf)
        return (f"Shape: {self.df.shape}\n\n"
                f"Stats:\n{self.df.describe(include='all').to_string()}\n\n"
                f"Nulls:\n{self.df.isnull().sum().to_string()}")

    def analyze(self, question: str) -> Dict:
        if self.df is None:
            return {"answer": "No data loaded. Please upload a file first.", "chart": None}
        sample = self.df.head(5).to_string()
        dtypes = self.df.dtypes.to_string()
        resp = llm_call([HumanMessage(content=(
            f"You are a data analyst. Answer this question:\n{question}\n\n"
            f"Dataset columns:\n{dtypes}\n\nSample rows:\n{sample}\n\n"
            "Reply in JSON only:\n"
            '{"answer":"analysis text","chart_type":"bar|line|scatter|histogram|pie|heatmap|box",'
            '"x_col":"column or null","y_col":"numeric column or null","title":"chart title"}'
        ))], temperature=0.2)
        try:
            m = re.search(r'\{.*\}', resp, re.DOTALL)
            plan = json.loads(m.group()) if m else {"answer": resp}
        except Exception:
            return {"answer": resp, "chart": None}

        chart = None
        if plan.get("chart_type"):
            chart = self.custom_chart(plan.get("chart_type","bar"),
                                      plan.get("x_col"), plan.get("y_col"),
                                      plan.get("title","Chart"))
        return {"answer": plan.get("answer", resp), "chart": chart}

    def custom_chart(self, chart_type: str, x_col=None, y_col=None, title="Chart") -> Optional[str]:
        if self.df is None: return None
        df = self.df
        num = df.select_dtypes(include=np.number).columns.tolist()
        cat = df.select_dtypes(include=["object","category"]).columns.tolist()
        if x_col not in df.columns: x_col = cat[0] if cat else (num[0] if num else None)
        if y_col not in df.columns: y_col = num[0] if num else None

        pal = sns.color_palette("viridis", 12)
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        _dark(ax, fig)
        try:
            if chart_type == "bar" and x_col and y_col:
                d = df.groupby(x_col)[y_col].mean().reset_index().head(15)
                ax.bar(d[x_col].astype(str), d[y_col], color=pal)
                plt.xticks(rotation=45, ha="right", color="white")
            elif chart_type == "line" and y_col:
                s = df[y_col].dropna().head(150)
                ax.plot(range(len(s)), s.values, color="#7c6df2", linewidth=2)
            elif chart_type == "scatter" and x_col and y_col:
                ax.scatter(df[x_col], df[y_col], alpha=0.6, color=pal[3], s=40)
            elif chart_type == "histogram" and y_col:
                ax.hist(df[y_col].dropna(), bins=30, color=pal[4])
            elif chart_type == "pie" and x_col:
                c = df[x_col].value_counts().head(8)
                ax.pie(c.values, labels=c.index.astype(str), autopct="%1.1f%%",
                       colors=pal, textprops={"color":"white"})
            elif chart_type == "heatmap" and len(num) >= 2:
                sns.heatmap(df[num].corr(), ax=ax, cmap="viridis", annot=True, fmt=".2f")
            elif chart_type == "box" and y_col:
                ax.boxplot(df[y_col].dropna(), patch_artist=True,
                           boxprops=dict(facecolor=pal[1]))
            else:
                if num: ax.bar(df[num].mean().index, df[num].mean().values, color=pal)
            ax.set_title(title, color="white", fontsize=12)
        except Exception as e:
            ax.text(0.5, 0.5, f"Chart error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        return _b64(fig)


# ── CODE GENERATOR AGENT ──────────────────────────────────────────────────────

def _extract_code(text: str) -> str:
    m = re.search(r"```[\w]*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


class CodeGeneratorAgent:
    name = "Code Generator"

    def generate(self, request: str, language: str = "Python") -> Dict:
        resp = llm_call([HumanMessage(content=(
            f"Write simple, clean {language} code for: {request}\n\n"
            "Keep it short and easy to understand. "
            "Add brief comments. No unnecessary complexity. "
            "Put code in a code block, then give a short explanation."
        ))], temperature=0.2)
        code = _extract_code(resp)
        explanation = re.sub(r"```[\w]*\n?.*?```", "", resp, flags=re.DOTALL).strip()
        return {"code": code, "explanation": explanation, "language": language}

    def explain(self, code: str) -> str:
        return llm_call([HumanMessage(content=(
            f"Explain this code simply:\n\n```\n{code}\n```\n\n"
            "What it does and how it works."
        ))])

    def debug(self, code: str, error: str = "") -> Dict:
        resp = llm_call([HumanMessage(content=(
            f"Fix this code.\nError: {error or 'unknown'}\n\n"
            f"```\n{code}\n```\n\n"
            "Explain the bug and show the fixed code."
        ))])
        return {"fixed_code": _extract_code(resp), "explanation": resp}

    def run(self, python_code: str) -> Dict:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_code); tmp = f.name
        try:
            p = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=10)
            return {"stdout": p.stdout[:2000], "stderr": p.stderr[:500], "success": p.returncode == 0}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Timed out.", "success": False}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "success": False}


# ── DEEP RESEARCHER AGENT ─────────────────────────────────────────────────────

class DeepResearcherAgent:
    name = "Deep Researcher"

    def _search(self, query: str, n: int = 5) -> List[Dict]:
        if not DDGS_OK: return []
        try:
            with DDGS() as ddgs:
                return [{"title": r.get("title",""), "url": r.get("href",""),
                         "snippet": r.get("body","")}
                        for r in ddgs.text(query, max_results=n)]
        except Exception:
            return []

    def research(self, topic: str, depth: str = "standard") -> Dict:
        n = {"quick": 3, "standard": 5, "deep": 8}.get(depth, 5)

        q_resp = llm_call([HumanMessage(content=(
            f"Generate {n} search queries for: '{topic}'\nReturn ONLY a JSON array of {n} strings."
        ))], temperature=0.3)
        try:
            m = re.search(r'\[.*?\]', q_resp, re.DOTALL)
            queries = json.loads(m.group()) if m else [topic]
        except Exception:
            queries = [topic]

        all_results = []
        for q in queries[:n]:
            all_results.extend(self._search(q, 4))

        if not all_results:
            report = llm_call([HumanMessage(content=(
                f"Write a comprehensive research report on: {topic}\n\n"
                "Include: Overview, Key Facts, Analysis, Conclusion."
            ))], temperature=0.2)
            return {"report": report, "queries_used": queries, "sources_found": 0, "sources": []}

        context = "\n\n".join(
            f"[{i+1}] {r['title']}\n{r['snippet']}\n{r['url']}"
            for i, r in enumerate(all_results[:15])
        )
        report = llm_call([HumanMessage(content=(
            f"Write a comprehensive markdown report on: **{topic}**\n\n"
            "Sections: ## Overview, ## Key Findings, ## Analysis, ## Conclusion, ## Sources\n\n"
            f"Research data:\n{context}"
        ))], temperature=0.2)

        sources = [{"title": r["title"], "url": r["url"]} for r in all_results if r.get("url")][:12]
        return {"report": report, "queries_used": queries,
                "sources_found": len(all_results), "sources": sources}


# ── GENERAL CHATBOT ───────────────────────────────────────────────────────────

_PERSONA = ("You are NEXUS, a helpful AI assistant. Be concise and friendly. "
            "You have specialists available: documents, YouTube, data analysis, code, research.")


class GeneralChatbotAgent:
    name = "General Chatbot"

    def __init__(self):
        self._history: List = []

    def chat(self, message: str, context_info: Dict = None) -> Dict:
        ctx = context_info or {}
        msgs = [HumanMessage(content=_PERSONA)]
        for t in self._history[-20:]:
            msgs.append(HumanMessage(content=t["content"]) if t["role"] == "user"
                        else AIMessage(content=t["content"]))
        msgs.append(HumanMessage(content=message))
        reply = llm_call(msgs, temperature=0.7)
        self._history.append({"role": "user",      "content": message})
        self._history.append({"role": "assistant", "content": reply})
        return {"answer": reply}

    def detect_intent(self, message: str, context_info: Dict = None) -> str:
        ctx = context_info or {}
        resp = llm_call([HumanMessage(content=(
            f'Message: "{message}"\nLoaded: {json.dumps(ctx)}\n\n'
            'Best agent: "direct"|"rag"|"video"|"data"|"code"|"research"\n'
            'rag/video/data only if loaded. Reply ONE word.'
        ))], temperature=0.0)
        intent = resp.strip().lower().split()[0] if resp.strip() else "direct"
        return intent if intent in {"direct","rag","video","data","code","research"} else "direct"

    def smart_reply(self, message: str, orchestrator: Any, context_info: Dict = None) -> Dict:
        ctx = context_info or {}
        intent = self.detect_intent(message, ctx)
        if intent == "direct":
            return self.chat(message, ctx)
        result = {"delegated": True, "intent": intent}
        try:
            if intent == "rag":
                r = orchestrator.rag.query(message)
                result.update({"answer": r["answer"], "sources": r.get("sources",[])})
            elif intent == "video":
                r = orchestrator.video_rag.query(message)
                result["answer"] = r.get("answer","")
            elif intent == "data":
                r = orchestrator.data_analysis.analyze(message)
                result.update({"answer": r["answer"], "chart": r.get("chart")})
            elif intent == "code":
                r = orchestrator.code_gen.generate(message)
                result.update({"answer": r.get("explanation",""), "code": r.get("code",""), "language": "python"})
            elif intent == "research":
                r = orchestrator.researcher.research(message)
                result.update({"answer": r["report"], "research_sources": r.get("sources",[]),
                               "queries": r.get("queries_used",[])})
        except Exception:
            return self.chat(message, ctx)
        self._history.append({"role": "user",      "content": message})
        self._history.append({"role": "assistant", "content": result.get("answer","")})
        return result

    def clear_history(self): self._history.clear()

    def get_summary(self) -> str:
        if not self._history: return "No conversation yet."
        excerpt = "\n".join(f"{t['role'].upper()}: {t['content'][:200]}"
                            for t in self._history[-20:])
        return llm_call([HumanMessage(content=f"Summarize in 3-5 sentences:\n\n{excerpt}")])


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    def __init__(self):
        self.rag           = RAGAgent()
        self.video_rag     = VideoRAGAgent()
        self.data_analysis = DataAnalysisAgent()
        self.code_gen      = CodeGeneratorAgent()
        self.researcher    = DeepResearcherAgent()
        self.chatbot       = GeneralChatbotAgent()

    def route(self, query: str) -> str:
        resp = llm_call([HumanMessage(content=(
            f'Classify: "{query}"\nOptions: rag|video|data|code|research|chat\nOne word only.'
        ))], temperature=0.0).strip().lower().split()[0]
        return resp if resp in {"rag","video","data","code","research","chat"} else "chat"
