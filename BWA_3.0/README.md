# 🤖 BWA 3.0 — Blog Writing Agent

> The most advanced version. Adds intelligent routing, live web research via Tavily, citation grounding, recency filtering, and externalized prompts — producing research-backed blogs that know what they don't know.

---

## 📌 Overview

BWA 3.0 extends the pipeline with two new nodes: a **Router** that decides whether the topic needs live web research, and a **Research** node that fetches, filters, and structures real-world evidence from the web. Workers are then grounded against that evidence — they can only cite URLs that actually exist in the fetched results.

This version also introduces a three-mode content strategy (`closed_book`, `hybrid`, `open_book`) so the pipeline behaves correctly whether you ask it to write a timeless explainer or a current-events roundup.

---

## 🏗️ Full Pipeline Architecture

```
  [Topic]
     │
     ▼
┌──────────┐   closed_book   ┌──────────────┐
│  Router  │ ─────────────▶  │ Orchestrator │
│   Node   │                 │    (Plan)    │
└──────────┘   needs_research └──────┬───────┘
     │                               │  Fan-Out
     ▼                       ┌───────┴───────────────┐
┌──────────┐                 │                       │
│ Research │── evidence ──▶ [Worker 1] ... [Worker N] (parallel)
│   Node   │   (Tavily)      │                       │
└──────────┘                 └───────┬───────────────┘
                                     │  Fan-In
                                     ▼
                              ┌────────────┐
                              │  Reducer   │ ──▶ blog.md
                              └────────────┘
```

### The Three Modes

| Mode | When | Recency Window | Research |
|---|---|---|---|
| `closed_book` | Evergreen concepts (e.g. "How does TCP work") | N/A | ❌ Skipped |
| `hybrid` | Mostly evergreen but benefits from current examples | 45 days | ✅ |
| `open_book` | Volatile topics (weekly news, latest rankings) | 7 days | ✅ |

---

## 📁 File Structure

| File | Description |
|---|---|
| `app.py` | Entry point |
| `nodes.py` | All node functions + LangGraph graph definition |
| `custom_objects.py` | Pydantic schemas: `State`, `Plan`, `Task`, `RouterDecision`, `EvidenceItem`, `EvidencePack` |
| `prompts.py` | All system prompts externalized (`ROUTER_SYSTEM`, `RESEARCH_SYSTEM`, `ORCH_SYSTEM`, `WORKER_SYSTEM`) |
| `requirements.txt` | Python dependencies |
| `State of Multimodal LLMs in 2026.md` | Sample output from an `open_book` run |

---

## ✨ Sample Output

From a live `open_book` run on *"State of Multimodal LLMs in 2026"*:

> **Market Share and Trends**
>
> OpenAI and Google Cloud AI are leading the pack, followed by Microsoft Azure. Companies like Google (Gemini) are dominating specific sectors with their multimodal capabilities — Gemini's ability to process natural language and images has made it a top choice for industries such as healthcare and finance.
> ([Source](https://www.clarifai.com/blog/llms-and-ai-trends))
>
> **Multimodal LLM Use Cases**
>
> Companies are leveraging Voice AI with LLMs to transform customer conversations. The introduction of AI-native platforms has democratized access to generative AI — making it possible for a wider range of industries to benefit from multimodal models.
> ([Source](https://www.assemblyai.com/blog/llm-use-cases))

All source links are real, fetched, and deduplicated — the worker is forbidden from inventing URLs.

---

## 🔍 What's New in 3.0

### 1. Router Node
The LLM classifies the topic and returns a `RouterDecision`:
- What mode to use (`closed_book` / `hybrid` / `open_book`)
- Whether research is needed
- A list of 3–10 targeted search queries

### 2. Research Node (Tavily)
Runs all queries through Tavily, normalizes results, and uses a second LLM call with structured output to produce a deduplicated list of `EvidenceItem` objects — each with title, URL, snippet, and `published_at`.

For `open_book` mode, a **hard recency filter** is applied: items without a parseable ISO date, or outside the recency window, are dropped entirely.

### 3. Grounded Workers
Workers receive the evidence list and must follow a strict grounding policy:
- In `open_book` mode: every event/claim must be backed by an evidence URL.
- Unsupported claims must be written as: *"Not found in provided sources."*
- Workers may **only** use URLs from the evidence list — no hallucinated links.

### 4. Structured Planning (Enhanced)
The orchestrator now produces a richer `Plan` with:
- `audience` and `tone` fields
- `blog_kind` (`explainer`, `tutorial`, `news_roundup`, `comparison`, `system_design`)
- Per-task `requires_research`, `requires_citations`, `requires_code` flags
- Ordered section IDs so the reducer assembles sections deterministically

### 5. Externalized Prompts (`prompts.py`)
All four system prompts live in `prompts.py`, making them easy to read, edit, and version independently of the graph logic.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull the local model
```bash
ollama pull llama3.1
```

### 3. Set up environment variables
Create a `.env` file in `BWA_3.0/`:

```
TAVILY_API_KEY=your_tavily_api_key_here
```

Get a free Tavily API key at [tavily.com](https://tavily.com/).

### 4. Run
```bash
python app.py
```

Inside `app.py`:

```python
from nodes import run

# Evergreen topic — router will pick closed_book
run("How does the attention mechanism work in Transformers")

# Current-events topic — router will pick open_book + research
run("State of Multimodal LLMs in 2026")
```

---

## 📦 Dependencies

```
pydantic>=2.0.0
langchain>=0.3.0
langgraph>=0.2.0
langchain-ollama
langchain-community
python-dotenv
tavily-python
```

---

## 🗂️ State Schema

```python
class State(TypedDict):
    topic: str

    # Routing
    mode: str                     # "closed_book" | "hybrid" | "open_book"
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # Recency
    as_of: str                    # ISO date, e.g. "2026-05-30"
    recency_days: int             # 7 (open_book) | 45 (hybrid) | 3650 (closed_book)

    # Output
    sections: Annotated[List[tuple[int, str]], operator.add]
    final: str
```

---

## ⚠️ Known Limitations

- **Local LLM quality ceiling** — `llama3.1` via Ollama is capable but smaller cloud models (GPT-4o, Claude Sonnet) would produce noticeably better output.
- **Tavily free tier** — limited to 1,000 searches/month.
- **No human-in-the-loop** — the plan is generated and executed automatically; there is no step to review or edit the outline before workers run.

---

## 🗺️ Possible BWA 4.0 Ideas

- [ ] Human-in-the-loop approval step after the orchestrator
- [ ] Streamlit / FastAPI frontend
- [ ] Plug-in cloud LLMs via config (OpenAI, Anthropic)
- [ ] Final quality-evaluation node (scores the blog before saving)
- [ ] Multi-format export (HTML, PDF, Notion)

---

## ⬅️ Previous Versions

- [BWA 1.0](../BWA_1.0/) — Core pipeline
- [BWA 2.0](../BWA_2.0/) — Improved prompting