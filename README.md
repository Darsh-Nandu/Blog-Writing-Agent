<div align="center">

<img src="https://img.shields.io/badge/LangGraph-Powered-6366f1?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Ollama-llama3.1-black?style=for-the-badge&logo=ollama&logoColor=white"/>
<img src="https://img.shields.io/badge/Tavily-Research-0ea5e9?style=for-the-badge&logo=searxng&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>

<br/><br/>

# 🤖 Blog Writing Agent

### An agentic pipeline that turns a single topic into a complete, research-backed blog post.

*Built with LangGraph · Runs locally on Ollama · Evolved across 3 versions*

<br/>

[**BWA 1.0**](./BWA_1.0/) &nbsp;·&nbsp; [**BWA 2.0**](./BWA_2.0/) &nbsp;·&nbsp; [**BWA 3.0**](./BWA_3.0/)

</div>

---

## What Is This?

**Blog Writing Agent (BWA)** is a multi-node AI pipeline built with [LangGraph](https://github.com/langchain-ai/langgraph) that automates blog generation end-to-end. Give it a topic — it plans, researches, writes in parallel, and assembles a publication-ready Markdown blog post.

The architecture follows the **Orchestrator → Fan-Out → Fan-In** agentic design pattern. The project is structured as **three progressively advanced versions**, each a meaningful upgrade over the last.

<br/>

<div align="center">

| | [BWA 1.0](./BWA_1.0/) | [BWA 2.0](./BWA_2.0/) | [BWA 3.0](./BWA_3.0/) |
|:---|:---:|:---:|:---:|
| Core pipeline | ✅ | ✅ | ✅ |
| Parallel section writing | ✅ | ✅ | ✅ |
| Structured section planning | ❌ | ✅ | ✅ |
| Technical prompts (goals, bullets, word counts) | ❌ | ✅ | ✅ |
| Routing (web research vs. closed-book) | ❌ | ❌ | ✅ |
| Live web research (Tavily) | ❌ | ❌ | ✅ |
| Citation grounding | ❌ | ❌ | ✅ |
| Recency filtering | ❌ | ❌ | ✅ |
| Externalized prompts | ❌ | ❌ | ✅ |
| Deterministic section ordering | ❌ | ❌ | ✅ |

</div>

---

## Architecture

<details>
<summary><strong>BWA 3.0 — Full Pipeline</strong> (click to expand)</summary>

<br/>

```
  ┌──────────────────────────────────────────────────────────┐
  │                    BWA 3.0 Pipeline                       │
  └──────────────────────────────────────────────────────────┘

  [Topic]
     │
     ▼
  ┌──────────┐   mode=closed_book    ┌──────────────────┐
  │  Router  │ ────────────────────▶ │   Orchestrator   │
  │   Node   │                       │   (Structured    │
  └──────────┘   mode=hybrid/        │      Plan)       │
       │         open_book           └────────┬─────────┘
       ▼                                      │
  ┌──────────┐                       Fan-Out  │  (LangGraph Send API)
  │ Research │ ── evidence ──────────────┬────┴────┬─────────┐
  │  (Tavily)│                           ▼         ▼         ▼
  └──────────┘                       [Worker 1] [Worker 2] [Worker N]
                                         │         │         │
                                         └────┬────┴─────────┘
                                              │  Fan-In
                                              ▼
                                        ┌──────────┐
                                        │ Reducer  │ ──▶  blog.md
                                        └──────────┘
```

</details>

<details>
<summary><strong>BWA 1.0 & 2.0 — Core Pipeline</strong> (click to expand)</summary>

<br/>

```
  [Topic]
     │
     ▼
  ┌──────────────┐
  │ Orchestrator │  generates Plan (5–7 sections)
  └──────┬───────┘
         │  Fan-Out
    ┌────┴────┬──────┐
    ▼         ▼      ▼
 [Worker] [Worker] [Worker] ...  (parallel)
    │         │      │
    └────┬────┴──────┘
         │  Fan-In
         ▼
   ┌──────────┐
   │ Reducer  │ ──▶  blog.md
   └──────────┘
```

</details>

### The Three Research Modes (BWA 3.0)

| Mode | Trigger | Recency Window | Research |
|:---|:---|:---:|:---:|
| `closed_book` | Evergreen topics — concepts, fundamentals | — | ❌ |
| `hybrid` | Mostly evergreen, benefits from fresh examples | 45 days | ✅ |
| `open_book` | Volatile — news, rankings, latest releases | 7 days | ✅ |

---

## Sample Output

<details open>
<summary><strong>BWA 3.0</strong> — <em>"State of Multimodal LLMs in 2026"</em> (open_book, research-grounded)</summary>

<br/>

> **Market Share and Trends**
>
> OpenAI and Google Cloud AI are leading the pack, followed by Microsoft Azure. Companies like Google (Gemini) are dominating specific sectors — Gemini's ability to process natural language and images has made it a top choice for industries such as healthcare and finance.
> ([Source](https://www.clarifai.com/blog/llms-and-ai-trends))
>
> **Multimodal LLM Use Cases**
>
> Companies are leveraging Voice AI with LLMs to transform customer conversations. The introduction of AI-native platforms has democratized access to generative AI, making it possible for a wider range of industries to benefit from multimodal models.
> ([Source](https://www.assemblyai.com/blog/llm-use-cases))

*All source URLs are real, fetched, and deduplicated. Workers cannot invent citations.*

</details>

<details>
<summary><strong>BWA 2.0</strong> — <em>"Mastering Self-Attention"</em> (closed_book, technical blog)</summary>

<br/>

> **The Self-Attention Mechanism**
>
> The self-attention mechanism is based on the scaled dot-product attention formula:
> `α = softmax(QK^T / √d)`
>
> ```python
> class ScaledDotProductAttention(nn.Module):
>     def forward(self, Q, K, V):
>         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_model)
>         weights = F.softmax(scores, dim=-1)
>         return torch.matmul(weights, V)
> ```
>
> **Edge Cases and Failure Modes**
>
> Self-attention can struggle with very long sequences due to its quadratic complexity. A residual connection helps mitigate gradient vanishing:
>
> ```python
> class ResidualSelfAttention(nn.Module):
>     def forward(self, x):
>         return x + self.self_attention(x)
> ```

</details>

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with `llama3.1`
- *(BWA 3.0 only)* A free [Tavily](https://tavily.com/) API key

### Setup

```bash
# 1. Clone
git clone https://github.com/Darsh-Nandu/Blog-Writing-Agent.git
cd Blog-Writing-Agent

# 2. Pick a version
cd BWA_3.0          # or BWA_1.0 / BWA_2.0

# 3. Install dependencies
pip install -r requirements.txt

# 4. (BWA 3.0 only) Add your Tavily key
echo "TAVILY_API_KEY=your_key_here" > .env

# 5. Pull the model
ollama pull llama3.1

# 6. Run
python app.py
```

Then inside `app.py`, set your topic:

```python
from nodes import run

run("How does the attention mechanism work in Transformers")
# → closed_book: evergreen technical explainer

run("State of Multimodal LLMs in 2026")
# → open_book: research-grounded with live citations
```

---

## Repository Structure

```
Blog-Writing-Agent/
│
├── BWA_1.0/               # v1 · Core Orchestrator → Workers → Reducer
│   ├── app.py
│   ├── nodes.py
│   ├── custom_objects.py
│   ├── requirements.txt
│   └── README.md
│
├── BWA_2.0/               # v2 · Richer prompts, structured planning
│   ├── app.py
│   ├── nodes.py
│   ├── custom_objects.py
│   ├── BWA_2.0_test_1.md  ← sample output
│   ├── requirements.txt
│   └── README.md
│
└── BWA_3.0/               # v3 · Router, Tavily research, citations
    ├── app.py
    ├── nodes.py
    ├── custom_objects.py
    ├── prompts.py          ← all system prompts externalized
    ├── requirements.txt
    └── README.md
```

---

## Tech Stack

<div align="center">

| | Tool | Purpose |
|:---:|:---|:---|
| 🔗 | [LangGraph](https://github.com/langchain-ai/langgraph) | Agent graph orchestration & parallel execution |
| 🦜 | [LangChain](https://github.com/langchain-ai/langchain) | LLM interface & structured output |
| 🦙 | [Ollama](https://ollama.com/) · `llama3.1` | Local LLM inference — no API key needed |
| 🔍 | [Tavily](https://tavily.com/) | Live web research *(BWA 3.0)* |
| 🛡️ | [Pydantic v2](https://docs.pydantic.dev/) | Typed schemas for plans, tasks & evidence |

</div>

---

## Roadmap

```
✅ BWA 1.0  ── Core fan-out/fan-in pipeline
✅ BWA 2.0  ── Structured prompts & planning
✅ BWA 3.0  ── Web research, routing, citations
⬜ BWA 4.0  ── Human-in-the-loop plan approval
⬜ BWA 4.0  ── Streamlit / FastAPI frontend
⬜ BWA 4.0  ── Cloud LLM support (OpenAI / Anthropic)
⬜ BWA 4.0  ── Output quality evaluation node
⬜ BWA 4.0  ── Multi-format export (HTML, PDF, Notion)
```

---

<div align="center">

MIT License &nbsp;·&nbsp; Built by [Darsh Nandu](https://github.com/Darsh-Nandu)

</div>