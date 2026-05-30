# 🤖 BWA 1.0 — Blog Writing Agent

> The first version. A minimal but functional implementation of the **Orchestrator → Workers → Reducer** agentic pattern using LangGraph.

---

## 📌 Overview

BWA 1.0 is the foundational version of the Blog Writing Agent. It demonstrates the core agentic loop: a single topic goes in, a complete Markdown blog post comes out.

The pipeline follows a **Fan-Out / Fan-In** architecture — the orchestrator decomposes the topic into sections, parallel workers write each section independently, and the reducer assembles them into a final document.

---

## 🏗️ Architecture

```
  [Topic]
     │
     ▼
┌────────────┐
│Orchestrator│  ──── generates a Plan (5–7 sections)
└─────┬──────┘
      │  Fan-Out (LangGraph Send API)
      ├────────────┬────────────┐
      ▼            ▼            ▼
  [Worker 1]   [Worker 2]  [Worker 3] ...   (parallel)
      │            │            │
      └────────────┴────────────┘
                   │  Fan-In
                   ▼
            ┌────────────┐
            │  Reducer   │  ──── assembles + saves blog.md
            └────────────┘
```

---

## 📁 File Structure

| File | Description |
|---|---|
| `app.py` | Entry point — set your topic here and run |
| `nodes.py` | All node functions + LangGraph graph definition |
| `custom_objects.py` | Pydantic schemas: `State`, `Plan`, `Task` |
| `requirements.txt` | Python dependencies |

---

## 🔍 How It Works

### 1. Orchestrator Node
Takes the user's topic and calls the LLM with structured output to generate a `Plan` — a blog title plus a list of `Task` objects, each with an `id`, `title`, and `brief`.

```python
plan = llm.with_structured_output(Plan).invoke([
    SystemMessage(content="Create a blog plan with 5-7 sections on the following topic."),
    HumanMessage(content=f"Topic: {state['topic']}")
])
```

### 2. Fan-Out
LangGraph's `Send` API dispatches each task to a separate `worker` node instance — all running in parallel.

```python
def fanout(state: State):
    return [Send("worker", {"task": task, "topic": state['topic'], "plan": state["plan"]})
            for task in state["plan"].tasks]
```

### 3. Worker Nodes
Each worker receives one task and generates a Markdown section for it. Workers run concurrently, significantly reducing total latency.

### 4. Reducer Node
Collects all sections, concatenates them with the blog title, and writes the final `.md` file to disk.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running with llama3.1
ollama pull llama3.1

# Run
python app.py
```

Inside `app.py`, set your topic:

```python
from nodes import run
run("Your blog topic here")
```

---

## 📦 Dependencies

```
pydantic>=2.0.0
langchain>=0.3.0
langgraph>=0.2.0
langchain-ollama
langchain-community
```

---

## ⚠️ Known Limitations

- **Prompts are minimal** — the orchestrator and worker prompts are a single line each, which can produce shallow or unstructured content.
- **No section ordering guarantee** — sections are assembled in the order workers finish (non-deterministic).
- **No web research** — the agent only uses the LLM's training knowledge.
- **Bug:** The reducer sends the blog through LLM for cleanup, but saves the *pre-cleanup* version to disk. The cleaned version is returned in state but not persisted.

> These limitations are addressed in [BWA 2.0](../BWA_2.0/) and [BWA 3.0](../BWA_3.0/).

---

## ➡️ Next Version

**[BWA 2.0 →](../BWA_2.0/)** — Richer prompts, structured section planning with goals, bullets, and word count targets. Significant quality improvement over 1.0.