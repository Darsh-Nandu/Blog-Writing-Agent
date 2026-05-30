# 🤖 BWA 2.0 — Blog Writing Agent

> The second version. Same pipeline architecture as 1.0, but with substantially richer prompting — producing blogs that are more technical, better structured, and significantly higher quality.

---

## 📌 Overview

BWA 2.0 keeps the same **Orchestrator → Workers → Reducer** pipeline as v1, but completely rewrites the prompts for both the orchestrator and worker nodes. The result is blogs that are developer-focused, actionable, and include code snippets, edge-case analysis, and production checklists.

The biggest change: the orchestrator now generates **structured section plans** with explicit goals, bullet points, and word count targets — giving each worker node much clearer instructions to write against.

---

## 🆚 What Changed from BWA 1.0

| Dimension | BWA 1.0 | BWA 2.0 |
|---|---|---|
| Orchestrator prompt | 1 line | ~25 lines, developer-focused |
| Worker prompt | 2 lines | ~20 lines with hard constraints |
| Section plan | title + brief only | title + goal + bullets + word count |
| Code examples | Occasional | Explicitly required |
| Edge cases | Absent | Required in plan |
| Common mistakes | Absent | Mandated as a dedicated section |
| Blog quality | Basic | Technical, publication-ready |

---

## 🏗️ Architecture

The graph structure is identical to BWA 1.0:

```
  [Topic]
     │
     ▼
┌────────────┐
│Orchestrator│  ──── generates a structured Plan
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
| `app.py` | Entry point |
| `nodes.py` | Node functions + graph definition (with improved prompts) |
| `custom_objects.py` | Pydantic schemas: `State`, `Plan`, `Task` |
| `BWA_2.0_test_1.md` | Sample output: *"Mastering Self-Attention"* |
| `requirements.txt` | Python dependencies |

---

## ✨ Sample Output

The file `BWA_2.0_test_1.md` contains a full blog generated on *"Mastering Self-Attention"*. Here's a snippet:

> **Self-Attention Mechanism**
>
> The self-attention mechanism is based on the scaled dot-product attention formula:
>
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
> Self-attention can struggle with very long sequences due to its quadratic complexity in sequence length. One mitigation is residual connections:
>
> ```python
> class ResidualSelfAttention(nn.Module):
>     def forward(self, x):
>         return x + self.self_attention(x)
> ```

---

## 🔍 Key Prompt Improvements

### Orchestrator Prompt
The orchestrator is now instructed to act as a *"senior technical writer and developer advocate"* and must produce plans that include:
- A mandatory `common_mistakes` section
- At least one code sketch / MWE
- Edge cases and failure modes
- Actionable, testable bullets (e.g. *"Show a minimal code snippet for X"*, not *"Explain X"*)

### Worker Prompt
Workers are now instructed to:
- Cover **all bullets in order** — no skipping, no merging
- Stay within ±15% of the target word count
- Output clean Markdown only — no extra commentary

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

Inside `app.py`:

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

- **No web research** — relies entirely on the LLM's parametric knowledge; facts may be outdated.
- **No section ordering guarantee** — sections are still assembled in worker completion order.
- **Prompts are inline** — the large prompt strings live inside `nodes.py`, making them harder to maintain or swap.

> These are addressed in [BWA 3.0](../BWA_3.0/).

---

## ➡️ Next Version

**[BWA 3.0 →](../BWA_3.0/)** — Adds a Router node (decides if web research is needed), a Research node (Tavily search), citation grounding, recency filtering, and externalized prompts.