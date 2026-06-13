"""
nodes.py  –  BWA 4.0 pipeline nodes
Each node is a pure function (state → dict) so it can be used both
inside LangGraph and called directly from the Streamlit UI.
"""

from __future__ import annotations
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List, Callable

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from custom_objects import (
    State, Plan, Task, RouterDecision,
    EvidencePack, EvidenceItem,
)
from prompts import (
    ROUTER_SYSTEM, RESEARCH_SYSTEM,
    ORCH_SYSTEM, WORKER_SYSTEM,
)
from llm_factory import get_llm, get_structured_llm


# helpers

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query}) or []
    normalised: list[dict] = []
    for r in results:
        normalised.append({
            "title":        r.get("title") or "",
            "url":          r.get("url") or "",
            "snippet":      r.get("content") or r.get("snippet") or "",
            "published_at": r.get("published_date") or r.get("published_at"),
            "source":       r.get("source"),
        })
    return normalised


# node: router

def router_node(state: State, log: Optional[Callable] = None) -> dict:
    if log:
        log("🧭 Router: analysing topic and choosing research strategy…")

    decider = get_structured_llm(
        state["provider"], state["model_name"], RouterDecision, temperature=0.1
    )
    decision: RouterDecision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ])

    recency_days = {"open_book": 7, "hybrid": 45}.get(decision.mode, 3650)

    if log:
        emoji = {"open_book": "🌐", "hybrid": "🔀", "closed_book": "📚"}[decision.mode]
        log(f"{emoji} Mode: **{decision.mode}** — {decision.reason}")
        if decision.queries:
            log(f"🔍 Search queries: {', '.join(f'`{q}`' for q in decision.queries)}")

    return {
        "needs_research": decision.needs_research,
        "mode":           decision.mode,
        "queries":        decision.queries,
        "recency_days":   recency_days,
        # reset approval flags for this run
        "router_approved": False,
        "plan_approved":   False,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# node: research

def research_node(state: State, log: Optional[Callable] = None) -> dict:
    queries = state.get("queries") or []
    max_r   = 6

    if log:
        log(f"🔎 Researching {len(queries)} quer{'y' if len(queries)==1 else 'ies'} via Tavily…")

    raw: list[dict] = []
    for q in queries:
        if log:
            log(f"   Searching: `{q}`")
        raw.extend(_tavily_search(q, max_r))

    if not raw:
        if log:
            log("⚠️  No results found. Proceeding without evidence.")
        return {"evidence": []}

    extractor = get_structured_llm(
        state["provider"], state["model_name"], EvidencePack, temperature=0.0
    )
    pack: EvidencePack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"As-of date: {state['as_of']}\n"
            f"Recency days: {state['recency_days']}\n\n"
            f"Raw results:\n{raw}"
        )),
    ])

    # deduplicate by URL
    dedup = {e.url: e for e in pack.evidence if e.url}
    evidence = list(dedup.values())

    # Hard recency filter for open_book
    if state.get("mode") == "open_book":
        as_of   = date.fromisoformat(state["as_of"])
        cutoff  = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [
            e for e in evidence
            if (d := _iso_to_date(e.published_at)) and d >= cutoff
        ]

    if log:
        log(f"✅ Research complete: **{len(evidence)}** evidence items retained.")

    return {"evidence": evidence}


# node: orchestrator

def orchestrator_node(state: State, log: Optional[Callable] = None) -> dict:
    if log:
        log("🎼 Orchestrator: generating blog outline…")

    mode        = state.get("mode", "closed_book")
    evidence    = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    planner = get_structured_llm(
        state["provider"], state["model_name"], Plan, temperature=0.4
    )
    plan: Plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n"
            f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
            f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
            f"Evidence (use ONLY for up-to-date facts; may be empty):\n"
            f"{[e.model_dump() for e in evidence][:16]}\n\n"
            f"Instruction: If mode=open_book, do NOT drift into a tutorial."
        )),
    ])

    if forced_kind:
        plan.blog_kind = "news_roundup"

    if log:
        log(f"📋 Plan ready: **{plan.blog_title}** ({len(plan.tasks)} sections, {plan.blog_kind})")
        for t in plan.tasks:
            log(f"   • Section {t.id}: {t.title} (~{t.target_words} words)")

    return {"plan": plan, "plan_approved": False}


# node: worker

def worker_node(payload: dict, log: Optional[Callable] = None) -> dict:
    task     = Task(**payload["task"])
    plan     = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic    = payload["topic"]
    mode     = payload.get("mode", "closed_book")
    provider = payload["provider"]
    model    = payload["model_name"]

    if log:
        log(f"✍️  Writing section {task.id}: **{task.title}**…")

    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    ) if evidence else "(no evidence)"

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    llm = get_llm(provider, model, temperature=0.5)
    section_md = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n"
            f"Blog kind: {plan.blog_kind}\n"
            f"Constraints: {plan.constraints}\n"
            f"Topic: {topic}\n"
            f"Mode: {mode}\n"
            f"As-of: {payload.get('as_of')}\n\n"
            f"Section title: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"Tags: {task.tags}\n"
            f"requires_research: {task.requires_research}\n"
            f"requires_citations: {task.requires_citations}\n"
            f"requires_code: {task.requires_code}\n"
            f"Bullets:{bullets_text}\n\n"
            f"Evidence:\n{evidence_text}\n"
        )),
    ]).content.strip()

    return {"sections": [(task.id, section_md)]}


# node: reducer

def reducer_node(state: State, log: Optional[Callable] = None) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("Reducer called without a plan.")

    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body    = "\n\n".join(ordered).strip()
    final   = f"# {plan.blog_title}\n\n{body}\n"

    if log:
        log(f"📝 Blog assembled: **{len(final.split())}** words across {len(ordered)} sections.")

    return {"final": final}


# orchestrated pipeline (no human-in-the-loop, for CLI usage)

def run_pipeline(
    topic: str,
    provider: str,
    model_name: str,
    as_of: Optional[str] = None,
    log: Optional[Callable] = None,
) -> str:
    """
    Run the full BWA 4.0 pipeline without HITL pauses.
    Returns the final blog markdown.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    state: State = {
        "topic":          topic,
        "provider":       provider,
        "model_name":     model_name,
        "mode":           "",
        "needs_research": False,
        "queries":        [],
        "evidence":       [],
        "plan":           None,
        "as_of":          as_of,
        "recency_days":   7,
        "router_approved": True,
        "plan_approved":   True,
        "router_feedback": None,
        "plan_feedback":   None,
        "sections":       [],
        "final":          "",
    }

    # Router
    state.update(router_node(state, log))

    # Research (conditional)
    if state["needs_research"]:
        state.update(research_node(state, log))

    # Orchestrator
    state.update(orchestrator_node(state, log))

    # Workers (parallel payload building, sequential execution here)
    plan = state["plan"]
    for task in plan.tasks:
        payload = {
            "task":       task.model_dump(),
            "topic":      state["topic"],
            "mode":       state["mode"],
            "plan":       plan.model_dump(),
            "evidence":   [e.model_dump() for e in state.get("evidence", [])],
            "as_of":      state["as_of"],
            "recency_days": state["recency_days"],
            "provider":   state["provider"],
            "model_name": state["model_name"],
        }
        result = worker_node(payload, log)
        state["sections"] = state["sections"] + result["sections"]

    # Reducer
    state.update(reducer_node(state, log))

    return state["final"]