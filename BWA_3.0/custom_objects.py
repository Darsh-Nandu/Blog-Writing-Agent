import operator
from typing import List, Annotated, TypedDict, Optional, Literal
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section."
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-6 concrete, non-overlapping subpoints to cover in this section."
    )
    target_words: int = Field(..., description="Target word count for this section (120-550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool= False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: str = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(..., description="This list contains fw queries that will be searched on the web.")
    max_results_per_query: int = Field(5, description="How many results to fetch per query (3-8).")

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class State(TypedDict):
    topic: str  

    # Research / Routing
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # Recency
    as_of: str          # ISO date, e.g. "2026-01-29"
    recency_days: int   # 7 for weekly news, 30 for hybrid, etc 

    # Workers
    sections: Annotated[List[tuple[int, str]], operator.add] #(task_id, section_md)
    final: str


