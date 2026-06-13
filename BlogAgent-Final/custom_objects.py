import operator
from typing import List, Annotated, TypedDict, Optional, Literal
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: int
    title: str = Field(
        default="",
        description="Short display title for this section (e.g. 'Getting Started', 'Key Trade-offs')."
    )
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should know/do after this section."
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3-6 concrete, non-overlapping subpoints to cover in this section."
    )
    target_words: int = Field(..., description="Target word count for this section (150-600).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

    def model_post_init(self, __context):
        # If the LLM forgot to supply a title, derive one from the goal
        if not self.title:
            self.title = self.goal[:60].rstrip(".").strip()


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
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(..., description="Search queries to fetch from the web.")
    max_results_per_query: int = Field(5, description="How many results per query (3-8).")


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class HumanFeedback(BaseModel):
    approved: bool
    feedback: Optional[str] = None  # User notes / modifications


class State(TypedDict):
    topic: str

    # Provider settings
    provider: str          # "groq" | "ollama"
    model_name: str

    # Research / Routing
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # Recency
    as_of: str
    recency_days: int

    # Human-in-the-loop checkpoints
    router_approved: bool
    plan_approved: bool
    router_feedback: Optional[str]
    plan_feedback: Optional[str]

    # Workers
    sections: Annotated[List[tuple], operator.add]  # (task_id, section_md)
    final: str