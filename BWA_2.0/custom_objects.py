from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentance describing what the reader should be able to do/understand after this section."
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 concrete, non-overlaping subpoints to cover in this section."
    )
    target_words: int = Field(
        ...,
        description="Target word count for this section (120-450)."
    )
    section_type: Literal[
         "intro", "core", "examples", "checklist", "common_mistakes", "conclusion"
    ] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan."          
    )
    

class Plan(BaseModel):
    blog_title: str
    audience: str = Field(..., description="Who is this blog for.")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp).")
    task: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[list[str], operator.add]
    final: str