import operator
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field

class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What to cover.")

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str
    