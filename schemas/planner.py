from typing import Literal
from pydantic import BaseModel,Field


class PlannerDecision(BaseModel):
    """
    Decision made by the Planner Agent.
    """
    needs_cleaning:bool=Field(description="True if dataset needs cleaning, False otherwise")
    next_agent: Literal[
        "cleaning",
        "analysis",
    ] = Field(
        description="The next agent to execute."
    )
    reason: str = Field(
        description="Reason for choosing the next agent."
    )
