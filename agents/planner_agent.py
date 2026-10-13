import json
import pandas as pd
from schemas.planner import PlannerDecision
from models.llm import get_llm
from state import AgentState

llm=get_llm().with_structured_output(PlannerDecision)

def planner_agent(state:AgentState):

    metadata=state["dataset_metadata"]
    data=state["dataframe"]
    user_query=state["user_query"]
    prompt = f"""
    You are an expert Data Analytics Planner.

    Your job is to decide what should happen next.

    User Query:
    {user_query}

    Dataset Metadata:
    {json.dumps(metadata, indent=2)}

    Rules:

    1. If the dataset has missing values, duplicate rows, or incorrect data types that may affect the requested analysis,
    recommend cleaning first.

    2. If the dataset is already suitable,
    send it directly to analysis.

    3. Do not perform the analysis.

    4. Only return the structured output.
    """
    decision=llm.invoke(prompt)

    return {"planner_decision":decision}
