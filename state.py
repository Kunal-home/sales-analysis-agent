from typing import Any
from typing import TypedDict,Annotated
import pandas as pd
from langgraph.graph.message import add_messages,BaseMessage
from schemas.planner import PlannerDecision
from schemas.cleaning import CleaningPlan
from schemas.analysis import AnalysisPlan

class AgentState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    uploaded_file: Any | None
    user_query:str
    dataframe:pd.DataFrame|None
    dataset_metadata:dict
    analysis_result:dict
    generated_charts:list
    planner_decision: PlannerDecision | None
    cleaning_plan: CleaningPlan | None
    analysis_plan:AnalysisPlan|None
    cleaned_dataframe: pd.DataFrame | None
    analysis_answer:str|None

