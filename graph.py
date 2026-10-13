from langgraph.graph import StateGraph, END, START

from state import AgentState
from agents.data_agent import data_agent
from agents.profiling_agent import profiling
from agents.planner_agent import planner_agent
from agents.cleaning_agent import cleaning_agent
from agents.analysis_agent import analysis_agent
from tools.analysis_executor import analysis_executor
from agents.chart_agent import chart_node
from tools.cleaning_executor import cleaning_executor
from agents.answer_agent import answer_agent


def planner_router(state: AgentState):

    decision = state["planner_decision"]

    print("Planner Decision:", decision)
    print("Next Agent:", decision.next_agent)

    return decision.next_agent


builder = StateGraph(AgentState)

builder.add_node("data_agent", data_agent)
builder.add_node("profiling", profiling)
builder.add_node("planner_agent", planner_agent)

builder.add_node("cleaning", cleaning_agent)
builder.add_node("cleaning_executor", cleaning_executor)

builder.add_node("analysis", analysis_agent)
builder.add_node("analysis_executor", analysis_executor)

builder.add_node("answer_agent", answer_agent)
builder.add_node("generate_chart", chart_node)


builder.add_edge(START, "data_agent")

builder.add_edge("data_agent", "profiling")

builder.add_edge("profiling", "planner_agent")


builder.add_conditional_edges(
    "planner_agent",
    planner_router,
    {
        "cleaning": "cleaning",
        "analysis": "analysis",
    }
)


# Cleaning path
builder.add_edge("cleaning", "cleaning_executor")
builder.add_edge("cleaning_executor", "analysis")


# Analysis path
builder.add_edge("analysis", "analysis_executor")

# Answer AFTER actual calculation
builder.add_edge("analysis_executor", "answer_agent")

# Chart AFTER actual calculation
builder.add_edge("answer_agent", "generate_chart")

builder.add_edge("generate_chart", END)


graph = builder.compile()