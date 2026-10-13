import json

from langchain_core.messages import AIMessage

from models.llm import get_llm
from schemas.cleaning import CleaningPlan
from state import AgentState
from tools.issue_detector import detect_issues

llm = get_llm().with_structured_output(CleaningPlan)


def cleaning_agent(state: AgentState):

    df = state["dataframe"]
    user_query = state["user_query"]

    # Detect issues using Python
    issues = detect_issues(df)

    # If no issues are found, skip recommendations
    if not issues:
        return {
            "messages": [
                AIMessage(content="No cleaning issues detected.")
            ]
        }

    prompt = f"""
You are an expert Data Cleaning Assistant.

The following issues were detected by Python:

{json.dumps(issues, indent=2)}

User Query:
{user_query}

Instructions:

1. Recommend the best cleaning method for EACH issue.
2. Recommend only for the detected issues.
3. Do NOT invent new issues.
4. Each recommendation must refer to exactly ONE column.
5. Provide alternative cleaning methods.
6. Explain why your recommendation is the best.
7. Return ONLY structured output.
"""

    cleaning_plan = llm.invoke(prompt)

    return {
        "cleaning_plan": cleaning_plan,
        "messages": [
            AIMessage(content="Cleaning recommendations generated successfully.")
        ],
    }