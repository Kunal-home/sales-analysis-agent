
import json

from langchain_core.messages import AIMessage

from models.llm import get_llm
from state import AgentState


llm = get_llm()


def answer_agent(state: AgentState):

    user_query = state["user_query"]
    analysis_result = state.get("analysis_result") or {}

    # Optional cleaning information
    cleaned_df = state.get("cleaned_dataframe")
    cleaning_report = []

    if cleaned_df is not None:
        cleaning_report = cleaned_df.attrs.get(
            "cleaning_report", []
        )

    # Serialize the analysis plan safely
    analysis_plan = state.get("analysis_plan")

    if analysis_plan is not None:
        if hasattr(analysis_plan, "model_dump"):
            analysis_plan = analysis_plan.model_dump(
                mode="json"
            )
        elif hasattr(analysis_plan, "dict"):
            analysis_plan = analysis_plan.dict()

    # Prepare verified results for the LLM
    results_json = json.dumps(
        analysis_result,
        indent=2,
        default=str
    )

    plan_json = json.dumps(
        analysis_plan,
        indent=2,
        default=str
    )

    cleaning_json = json.dumps(
        cleaning_report,
        indent=2,
        default=str
    )

    
    prompt = f"""
You are a senior data analyst and business advisor.

Your goal is to answer the user's question, explain what the data
reveals, compare performance using numbers and percentages when
available, and recommend useful next steps based on the findings.

USER QUESTION:
{user_query}

VERIFIED ANALYSIS RESULTS:
{json.dumps(
    analysis_result,
    indent=2,
    default=str
)}

ANALYSIS PLAN:
{json.dumps(
    analysis_plan,
    indent=2,
    default=str
)}

CLEANING INFORMATION:
{json.dumps(
    cleaning_report,
    indent=2,
    default=str
)}

Follow this response structure:

## 1. Direct Answer

- Start by answering the user's original question directly.
- Summarize the main result in 1-3 sentences.
- Include relevant totals, dates, quantities, or percentages
  when they are available in the computed results.
- Do not begin with a generic introduction.

## 2. Analysis and Key Findings

Explain what the results mean, rather than merely repeating
the table.

Identify the most relevant findings, such as:
- Best and worst performing categories, products, or segments.
- Differences between categories or groups.
- Each group's contribution to the total, if available.
- Important gaps, unusual values, or patterns.
- Changes over time, but only when multiple periods are available.

Use bullet points with descriptive labels.

### Percentage comparisons

- Include percentages when they are provided in the results.
- Explain percentage shares, growth rates, and differences
  when those metrics are available.
- Compare groups using the same metric and comparable periods.
- Clearly distinguish percentage change from percentage-point
  differences.
- Never invent percentages or perform calculations that the
  analysis results do not provide.
- If a useful comparison cannot be calculated from the supplied
  results, explain the limitation instead of guessing.

## 3. Recommendations

Give 2-3 practical recommendations when the data supports them.

Recommendations should:
- Follow logically from the observed findings.
- Explain what the user could investigate or do next.
- Prioritize meaningful opportunities or potential concerns.
- Be specific to the actual dataset and question.
- Avoid generic business advice that could apply to any dataset.
- Distinguish evidence-backed conclusions from hypotheses
  that need further investigation.

Do not recommend an action as certain to improve results
unless the analysis provides evidence for that conclusion.

## 4. Assumptions

Briefly mention relevant issues such as:
- Small sample sizes or incomplete data.
- Missing values that may affect the analysis.
- Date-range assumptions.
- Missing comparison periods.
- Information needed to make a stronger recommendation.

Include this section only when relevant.

STRICT RULES:

1. Use the verified analysis results as the source of numerical facts.
2. Never invent figures, totals, percentages, dates, or trends.
3. Do not perform calculations that the analysis results do not support.
4. Do not claim causation from a simple correlation or ranking.
5. Distinguish rows, units, orders, revenue, and other metrics.
6. If no matching data is returned, clearly say so.
7. If there is insufficient information, explain what is missing.
8. Use ₹ for monetary values when the currency is known to be INR.
9. Do not convert currencies.
10. Do not mention Python, Pandas, agents, prompts, or internal logic.
11. Do not repeat every row of the result table in the explanation.
12. Adapt the length to the complexity of the question.
13. Recommendations must be grounded in the available findings.

FORMATTING:
- Use Markdown headings, bold text, and concise bullet points.
- Answer first, analyze second, recommend third, and explain
  caveats last when needed.
- Keep simple questions concise and complex questions detailed.
- Return only the natural-language answer in Markdown.
"""
 

    response = llm.invoke(prompt)

    answer = response.content

    # Handle models that return a non-string content structure
    if not isinstance(answer, str):
        answer = str(answer)

    answer = answer.strip()

    return {
        "analysis_answer": answer,
        "messages": [
            AIMessage(content=answer)
        ]
    }
