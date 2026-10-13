from state import AgentState
from tools.chart_generator import generate_chart
def chart_node(state: AgentState):

    analysis_result = state["analysis_result"]

    fig = generate_chart(analysis_result)

    if fig is None:
        return {
            "generated_charts": []
        }

    return {
        "generated_charts": [fig]
    }