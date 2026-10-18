from state import AgentState
from tools.cleaner import apply_cleaning


def cleaning_executor(state: AgentState):

    df = state["dataframe"]
    cleaning_plan = state["cleaning_plan"]

    cleaned_df = apply_cleaning(
        df,
        cleaning_plan,
        allow_outlier_changes=False,
    )

    return {
        "cleaned_dataframe": cleaned_df
    }