from state import AgentState
from tools.data_loader import load_data


def data_agent(state:AgentState):
    file=state["uploaded_file"]
    if file is None:
        raise ValueError("No uploaded file")
    dataframe=load_data(file)

    return{"dataframe":dataframe}