
import pandas as pd

from state import AgentState

def profiling(state:AgentState):
    df=state["dataframe"]
    metadata={
    "rows": df.shape[0],
    "columns":df.shape[1],
    "column_names": df.columns.tolist(),
    "data_types":df.dtypes.astype("str").to_dict() ,
    "missing_values":df.isnull().sum().to_dict() ,
    "duplicate_rows": int(df.duplicated().sum()),
    "numerical_columns":df.select_dtypes(include="number").columns.tolist() ,
    "categorical_columns":df.select_dtypes(include=["object","category"]).columns.tolist(),
    "memory_usage_mb": round(df.memory_usage(deep=True).sum()/(1024*1024),2)
    }

    return {"dataset_metadata":metadata}