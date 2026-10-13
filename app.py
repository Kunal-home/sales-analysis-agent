import streamlit as st

from graph import graph


st.set_page_config(
    page_title="Data Analytics AI Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Analytics AI Agent")

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "json", "parquet"],
)
user_query = st.text_input(
    "Ask a question about your dataset",
    placeholder="e.g. Show average sales by outlet"
    )

analyze = st.button(
    "🚀 Analyze Dataset",
    type="primary",
    use_container_width=True,
)
if analyze:

    if uploaded_file is None:
        st.warning("Please upload a dataset.")
        st.stop()

    if not user_query.strip():
        st.warning("Please enter your question.")
        st.stop()

    initial_state = {
        "messages": [],
        "uploaded_file": uploaded_file,
        "user_query": user_query,
        "dataframe": None,
        "dataset_metadata": {},
        "planner_decision": None,
        "analysis_result": {},
        "generated_charts": [],
        "cleaning_plan": None,
        "analysis_plan": None
    }

    with st.spinner("Analyzing dataset..."):

        result = graph.invoke(initial_state)

    st.success("Dataset analyzed successfully ✅")

    st.subheader("📊 Dataset Metadata")
    st.json(result["dataset_metadata"])

   

    st.subheader("🧠 Planner Decision")
    st.json(result["planner_decision"].model_dump())

    if result.get("cleaning_plan"):
        st.subheader("🧹 Cleaning Plan")
        st.json(result["cleaning_plan"].model_dump())

    if result.get("analysis_plan"):
        st.subheader("📈 Analysis Plan")
        st.json(result["analysis_plan"].model_dump())

    if result.get("analysis_result"):
        st.subheader("📈 Analysis Result")
        st.json(result["analysis_result"])

    if result.get("generated_charts"):

        st.subheader("📊 Visualization")

        for fig in result["generated_charts"]:
            st.plotly_chart(
                fig,
                use_container_width=True
            )