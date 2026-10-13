import streamlit as st

from graph import graph


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Data Analytics AI Agent",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Data Analytics AI Agent")


# ============================================================
# FILE UPLOAD
# ============================================================
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
            "json",
            "parquet"
        ],
    )


# ============================================================
# USER QUESTION
# ============================================================

user_query = st.chat_input("Ask a question about your sales")


# ============================================================
# ANALYZE BUTTON
# ============================================================



# ============================================================
# RUN ANALYSIS
# ============================================================

if user_query:
    with st.chat_message("user"): st.write(user_query)

    # --------------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------------

    if uploaded_file is None:

        st.warning(
            "Please upload your dataset."
        )

        st.stop()


    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    if not user_query.strip():

        st.warning(
            "Please enter your question."
        )

        st.stop()


    # ========================================================
    # INITIAL LANGGRAPH STATE
    # ========================================================

    initial_state = {

        "messages": [],

        "uploaded_file": uploaded_file,

        "user_query": user_query,

        "dataframe": None,

        "cleaned_dataframe": None,

        "dataset_metadata": {},

        "planner_decision": None,

        "cleaning_plan": None,

        "analysis_plan": None,

        "analysis_result": {},

        "analysis_answer": None,

        "generated_charts": [],

    }


    # ========================================================
    # RUN GRAPH
    # ========================================================

    with st.spinner("Analyzing..."):

        result = graph.invoke(
            initial_state
        )


    st.success(
        "Analysis completed ✅"
    )


    # ========================================================
    # HUMAN-READABLE INSIGHT
    # ========================================================

    if result.get("analysis_answer"):

        st.subheader(
            "💡 Insight"
        )

        st.write(
            result["analysis_answer"]
        )


    # ========================================================
    # ANALYSIS RESULT
    # ========================================================

    analysis_result = result.get(
        "analysis_result",
        {}
    )


    result_df = analysis_result.get(
        "result_df",
        []
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if result_df:

        st.subheader(
            "📊 Result"
        )


        # ====================================================
        # SINGLE VALUE RESULT
        # ====================================================

        if (
            len(result_df) == 1
            and len(result_df[0]) == 1
        ):

            value = list(
                result_df[0].values()
            )[0]


            # -----------------------------------------------
            # Display numeric value
            # -----------------------------------------------

            if isinstance(
                value,
                (int, float)
            ):

                st.metric(
                    label="Result",
                    value=f"{value:,}"
                )

            else:

                st.metric(
                    label="Result",
                    value=str(value)
                )


        # ====================================================
        # MULTIPLE ROWS / COLUMNS
        # ====================================================

        else:

            # ------------------------------------------------
            # DYNAMIC COLUMN CONFIGURATION
            # ------------------------------------------------

            column_config = {}


            # ------------------------------------------------
            # Inspect result columns
            # ------------------------------------------------

            for column in result_df[0].keys():

                # --------------------------------------------
                # Detect growth columns automatically
                #
                # Examples:
                #
                # SALES_growth
                # REVENUE_growth
                # PROFIT_growth
                # AMOUNT_growth
                #
                # --------------------------------------------

                if column.endswith(
                    "_growth"
                ):

                    display_name = (
                        column
                        .replace(
                            "_growth",
                            " Growth"
                        )
                    )


                    column_config[column] = (
                        st.column_config.NumberColumn(

                            label=display_name,

                            format="%.2f%%"

                        )
                    )


            # ------------------------------------------------
            # DISPLAY TABLE
            # ------------------------------------------------

            st.dataframe(

                result_df,

                column_config=column_config,

                use_container_width=True

            )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    charts = result.get(
        "generated_charts",
        []
    )


    if charts:

        st.subheader(
            "📈 Visualization"
        )


        for fig in charts:

            st.plotly_chart(

                fig,

                use_container_width=True

            )


    elif analysis_result:

        st.info(
            "No visualization was generated "
            "for this analysis."
        )