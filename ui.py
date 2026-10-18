
import io

import pandas as pd
import streamlit as st
import plotly.express as px
from graph import graph


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Sales Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================
# CUSTOM STYLING
# =========================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F6F8F7;
        color: #10201F;
    }

    [data-testid="stSidebar"] {
        background-color: #E6EDEB;
        border-right: 1px solid #D0DCD8;
    }

    .hero {
        padding: 1.3rem;
        border: 1px solid #D0DCD8;
        border-radius: 16px;
        background: linear-gradient(
            120deg,
            #FFFFFF,
            #E6EDEB
        );
        margin-bottom: 1rem;
    }

    .muted {
        color: #526763;
        font-size: 0.9rem;
    }

    .small-label {
        color: #0B7A68;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D0DCD8;
        padding: 14px 16px;
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #B9CDC7;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    /* Hide user and assistant chat avatars */
    [data-testid="stChatMessageAvatar"],
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    /* Keep chat messages aligned after hiding avatars */
    [data-testid="stChatMessage"] {
        gap: 0 !important;
    }

    /* Use the application background for the chat input */
    [data-testid="stChatInput"] {
        background-color: #F6F8F7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================
# HELPER FUNCTIONS
# =========================================

def preview_uploaded_file(uploaded_file):
    """Preview uploaded data without consuming its stream."""

    if uploaded_file is None:
        return None

    raw_data = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                try:
                    return pd.read_csv(
                        io.BytesIO(raw_data),
                        encoding=encoding,
                    )
                except UnicodeDecodeError:
                    continue

            return None

        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(raw_data))

        if filename.endswith(".json"):
            try:
                return pd.read_json(io.BytesIO(raw_data))
            except ValueError:
                return pd.read_json(
                    io.BytesIO(raw_data),
                    lines=True,
                )

        if filename.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(raw_data))

    except Exception as exc:
        st.sidebar.warning(f"Could not preview the file: {exc}")

    return None


def safe_json(value):
    """Convert objects to JSON-compatible values for display."""

    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {
            str(key): safe_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def format_number(value):
    """Format numeric values with thousands separators."""

    try:
        number = float(value)

        if number.is_integer():
            return f"{int(number):,}"

        return f"{number:,.2f}"

    except (TypeError, ValueError):
        return str(value)


def show_result_table(records, title="Analysis Results"):
    """Display analysis records in a table."""

    if not records:
        st.info("No result rows were returned.")
        return

    result_df = pd.DataFrame(records)

    st.markdown(f"### {title}")
    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )


def show_analysis_details(result):
    """Display optional analysis plans and execution metadata."""

    with st.expander("How this was computed"):

        for key, label in [
            ("analysis_plan", "Analysis Plan"),
            ("cleaning_plan", "Cleaning Plan"),
            ("planner_decision", "Planner Decision"),
        ]:
            value = result.get(key)

            if value is not None:
                st.markdown(f"**{label}**")
                st.json(safe_json(value))

        analysis_result = result.get("analysis_result") or {}

        detail_keys = [
            "chart_type",
            "dimensions",
            "metrics",
            "filters",
            "sort",
            "limit",
            "rows_before_filters",
            "rows_analyzed",
            "rows_returned",
        ]

        details = {
            key: analysis_result[key]
            for key in detail_keys
            if key in analysis_result
        }

        if details:
            st.markdown("**Execution Details**")
            st.json(safe_json(details))


def render_result(result):
    """Render the answer, analysis results, and charts."""

    analysis_answer = result.get("analysis_answer")
    analysis_result = result.get("analysis_result") or {}

    result_records = analysis_result.get("result_df") or []
    charts = result.get("generated_charts") or []
    missing_records = (
        analysis_result.get("missing_dimension_result") or []
    )

    # Natural-language answer
    if analysis_answer:
        st.markdown("### 💡 Key Insight")
        st.markdown(analysis_answer)

    # Result table or single KPI
    if result_records:
        result_df = pd.DataFrame(result_records)

        if result_df.shape == (1, 1):
            column_name = str(result_df.columns[0])
            value = result_df.iloc[0, 0]

            st.metric(
                label=column_name.replace("_", " ").title(),
                value=format_number(value),
            )
        else:
            show_result_table(result_records)

    # Charts remain enabled
    if charts:
        st.markdown("### 📈 Visualization")

        for fig in charts:
            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    elif analysis_result:
        st.info(
            "Analysis completed, but no chart was generated "
            "for this question."
        )

    # Optional rows with missing dimension values
    if missing_records:
        with st.expander("Rows with missing dimension values"):
            st.dataframe(
                pd.DataFrame(missing_records),
                use_container_width=True,
                hide_index=True,
            )

    show_analysis_details(result)


# =========================================
# SESSION STATE
# =========================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""


# =========================================
# SIDEBAR
# =========================================

with st.sidebar:
    st.markdown("## 📊 Sales Analyst Agent")

    st.caption("Analyze your dataset using natural language.")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
    )

    preview_df = None

    if uploaded_file is not None:
        preview_df = preview_uploaded_file(uploaded_file)

        st.success(f"File uploaded: {uploaded_file.name}")

        if preview_df is not None:
            col1, col2 = st.columns(2)

            col1.metric("Rows", f"{len(preview_df):,}")
            col2.metric("Columns", f"{len(preview_df.columns):,}")

            with st.expander("Detected Columns"):
                for column in preview_df.columns:
                    st.write(f"• `{column}`")

    else:
        st.info("Upload a dataset to get started.")

        st.markdown(
            """
            **Quick Start**

            1. Upload your sales dataset.
            2. Ask a question in plain English.
            3. Review insights and visualizations.
            """
        )

    st.divider()
    st.markdown("### Example Questions")

    example_questions = [
        "What are the total sales?",
        "Show revenue by product",
        "What were the monthly sales?",
        "Which products performed best?",
    ]

    for question in example_questions:
        if st.button(
            question,
            key=f"example_{question}",
            use_container_width=True,
        ):
            st.session_state.pending_query = question

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.analysis_history = []
        st.session_state.latest_result = None
        st.session_state.pending_query = ""
        st.rerun()


# =========================================
# MAIN HEADER
# =========================================

st.markdown(
    """
    <div class="hero">
        <div class="small-label">AI-POWERED ANALYTICS</div>
        <h1 style="margin: 0.35rem 0;">
            Sales Analyst Agent
        </h1>
        <div class="muted">
            Ask questions about your sales data.
            Your LangGraph agents will handle the analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================
# TABS
# =========================================

ask_tab, overview_tab, data_tab = st.tabs(
    ["💬 Ask", "📈 Overview", "🧾 Data"]
)


# =========================================
# ASK TAB
# =========================================

with ask_tab:
    st.markdown("### Ask Your Sales Data")
    st.caption("Example: Show total revenue by product.")

    # Render the existing conversation in order
    for item in st.session_state.analysis_history:
        with st.chat_message("user"):
            st.write(item["query"])

        with st.chat_message("assistant"):
            if item.get("error"):
                st.error(item["error"])
            else:
                render_result(item["result"])

    # IMPORTANT:
    # Keep the chat input in the same main-page tab as the conversation.
    # Do not apply custom position: fixed/top/bottom CSS to this widget.
    typed_query = st.chat_input(
        "Ask a question about your sales...",
        key="sales_chat_input",
    )

    query_to_run = typed_query or st.session_state.pending_query

    if st.session_state.pending_query:
        st.session_state.pending_query = ""

    if query_to_run:
        if uploaded_file is None:
            st.warning("Please upload your dataset first.")

        elif not query_to_run.strip():
            st.warning("Please enter a question.")

        else:
            # Show the question immediately
            with st.chat_message("user"):
                st.write(query_to_run)

            initial_state = {
                "messages": [],
                "uploaded_file": uploaded_file,
                "user_query": query_to_run.strip(),
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

            try:
                uploaded_file.seek(0)

                with st.spinner(
                    "Your AI agents are analyzing the data..."
                ):
                    result = graph.invoke(initial_state)

                st.session_state.latest_result = result

                st.session_state.analysis_history.append(
                    {
                        "query": query_to_run.strip(),
                        "result": result,
                    }
                )

                # Rerender so the answer is saved in conversation history
                # and the input is ready for the next question.
                st.rerun()

            except Exception as exc:
                error_message = f"Analysis failed: {exc}"

                st.session_state.analysis_history.append(
                    {
                        "query": query_to_run.strip(),
                        "error": error_message,
                    }
                )

                st.error(error_message)

                with st.expander("Technical Error Details"):
                    st.exception(exc)


# =========================================
# OVERVIEW TAB
# =========================================

with overview_tab:
    st.markdown("### Dataset Overview")

    if preview_df is None:
        st.info("Upload a dataset to see its overview.")

    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", f"{len(preview_df):,}")
        col2.metric("Total Columns", f"{len(preview_df.columns):,}")
        col3.metric(
            "Missing Cells",
            f"{int(preview_df.isna().sum().sum()):,}",
        )

        st.markdown("#### Column Summary")

        summary_df = pd.DataFrame(
            {
                "Column": preview_df.columns.astype(str),
                "Data Type": [
                    str(dtype) for dtype in preview_df.dtypes
                ],
                "Missing Values": [
                    int(preview_df[col].isna().sum())
                    for col in preview_df.columns
                ],
                "Unique Values": [
                    int(preview_df[col].nunique(dropna=True))
                    for col in preview_df.columns
                ],
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        numeric_columns = preview_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            st.markdown("#### Numeric Summary")
            st.dataframe(
                preview_df[numeric_columns].describe().T,
                use_container_width=True,
            )
            
        # Add this section inside your Overview tab,
        # after the metric cards.

        st.divider()
        st.markdown("### 📊 Data Distribution & Quality")

        numeric_cols = preview_df.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_cols = preview_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        datetime_cols = preview_df.select_dtypes(
            include=["datetime"]
        ).columns.tolist()

        # 1. Missing values chart
        missing = preview_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if not missing.empty:
            st.markdown("#### Missing Values by Column")
            missing_df = missing.rename("Missing Values").reset_index()
            missing_df.columns = ["Column", "Missing Values"]

            fig = px.bar(
                missing_df,
                x="Column",
                y="Missing Values",
                title="Missing Values Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset.")

        # 2. Numeric distribution
        if numeric_cols:
            st.markdown("#### Numeric Data Distribution")

            selected_num = st.selectbox(
                "Select a numeric column",
                numeric_cols,
                key="overview_numeric"
            )

            fig = px.histogram(
                preview_df,
                x=selected_num,
                nbins=30,
                title=f"Distribution of {selected_num}"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 3. Categorical distribution
            if categorical_cols:
                st.markdown("#### Category Breakdown")

                selected_cat = st.selectbox(
                    "Select a categorical column",
                    categorical_cols,
                    key="overview_category"
                )

                category_counts = (
                    preview_df[selected_cat]
                    .fillna("Missing")
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .rename_axis(selected_cat)
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    category_counts,
                    x=selected_cat,
                    y="Count",
                    title=f"Top 10 Values in {selected_cat}"
                )
                st.plotly_chart(fig, use_container_width=True)

        # 4. Trend over time
        if datetime_cols and numeric_cols:
            st.markdown("#### Trend Over Time")

            col1, col2 = st.columns(2)

            with col1:
                selected_date = st.selectbox(
                    "Date column",
                    datetime_cols,
                    key="overview_date"
                )

            with col2:
                selected_value = st.selectbox(
                    "Value column",
                    numeric_cols,
                    key="overview_trend_value"
                )

            trend_df = preview_df[
                [selected_date, selected_value]
            ].dropna()

            if not trend_df.empty:
                trend_df = (
                    trend_df.groupby(selected_date, as_index=False)[selected_value]
                    .sum()
                    .sort_values(selected_date)
                )

                fig = px.line(
                    trend_df,
                    x=selected_date,
                    y=selected_value,
                    markers=True,
                    title=f"{selected_value} Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Trend chart requires at least one datetime column "
                "and one numeric column."
            )


    if st.session_state.latest_result:
        st.markdown("#### Latest Analysis")

        latest = st.session_state.latest_result

        if latest.get("analysis_answer"):
            st.write(latest["analysis_answer"])

        latest_analysis = latest.get("analysis_result") or {}
        metric_columns = st.columns(3)

        if "rows_analyzed" in latest_analysis:
            metric_columns[0].metric(
                "Rows Analyzed",
                format_number(latest_analysis["rows_analyzed"]),
            )

        if "rows_returned" in latest_analysis:
            metric_columns[1].metric(
                "Rows Returned",
                format_number(latest_analysis["rows_returned"]),
            )

        if latest_analysis.get("chart_type"):
            metric_columns[2].metric(
                "Chart Type",
                str(latest_analysis["chart_type"]).title(),
            )


# =========================================
# DATA TAB
# =========================================

with data_tab:
    st.markdown("### Dataset Preview")

    if preview_df is None:
        st.info(
            "Upload a CSV, Excel, JSON, or Parquet file "
            "to preview the data."
        )

    else:
        st.caption(
            f"Showing the first 200 rows of {uploaded_file.name}."
        )

        st.dataframe(
            preview_df.head(200),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Preview as CSV",
            data=preview_df.head(200).to_csv(
                index=False
            ).encode("utf-8"),
            file_name="dataset_preview.csv",
            mime="text/csv",
        )
