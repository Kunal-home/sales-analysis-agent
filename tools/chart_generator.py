import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def generate_chart(result: dict) -> go.Figure | None:

    chart_type = result.get("chart_type", "none")

    if chart_type == "none":
        return None

    records = result.get("result_df")

    if not records:
        return None

    df = pd.DataFrame(records)

    if df.empty:
        return None

    dimensions = result.get("dimensions", [])
    metrics = result.get("metrics", [])

    # -----------------------------------------
    # Determine X and Y columns
    # -----------------------------------------

    if dimensions:
        x_col = dimensions[0]
    else:
        x_col = None

    if metrics:
        y_col = metrics[0]["column"]
        
        # IMPORTANT:
        # Executor may create names such as:
        # quantity_sum
        # total_price_sum
        # quantity_mean
        aggregation = metrics[0]["aggregation"]

        possible_metric_names = [
            f"{y_col}_{aggregation}",
            y_col
        ]

        y_col = next(
            (col for col in possible_metric_names if col in df.columns),
            None
        )
    else:
        y_col = None

    # -----------------------------------------
    # Safety checks
    # -----------------------------------------

    if chart_type in ["bar", "line", "scatter", "pie"]:

        if x_col is None or y_col is None:
            return None

        if x_col not in df.columns or y_col not in df.columns:
            return None

    # -----------------------------------------
    # Chart title
    # -----------------------------------------

    if dimensions and metrics:
        metric = metrics[0]
        aggregation = metric["aggregation"]
        column = metric["column"]

        title = f"{aggregation.title()} of {column} by {x_col}"

    elif metrics:
        metric = metrics[0]
        title = f"{metric['aggregation'].title()} of {metric['column']}"

    else:
        title = "Analysis Result"

    # -----------------------------------------
    # Create chart
    # -----------------------------------------

    fig = None

    if chart_type == "bar":

        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title,
            text_auto=True
        )

    elif chart_type == "line":

        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title,
            markers=True
        )

    elif chart_type == "pie":

        fig = px.pie(
            df,
            names=x_col,
            values=y_col,
            title=title
        )

    elif chart_type == "scatter":

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title
        )

    elif chart_type == "histogram":

        # For histogram, use the metric column if available.
        histogram_column = y_col

        if histogram_column is None:
            histogram_column = df.columns[0]

        fig = px.histogram(
            df,
            x=histogram_column,
            title=title
        )

    # -----------------------------------------
    # Styling
    # -----------------------------------------

    if fig is not None:

        fig.update_layout(
            template="plotly_dark",
            font=dict(size=13),
            title_font_size=18,
            margin=dict(
                t=50,
                b=40,
                l=40,
                r=20
            )
        )

    return fig