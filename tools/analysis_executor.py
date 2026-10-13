import pandas as pd

from state import AgentState
from schemas.analysis import Filter
from langchain_core.messages import AIMessage


# =========================================================
# TIME FILTER
# =========================================================

def apply_time_filter(
    df: pd.DataFrame,
    column: str,
    time_filter
) -> pd.DataFrame:

    if column not in df.columns:
        raise ValueError(
            f"Date column '{column}' not found."
        )

    # Convert column to datetime
    dates = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    if dates.isna().all():
        raise ValueError(
            f"Column '{column}' does not contain valid dates."
        )

    # Use latest date in dataset as anchor
    anchor = dates.max().normalize()

    amount = time_filter.amount
    unit = time_filter.unit
    kind = time_filter.kind
    period = time_filter.period

    # =====================================================
    # ROLLING PERIOD
    # =====================================================

    if kind == "rolling":

        # ---------------------------------------------
        # DAYS
        # ---------------------------------------------

        if unit == "day":

            start = (
                anchor
                - pd.Timedelta(days=amount - 1)
            )

            end = (
                anchor
                + pd.Timedelta(days=1)
            )

        # ---------------------------------------------
        # WEEKS
        # ---------------------------------------------

        elif unit == "week":

            total_days = amount * 7

            start = (
                anchor
                - pd.Timedelta(days=total_days - 1)
            )

            end = (
                anchor
                + pd.Timedelta(days=1)
            )

        # ---------------------------------------------
        # MONTHS
        # ---------------------------------------------

        elif unit == "month":

            end = (
                anchor
                + pd.Timedelta(days=1)
            )

            start = (
                end
                - pd.DateOffset(months=amount)
            )

            start = start.normalize()

        # ---------------------------------------------
        # YEARS
        # ---------------------------------------------

        elif unit == "year":

            end = (
                anchor
                + pd.Timedelta(days=1)
            )

            start = (
                end
                - pd.DateOffset(years=amount)
            )

            start = start.normalize()

        else:

            raise ValueError(
                f"Unsupported rolling unit: {unit}"
            )

    # =====================================================
    # CALENDAR PERIOD
    # =====================================================

    elif kind == "calendar":

        if period not in {"current", "previous"}:

            raise ValueError(
                "Calendar filter requires "
                "'current' or 'previous' period."
            )

        # ---------------------------------------------
        # MONTH
        # ---------------------------------------------

        if unit == "month":

            current_month_start = (
                anchor
                .to_period("M")
                .start_time
            )

            if period == "current":

                start = current_month_start

                end = (
                    start
                    + pd.DateOffset(
                        months=amount
                    )
                )

            else:

                end = current_month_start

                start = (
                    end
                    - pd.DateOffset(
                        months=amount
                    )
                )

        # ---------------------------------------------
        # YEAR
        # ---------------------------------------------

        elif unit == "year":

            current_year_start = pd.Timestamp(
                year=anchor.year,
                month=1,
                day=1
            )

            if period == "current":

                start = current_year_start

                end = (
                    start
                    + pd.DateOffset(
                        years=amount
                    )
                )

            else:

                end = current_year_start

                start = (
                    end
                    - pd.DateOffset(
                        years=amount
                    )
                )

        # ---------------------------------------------
        # WEEK
        # ---------------------------------------------

        elif unit == "week":

            current_week_start = (
                anchor
                - pd.Timedelta(
                    days=anchor.weekday()
                )
            ).normalize()

            if period == "current":

                start = current_week_start

                end = (
                    start
                    + pd.Timedelta(
                        weeks=amount
                    )
                )

            else:

                end = current_week_start

                start = (
                    end
                    - pd.Timedelta(
                        weeks=amount
                    )
                )

        # ---------------------------------------------
        # DAY
        # ---------------------------------------------

        elif unit == "day":

            if period == "current":

                start = anchor

                end = (
                    start
                    + pd.Timedelta(
                        days=amount
                    )
                )

            else:

                end = anchor

                start = (
                    end
                    - pd.Timedelta(
                        days=amount
                    )
                )

        else:

            raise ValueError(
                f"Unsupported calendar unit: {unit}"
            )

    else:

        raise ValueError(
            f"Unsupported time filter kind: {kind}"
        )

    # =====================================================
    # APPLY DATE MASK
    # =====================================================

    mask = (
        (dates >= start)
        &
        (dates < end)
    )

    filtered_df = df.loc[mask].copy()

    # =====================================================
    # DEBUG
    # =====================================================

    print("\n==============================")
    print("TIME FILTER")
    print("==============================")

    print("Column:", column)
    print("Kind:", kind)
    print("Amount:", amount)
    print("Unit:", unit)
    print("Period:", period)
    print("Anchor:", anchor)
    print("Start:", start)
    print("End:", end)
    print("Rows:", len(filtered_df))

    return filtered_df


# =========================================================
# GENERAL FILTERS
# =========================================================

def apply_filters(
    df: pd.DataFrame,
    filters: list[Filter]
) -> pd.DataFrame:

    result = df.copy()

    for f in filters:

        # =================================================
        # TIME FILTER
        # =================================================

        if f.type == "time":

            if f.time is None:

                raise ValueError(
                    "Time filter requires a time configuration."
                )

            result = apply_time_filter(
                result,
                f.column,
                f.time
            )

        # =================================================
        # EQUALS
        # =================================================

        elif f.type == "equals":

            if f.column not in result.columns:

                raise ValueError(
                    f"Column '{f.column}' not found."
                )

            if f.value is None:

                raise ValueError(
                    "Equals filter requires a value."
                )

            result = result[
                result[f.column] == f.value
            ]

        # =================================================
        # GREATER THAN
        # =================================================

        elif f.type == "greater_than":

            if f.column not in result.columns:

                raise ValueError(
                    f"Column '{f.column}' not found."
                )

            if f.value is None:

                raise ValueError(
                    "Filter value is required."
                )

            result = result[
                result[f.column] > f.value
            ]

        # =================================================
        # LESS THAN
        # =================================================

        elif f.type == "less_than":

            if f.column not in result.columns:

                raise ValueError(
                    f"Column '{f.column}' not found."
                )

            if f.value is None:

                raise ValueError(
                    "Filter value is required."
                )

            result = result[
                result[f.column] < f.value
            ]

        # =================================================
        # GREATER OR EQUAL
        # =================================================

        elif f.type == "greater_equal":

            if f.column not in result.columns:

                raise ValueError(
                    f"Column '{f.column}' not found."
                )

            if f.value is None:

                raise ValueError(
                    "Filter value is required."
                )

            result = result[
                result[f.column] >= f.value
            ]

        # =================================================
        # LESS OR EQUAL
        # =================================================

        elif f.type == "less_equal":

            if f.column not in result.columns:

                raise ValueError(
                    f"Column '{f.column}' not found."
                )

            if f.value is None:

                raise ValueError(
                    "Filter value is required."
                )

            result = result[
                result[f.column] <= f.value
            ]

        else:

            raise ValueError(
                f"Unsupported filter type: {f.type}"
            )

    return result


# =========================================================
# MISSING DIMENSION HANDLER
# =========================================================

def separate_missing_dimension_groups(
    result_df: pd.DataFrame,
    dimensions: list[str]
):
    """
    Separate rows where one or more grouping dimensions
    are missing or blank.

    The missing group is NOT treated as a real category.

    Returns:

        valid_result
        missing_result
    """

    if not dimensions:

        return result_df, None

    missing_mask = pd.Series(
        False,
        index=result_df.index
    )

    for dimension in dimensions:

        if dimension not in result_df.columns:

            continue

        column = result_df[dimension]

        dimension_missing = (
            column.isna()
            |
            column.astype(str)
            .str.strip()
            .eq("")
        )

        missing_mask = (
            missing_mask
            |
            dimension_missing
        )

    missing_rows = result_df[
        missing_mask
    ].copy()

    valid_rows = result_df[
        ~missing_mask
    ].copy()

    if missing_rows.empty:

        missing_rows = None

    return valid_rows, missing_rows


# =========================================================
# ANALYSIS EXECUTOR
# =========================================================

def analysis_executor(state: AgentState):

    # =====================================================
    # SELECT DATAFRAME
    # =====================================================

    cleaned_df = state.get("cleaned_dataframe")

    if cleaned_df is not None:

        print("\nUsing CLEANED dataframe")

        df = cleaned_df

    else:

        print("\nUsing ORIGINAL dataframe")

        df = state["dataframe"]

    # =====================================================
    # GET PLAN
    # =====================================================

    plan = state["analysis_plan"]

    print("\n==============================")
    print("ANALYSIS PLAN")
    print("==============================")

    print(plan.model_dump())

    # =====================================================
    # VALIDATE DIMENSIONS
    # =====================================================

    dimensions = plan.dimensions or []

    for column in dimensions:

        if column not in df.columns:

            raise ValueError(
                f"Dimension column "
                f"'{column}' not found in dataset."
            )

    # =====================================================
    # VALIDATE METRICS
    # =====================================================

    for metric in plan.metrics:

        # ---------------------------------------------
        # MAIN METRIC COLUMN
        # ---------------------------------------------

        if metric.column not in df.columns:

            raise ValueError(
                f"Metric column "
                f"'{metric.column}' not found in dataset."
            )

        # ---------------------------------------------
        # RATIO VALIDATION
        # ---------------------------------------------

        if metric.aggregation == "ratio":

            if not metric.denominator_column:

                raise ValueError(
                    f"Ratio metric for "
                    f"'{metric.column}' requires "
                    f"denominator_column."
                )

            if metric.denominator_column not in df.columns:

                raise ValueError(
                    f"Denominator column "
                    f"'{metric.denominator_column}' "
                    f"not found in dataset."
                )

        # ---------------------------------------------
        # GROWTH VALIDATION
        # ---------------------------------------------

        if metric.aggregation == "growth":

            if len(dimensions) != 1:

                raise ValueError(
                    "Growth analysis requires exactly "
                    "one time dimension."
                )

            if not pd.api.types.is_numeric_dtype(
                df[metric.column]
            ):

                raise ValueError(
                    f"Growth metric column "
                    f"'{metric.column}' must be numeric."
                )

    # =====================================================
    # FILTERS
    # =====================================================

    filters = plan.filters or []

    print("\nFILTERS:")

    print([
        filter_item.model_dump()
        for filter_item in filters
    ])

    # =====================================================
    # APPLY FILTERS
    # =====================================================

    rows_before_filters = len(df)

    df = apply_filters(
        df,
        filters
    )

    rows_after_filters = len(df)

    if df.empty:

        raise ValueError(
            "No rows remain after applying the filters."
        )

    print(
        f"\nRows before filtering: "
        f"{rows_before_filters}"
    )

    print(
        f"Rows after filtering: "
        f"{rows_after_filters}"
    )

    # =====================================================
    # RESULT DATAFRAME
    # =====================================================

    result_df = None

    missing_dimension_result = None

    # =====================================================
    # GROUPED ANALYSIS
    # =====================================================

    if dimensions:

        print(
            "\nRunning GROUPED analysis"
        )

        group_columns = dimensions

        grouped = df.groupby(
            group_columns,
            dropna=False
        )

        # -----------------------------------------
        # CREATE BASE RESULT
        # -----------------------------------------

        result_df = (
            grouped
            .size()
            .reset_index(
                name="_row_count"
            )
            .drop(
                columns=["_row_count"]
            )
        )

        # -----------------------------------------
        # CALCULATE METRICS
        # -----------------------------------------

        for metric in plan.metrics:

            column = metric.column

            aggregation = metric.aggregation

            # -------------------------------------
            # COUNT ROWS
            # -------------------------------------

            if aggregation == "count":

                values = grouped[
                    column
                ].count()

                metric_name = column

            # -------------------------------------
            # COUNT DISTINCT
            # -------------------------------------

            elif aggregation == "count_distinct":

                values = grouped[
                    column
                ].nunique()

                metric_name = column

            # -------------------------------------
            # RATIO
            #
            # SUM(amount)
            # /
            # COUNT DISTINCT(entity)
            #
            # Example:
            #
            # SUM(SALES)
            # /
            # COUNT DISTINCT(ORDERNUMBER)
            #
            # -------------------------------------

            elif aggregation == "ratio":

                denominator_column = (
                    metric.denominator_column
                )

                if not denominator_column:

                    raise ValueError(
                        "Ratio metric requires "
                        "denominator_column."
                    )

                if denominator_column not in df.columns:

                    raise ValueError(
                        f"Denominator column "
                        f"'{denominator_column}' "
                        f"not found."
                    )

                # Numerator:
                # SUM(amount)
                numerator = grouped[
                    column
                ].sum()

                # Denominator:
                # COUNT DISTINCT(entity)
                denominator = grouped[
                    denominator_column
                ].nunique()

                # Avoid division by zero
                denominator = denominator.replace(
                    0,
                    pd.NA
                )

                # Final ratio
                values = (
                    numerator
                    / denominator
                )

                metric_name = column

            # -------------------------------------
            # PERIOD-OVER-PERIOD GROWTH
            #
            # Example:
            #
            # 2003 SALES = 100
            # 2004 SALES = 120
            #
            # growth = ((120 - 100) / 100) * 100
            #        = 20%
            #
            # -------------------------------------

            elif aggregation == "growth":

                if len(group_columns) != 1:

                    raise ValueError(
                        "Growth analysis requires exactly "
                        "one time dimension."
                    )

                if not pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    raise ValueError(
                        f"Growth metric column "
                        f"'{column}' must be numeric."
                    )

                # First calculate total value
                # for every period.
                period_values = (
                    grouped[column]
                    .sum()
                    .sort_index()
                )

                # Percentage change from
                # previous period.
                values = (
                    period_values
                    .pct_change(fill_method=None)
                    * 100
                )

                # IMPORTANT:
                # Do not overwrite SALES if another
                # SALES metric exists in the result.
                metric_name = (
                    f"{column}_growth"
                )

            # -------------------------------------
            # OTHER AGGREGATIONS
            # -------------------------------------

            else:

                if (
                    not pd.api.types.is_numeric_dtype(
                        df[column]
                    )
                    and aggregation in {
                        "sum",
                        "mean",
                        "median"
                    }
                ):

                    raise ValueError(
                        f"Cannot apply "
                        f"{aggregation} "
                        f"to non-numeric column "
                        f"'{column}'."
                    )

                values = grouped[
                    column
                ].agg(
                    aggregation
                )

                metric_name = column

            # -------------------------------------
            # CONVERT TO DATAFRAME
            # -------------------------------------

            metric_df = (
                values
                .reset_index(
                    name=metric_name
                )
            )

            # -------------------------------------
            # HANDLE DUPLICATE METRIC NAMES
            # -------------------------------------

            #
            # Example:
            #
            # metrics:
            # SALES sum
            # SALES growth
            #
            # SALES growth is already named
            # SALES_growth, so both can coexist.
            #

            duplicate_columns = [
                col
                for col in metric_df.columns
                if (
                    col in result_df.columns
                    and col not in group_columns
                )
            ]

            if duplicate_columns:

                for duplicate_column in duplicate_columns:

                    new_name = (
                        f"{duplicate_column}_"
                        f"{aggregation}"
                    )

                    metric_df = metric_df.rename(
                        columns={
                            duplicate_column: new_name
                        }
                    )

            # -------------------------------------
            # MERGE METRIC
            # -------------------------------------

            result_df = result_df.merge(
                metric_df,
                on=group_columns,
                how="left"
            )

        # -----------------------------------------
        # SEPARATE MISSING DIMENSIONS
        #
        # BEFORE RANKING / LIMITING
        # -----------------------------------------

        (
            result_df,
            missing_dimension_result
        ) = separate_missing_dimension_groups(
            result_df,
            group_columns
        )

    # =====================================================
    # OVERALL ANALYSIS
    # =====================================================

    else:

        print(
            "\nRunning OVERALL analysis"
        )

        results = {}

        for metric in plan.metrics:

            column = metric.column

            aggregation = metric.aggregation

            # -------------------------------------
            # COUNT ROWS
            # -------------------------------------

            if aggregation == "count":

                value = df[
                    column
                ].count()

                result_name = column

            # -------------------------------------
            # COUNT DISTINCT
            # -------------------------------------

            elif aggregation == "count_distinct":

                value = df[
                    column
                ].nunique()

                result_name = column

            # -------------------------------------
            # RATIO
            #
            # SUM(amount)
            # /
            # COUNT DISTINCT(entity)
            #
            # -------------------------------------

            elif aggregation == "ratio":

                denominator_column = (
                    metric.denominator_column
                )

                if not denominator_column:

                    raise ValueError(
                        "Ratio metric requires "
                        "denominator_column."
                    )

                if denominator_column not in df.columns:

                    raise ValueError(
                        f"Denominator column "
                        f"'{denominator_column}' "
                        f"not found."
                    )

                # Numerator
                numerator = df[
                    column
                ].sum()

                # Denominator
                denominator = df[
                    denominator_column
                ].nunique()

                if denominator == 0:

                    raise ValueError(
                        "Cannot calculate ratio "
                        "because denominator "
                        "is zero."
                    )

                # Final ratio
                value = (
                    numerator
                    / denominator
                )

                result_name = column

            # -------------------------------------
            # GROWTH
            # -------------------------------------

            elif aggregation == "growth":

                raise ValueError(
                    "Growth analysis requires a "
                    "time dimension. For example, "
                    "YEAR_ID or a monthly period."
                )

            # -------------------------------------
            # OTHER AGGREGATIONS
            # -------------------------------------

            else:

                if (
                    not pd.api.types.is_numeric_dtype(
                        df[column]
                    )
                    and aggregation in {
                        "sum",
                        "mean",
                        "median"
                    }
                ):

                    raise ValueError(
                        f"Cannot apply "
                        f"{aggregation} "
                        f"to non-numeric column "
                        f"'{column}'."
                    )

                value = df[
                    column
                ].agg(
                    aggregation
                )

                result_name = column

            # -------------------------------------
            # HANDLE DUPLICATE RESULT NAMES
            # -------------------------------------

            if result_name in results:

                result_name = (
                    f"{result_name}_"
                    f"{aggregation}"
                )

            results[result_name] = value

        result_df = pd.DataFrame(
            [results]
        )

    # =====================================================
    # SORT
    # =====================================================

    if plan.sort:

        sort_column = plan.sort.column

        # ---------------------------------------------
        # SUPPORT GROWTH SORT NAME
        # ---------------------------------------------

        if sort_column not in result_df.columns:

            possible_growth_column = (
                f"{sort_column}_growth"
            )

            if possible_growth_column in result_df.columns:

                sort_column = possible_growth_column

            else:

                raise ValueError(
                    f"Sort column "
                    f"'{plan.sort.column}' "
                    f"not found in result. "
                    f"Available columns: "
                    f"{list(result_df.columns)}"
                )

        result_df = result_df.sort_values(
            by=sort_column,
            ascending=(
                plan.sort.direction == "asc"
            )
        )

    # =====================================================
    # LIMIT
    # =====================================================

    if plan.limit is not None:

        result_df = result_df.head(
            plan.limit
        )

    # =====================================================
    # RESET INDEX
    # =====================================================

    result_df = (
        result_df
        .reset_index(drop=True)
    )

    if missing_dimension_result is not None:

        missing_dimension_result = (
            missing_dimension_result
            .reset_index(drop=True)
        )

    # =====================================================
    # PRINT RESULT
    # =====================================================

    print("\n==============================")
    print("ANALYSIS RESULT")
    print("==============================")

    print(result_df)

    if missing_dimension_result is not None:

        print("\n==============================")
        print("MISSING DIMENSION GROUP")
        print("==============================")

        print(
            missing_dimension_result
        )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    analysis_result = {

        "result_df":
            result_df.to_dict(
                orient="records"
            ),

        "missing_dimension_result": (

            missing_dimension_result.to_dict(
                orient="records"
            )

            if missing_dimension_result is not None

            else []
        ),

        "chart_type":
            plan.chart,

        "dimensions":
            plan.dimensions,

        "metrics":
            [
                metric.model_dump()
                for metric in plan.metrics
            ],

        "filters":
            [
                filter_item.model_dump()
                for filter_item in filters
            ],

        "sort":
            (
                plan.sort.model_dump()
                if plan.sort
                else None
            ),

        "limit":
            plan.limit,

        "rows_before_filters":
            rows_before_filters,

        "rows_analyzed":
            rows_after_filters,

        "rows_returned":
            len(result_df)
    }

    # =====================================================
    # RETURN STATE
    # =====================================================

    return {

        "analysis_result":
            analysis_result,

        "messages": [

            AIMessage(
                content=(
                    "Analysis calculated successfully "
                    "from the dataset."
                )
            )

        ]
    }