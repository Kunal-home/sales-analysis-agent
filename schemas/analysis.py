from pydantic import BaseModel, Field, field_validator
from typing import Literal


# ==============================
# TIME FILTER
# ==============================

class TimeFilter(BaseModel):
    """
    Represents a generic time-based filter.  

    Examples:

    last 10 days
    {
        "kind": "rolling",
        "amount": 10,
        "unit": "day"
    }

    last 3 weeks
    {
        "kind": "rolling",
        "amount": 3,
        "unit": "week"
    }

    last month
    {
        "kind": "calendar",
        "amount": 1,
        "unit": "month",
        "period": "previous"
    }

    this month
    {
        "kind": "calendar",
        "amount": 1,
        "unit": "month",
        "period": "current"
    }
    """

    kind: Literal["rolling", "calendar"]

    amount: int = Field(
        default=1,
        ge=1,
        description="Number of time units."
    )

    unit: Literal[
        "day",
        "week",
        "month",
        "year"
    ]

    period: Literal[
        "current",
        "previous"
    ] | None = None


# ==============================
# METRIC
# ==============================

class Metric(BaseModel):

    column: str

    aggregation: Literal[
        "sum",
        "mean",
        "count",
        "count_distinct",
        "min",
        "max",
        "median",
        "ratio",
        "growth"
    ]
    denominator_column: str | None = None


# ==============================
# FILTER
# ==============================

class Filter(BaseModel):

    type: Literal[
        "time",
        "equals",
        "greater_than",
        "less_than",
        "greater_equal",
        "less_equal"
    ]

    column: str

    value: str | int | float | None = None

    time: TimeFilter | None = None


# ==============================
# SORT
# ==============================

class Sort(BaseModel):

    column: str

    direction: Literal[
        "asc",
        "desc"
    ]


# ==============================
# ANALYSIS PLAN
# ==============================

class AnalysisPlan(BaseModel):

    dimensions: list[str] | None = Field(
        default=None,
        description=(
            "Columns used for grouping or breaking down the analysis. "
            "Use an empty array when there is no grouping."
        )
    )

    metrics: list[Metric] = Field(
        description="Numerical calculations that must be performed."
    )

    filters: list[Filter] | None = Field(
        default=None,
        description=(
            "Filters to apply before calculating metrics. "
            "Use an empty array when there are no filters."
        )
    )

    sort: Sort | None = None

    limit: int | None = Field(
        default=None,
        ge=1
    )

    chart: Literal[
        "bar",
        "line",
        "pie",
        "scatter",
        "histogram",
        "none"
    ] = "none"

    reason: str

    # --------------------------
    # Normalize NULL dimensions
    # --------------------------

    @field_validator("dimensions", mode="before")
    @classmethod
    def normalize_dimensions(cls, value):

        if value is None:
            return []

        return value

    # --------------------------
    # Normalize NULL filters
    # --------------------------

    @field_validator("filters", mode="before")
    @classmethod
    def normalize_filters(cls, value):

        if value is None:
            return []

        return value