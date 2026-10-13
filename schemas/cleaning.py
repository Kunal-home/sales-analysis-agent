from pydantic import BaseModel, Field
from typing import Literal,List

CleaningMethod = Literal[
    "fill_mean",
    "fill_median",
    "fill_mode",
    "drop_rows",
    "drop_duplicates",
    "clip_outliers",
    "remove_outliers"
]

class CleaningRecommendation(BaseModel):
    """
    Cleaning recommendation for a single data quality issue.
    """

    columns: list[str] = Field(
    description="Column(s) affected by this issue."
    )

    issue: str = Field(
        description="Type of issue detected, e.g. missing_values, duplicates, outliers, datatype, inconsistent_values, invalid_values."
    )

    recommended_method:CleaningMethod = Field(
        description="Best cleaning method recommended by the AI."
    )

    available_methods:List[CleaningMethod] = Field(
        description="List of all applicable cleaning methods the user can choose from."
    )

    reason: str = Field(
        description="Why the recommended method is the best choice."
    )


class CleaningPlan(BaseModel):
    """
    Complete cleaning plan for the dataset.
    """

    recommendations: list[CleaningRecommendation] = Field(
        description="List of cleaning recommendations for the dataset."
    )