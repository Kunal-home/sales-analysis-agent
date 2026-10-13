from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd


if TYPE_CHECKING:
    from schemas.cleaning import CleaningPlan


# Methods that deal with outliers
OUTLIER_METHODS = {
    "clip_outliers",
    "remove_outliers",
}


# Methods that deal with missing values
MISSING_VALUE_METHODS = {
    "fill_mean",
    "fill_median",
    "fill_mode",
    "drop_rows",
}


def _iqr_bounds(series: pd.Series) -> tuple[float, float] | None:
    """
    Calculate lower and upper IQR bounds for a numeric Series.

    Returns
    -------
    tuple[float, float] | None
        (lower_bound, upper_bound), or None if there is
        not enough data to calculate the bounds.
    """

    # Remove missing values before calculating statistics
    series = series.dropna()

    if series.empty:
        return None

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)

    return lower, upper


def apply_cleaning(
    df: pd.DataFrame,
    cleaning_plan: "CleaningPlan | None",
    allow_outlier_changes: bool = False,
) -> pd.DataFrame:
    """
    Apply the LLM-generated CleaningPlan to a copy of the dataset.

    This function performs ONE-TIME, DATASET-LEVEL cleaning.

    Important design decisions
    ---------------------------
    1. The original dataframe is never modified.
    2. Cleaning does not depend on the user's current question.
    3. Numeric missing values can be cleaned.
    4. Missing values in categorical/text columns are left unchanged.
    5. Duplicate rows can be removed.
    6. Outliers are detected but are NOT changed by default.
    7. Outlier changes can be explicitly enabled.
    8. A human-readable cleaning report is stored in:
           cleaned_df.attrs["cleaning_report"]
    """

    # ---------------------------------------------------------
    # 1. ALWAYS work on a copy
    # ---------------------------------------------------------

    cleaned = df.copy()

    # Store information about what was actually done
    cleaning_report: list[str] = []

    # ---------------------------------------------------------
    # 2. No cleaning plan
    # ---------------------------------------------------------

    if not cleaning_plan or not cleaning_plan.recommendations:

        cleaning_report.append(
            "No cleaning actions were required."
        )

        cleaned.attrs["cleaning_report"] = cleaning_report

        return cleaned

    # ---------------------------------------------------------
    # 3. Apply each recommendation
    # ---------------------------------------------------------

    for rec in cleaning_plan.recommendations:

        method = rec.recommended_method

        # Only use columns that actually exist
        columns = [
            col
            for col in rec.columns
            if col in cleaned.columns
        ]

        # -----------------------------------------------------
        # Invalid / missing columns
        # -----------------------------------------------------

        if not columns and method != "drop_duplicates":

            cleaning_report.append(
                f"Skipped '{method}' because none of the "
                f"specified columns exist in the dataset."
            )

            continue

        # =====================================================
        # DUPLICATES
        # =====================================================

        if method == "drop_duplicates":

            rows_before = len(cleaned)

            cleaned = cleaned.drop_duplicates()

            rows_removed = rows_before - len(cleaned)

            cleaning_report.append(
                f"Removed {rows_removed} duplicate rows."
            )

            continue

        # =====================================================
        # PROCESS EACH COLUMN
        # =====================================================

        for col in columns:

            # Current column
            series = cleaned[col]

            # Check whether the column is numeric
            is_numeric = pd.api.types.is_numeric_dtype(series)

            # Number of missing values before cleaning
            missing_before = int(series.isna().sum())

            # =================================================
            # MISSING VALUE METHODS
            # =================================================

            if method in MISSING_VALUE_METHODS:

                # -------------------------------------------------
                # Do NOT automatically modify categorical/text data
                # -------------------------------------------------

                if not is_numeric:

                    if missing_before > 0:

                        cleaning_report.append(
                            f"{col}: '{method}' was not applied because "
                            f"the column is non-numeric. "
                            f"{missing_before} missing values were "
                            f"left unchanged."
                        )

                    else:

                        cleaning_report.append(
                            f"{col}: '{method}' skipped because "
                            f"the column is non-numeric and contains "
                            f"no missing values."
                        )

                    continue

                # -------------------------------------------------
                # Numeric column: fill with mean
                # -------------------------------------------------

                if method == "fill_mean":

                    if missing_before == 0:

                        cleaning_report.append(
                            f"{col}: no missing values found; "
                            f"mean imputation was not needed."
                        )

                        continue

                    mean_value = cleaned[col].mean()

                    # If mean cannot be calculated
                    if pd.isna(mean_value):

                        cleaning_report.append(
                            f"{col}: mean imputation skipped because "
                            f"no valid numeric value was available."
                        )

                        continue

                    cleaned[col] = cleaned[col].fillna(mean_value)

                    cleaning_report.append(
                        f"{col}: filled {missing_before} missing "
                        f"values using mean ({mean_value:.2f})."
                    )

                # -------------------------------------------------
                # Numeric column: fill with median
                # -------------------------------------------------

                elif method == "fill_median":

                    if missing_before == 0:

                        cleaning_report.append(
                            f"{col}: no missing values found; "
                            f"median imputation was not needed."
                        )

                        continue

                    median_value = cleaned[col].median()

                    if pd.isna(median_value):

                        cleaning_report.append(
                            f"{col}: median imputation skipped because "
                            f"no valid numeric value was available."
                        )

                        continue

                    cleaned[col] = cleaned[col].fillna(
                        median_value
                    )

                    cleaning_report.append(
                        f"{col}: filled {missing_before} missing "
                        f"values using median ({median_value:.2f})."
                    )

                # -------------------------------------------------
                # Numeric column: fill with mode
                # -------------------------------------------------

                elif method == "fill_mode":

                    if missing_before == 0:

                        cleaning_report.append(
                            f"{col}: no missing values found; "
                            f"mode imputation was not needed."
                        )

                        continue

                    mode = cleaned[col].mode()

                    if mode.empty:

                        cleaning_report.append(
                            f"{col}: mode imputation skipped because "
                            f"no valid value was available."
                        )

                        continue

                    mode_value = mode.iloc[0]

                    cleaned[col] = cleaned[col].fillna(
                        mode_value
                    )

                    cleaning_report.append(
                        f"{col}: filled {missing_before} missing "
                        f"values using mode ({mode_value})."
                    )

                # -------------------------------------------------
                # Numeric column: drop rows
                # -------------------------------------------------

                elif method == "drop_rows":

                    if missing_before == 0:

                        cleaning_report.append(
                            f"{col}: no missing values found; "
                            f"no rows were removed."
                        )

                        continue

                    rows_before = len(cleaned)

                    cleaned = cleaned.dropna(
                        subset=[col]
                    )

                    rows_removed = (
                        rows_before - len(cleaned)
                    )

                    cleaning_report.append(
                        f"{col}: removed {rows_removed} rows "
                        f"containing missing values."
                    )

                continue

            # =================================================
            # OUTLIER METHODS
            # =================================================

            if method in OUTLIER_METHODS:

                # -------------------------------------------------
                # Outliers only make sense for numeric columns
                # -------------------------------------------------

                if not is_numeric:

                    cleaning_report.append(
                        f"{col}: '{method}' skipped because "
                        f"the column is non-numeric."
                    )

                    continue

                # -------------------------------------------------
                # IMPORTANT:
                # Calculate bounds from ORIGINAL dataframe.
                #
                # This prevents one cleaning recommendation from
                # changing the IQR used by another recommendation.
                # -------------------------------------------------

                bounds = _iqr_bounds(df[col])

                if bounds is None:

                    cleaning_report.append(
                        f"{col}: outlier detection skipped because "
                        f"there was not enough numeric data."
                    )

                    continue

                lower, upper = bounds

                # Count outliers in current dataframe
                outlier_mask = (
                    (cleaned[col] < lower)
                    | (cleaned[col] > upper)
                )

                n_outliers = int(
                    outlier_mask.sum()
                )

                # -------------------------------------------------
                # DEFAULT:
                # Detect but do NOT modify outliers
                # -------------------------------------------------

                if not allow_outlier_changes:

                    cleaning_report.append(
                        f"{col}: detected {n_outliers} outliers "
                        f"using IQR; outliers were left unchanged."
                    )

                    continue

                # -------------------------------------------------
                # Clip outliers
                # -------------------------------------------------

                if method == "clip_outliers":

                    cleaned[col] = cleaned[col].clip(
                        lower=lower,
                        upper=upper
                    )

                    cleaning_report.append(
                        f"{col}: clipped {n_outliers} outliers "
                        f"to the IQR bounds "
                        f"({lower:.2f}, {upper:.2f})."
                    )

                # -------------------------------------------------
                # Remove outlier rows
                # -------------------------------------------------

                elif method == "remove_outliers":

                    rows_before = len(cleaned)

                    # Keep missing values.
                    #
                    # Missing values are a separate issue and
                    # should not accidentally be removed here.
                    keep_mask = (
                        cleaned[col].isna()
                        | cleaned[col].between(
                            lower,
                            upper
                        )
                    )

                    cleaned = cleaned[keep_mask]

                    rows_removed = (
                        rows_before - len(cleaned)
                    )

                    cleaning_report.append(
                        f"{col}: removed {rows_removed} "
                        f"outlier rows using IQR."
                    )

    # ---------------------------------------------------------
    # 4. Store cleaning report
    #
    # DataFrame.attrs does NOT create a new dataframe column.
    # ---------------------------------------------------------

    cleaned.attrs["cleaning_report"] = cleaning_report

    return cleaned