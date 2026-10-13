import pandas as pd


def detect_issues(df: pd.DataFrame) -> list[dict]:
    """
    Detect common data quality issues in a dataframe.
    """

    issues = []

    # -------------------------
    # Missing Values
    # -------------------------
    missing = df.isnull().sum()

    for column, count in missing.items():

        if count > 0:
            issues.append(
                {
                    "column": column,
                    "issue": "missing_values",
                    "count": int(count),
                    "dtype": str(df[column].dtype),
                }
            )

    # -------------------------
    # Duplicate Rows
    # -------------------------
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        issues.append(
            {
                "column": "Entire Dataset",
                "issue": "duplicate_rows",
                "count": duplicate_count,
            }
        )

    # -------------------------
    # Outlier Detection (IQR)
    # -------------------------
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        outlier_count = ((series < lower) | (series > upper)).sum()

        if outlier_count > 0:
            issues.append(
                {
                    "column": column,
                    "issue": "outliers",
                    "count": int(outlier_count),
                    "dtype": str(series.dtype),
                }
            )

    return issues