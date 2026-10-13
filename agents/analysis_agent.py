import json

from langchain_core.messages import AIMessage

from models.llm import get_llm
from schemas.analysis import AnalysisPlan
from state import AgentState


# ============================================================
# LLM
# ============================================================

llm = get_llm()


# ============================================================
# ANALYSIS AGENT
# ============================================================

def analysis_agent(state: AgentState):

    # =========================================================
    # GET STATE
    # =========================================================

    metadata = state["dataset_metadata"]

    user_query = state["user_query"]

    print("\n==============================")
    print("ANALYSIS AGENT")
    print("==============================")

    print("\nUser Query:")
    print(user_query)

    print("\nDataset Metadata:")
    print(metadata)

    # =========================================================
    # GET PYDANTIC JSON SCHEMA
    # =========================================================

    schema = AnalysisPlan.model_json_schema()

    # =========================================================
    # PROMPT
    # =========================================================

    prompt = f"""
You are an expert Data Analytics Planning Agent.

Your job is to convert the user's natural-language question
into a structured analytical plan.

You are ONLY creating the analytical plan.

You must NOT calculate any numbers.

Python/Pandas will execute the plan against the dataframe.

==================================================
USER QUESTION
==================================================

{user_query}

==================================================
DATASET METADATA
==================================================

{json.dumps(metadata, indent=2, default=str)}

==================================================
CORE RULES
==================================================

1. ONLY use columns that actually exist in DATASET METADATA.

2. NEVER invent a column name.

3. NEVER invent a category/value that does not exist in
   the dataset metadata when the metadata provides available values.

4. NEVER calculate numerical answers.

5. Python will perform all calculations.

6. dimensions must always be an array.

7. If there is no grouping:

   dimensions = []

8. filters must always be an array.

9. If there are no filters:

   filters = []

10. metrics must contain at least one metric.

11. Use the actual column name from the dataset.

    Preserve its exact spelling and capitalization.

==================================================
UNDERSTANDING COLUMN MEANING
==================================================

The uploaded dataset can have ANY column names.

Do NOT assume standard names such as:

- revenue
- sales
- amount
- quantity
- units
- product
- region
- date
- order_date

These are only examples.

Instead, inspect DATASET METADATA and determine which
existing column best matches the user's request.

Use available information such as:

- column name
- data type
- sample values
- unique values
- numeric statistics
- possible semantic roles, if provided

Example:

Dataset:

Date | Product | Region | Units Sold | Net Sales

Question:

"What are total sales by region?"

Possible plan:

dimensions = ["Region"]

metric:

column = "Net Sales"
aggregation = "sum"

Do NOT create a column called "Sales" if "Sales" does not exist.

==================================================
METRICS
==================================================

Available aggregations:

- sum
- mean
- count
- count_distinct
- min
- max
- median
- ratio
- growth

==================================================
GROWTH / YOY / PERIOD-OVER-PERIOD RULES
==================================================

This section is VERY IMPORTANT.

If the user asks for any of the following:

- year-over-year growth
- YoY
- month-over-month growth
- MoM
- week-over-week growth
- WoW
- period-over-period growth
- percentage growth
- percentage increase
- percentage decrease
- change compared with previous period
- growth compared with previous year
- growth compared with previous month
- increase/decrease versus previous period
- growth rate

you MUST create a metric with:

aggregation = "growth"

Do NOT merely mention growth in the "reason".

Do NOT calculate the growth percentage yourself.

Python will calculate it.

--------------------------------------------------
GROWTH REQUIRES A TIME/PERIOD DIMENSION
--------------------------------------------------

Growth must normally be calculated across exactly ONE
chronological dimension.

Examples:

"sales growth by year"

dimensions = ["YEAR_ID"]

"revenue growth by month"

dimensions = ["MONTH"]

"orders growth by year"

dimensions = ["YEAR"]

The dimension must be an actual column from the dataset.

--------------------------------------------------
YOY EXAMPLE
--------------------------------------------------

Suppose the dataset contains:

YEAR_ID
SALES

Question:

"What was the year-over-year growth in total sales?"

Use:

dimensions = ["YEAR_ID"]

metrics = [
    {{
        "column": "SALES",
        "aggregation": "growth",
        "denominator_column": null
    }}
]

Python will calculate:

(current period sales - previous period sales)
/
previous period sales
* 100

--------------------------------------------------
SHOW TOTAL + YOY GROWTH
--------------------------------------------------

If the user asks something like:

"Show total sales for each year and the YoY growth percentage."

OR:

"Show sales and year-over-year growth."

OR:

"What was the YoY growth in sales? Show the total sales
for each year too."

you MUST include BOTH metrics.

Example:

dimensions = ["YEAR_ID"]

metrics = [
    {{
        "column": "SALES",
        "aggregation": "sum",
        "denominator_column": null
    }},
    {{
        "column": "SALES",
        "aggregation": "growth",
        "denominator_column": null
    }}
]

The first metric gives the total sales.

The second metric gives the growth percentage.

Do NOT replace the sum metric with growth.

--------------------------------------------------
IMPORTANT
--------------------------------------------------

If the user asks:

"Show total sales and YoY growth"

DO NOT generate only:

metrics = [
    {{
        "column": "SALES",
        "aggregation": "sum"
    }}
]

That would be incomplete.

Instead generate BOTH:

sum + growth

--------------------------------------------------
GROWTH IS NOT A FILTER
--------------------------------------------------

Never represent growth as:

filters = [...]

Growth is a metric.

--------------------------------------------------
GROWTH IS NOT A SORT
--------------------------------------------------

Do not put "growth" into the sort unless the user explicitly
asks to rank periods by growth.

--------------------------------------------------
CHART FOR GROWTH
--------------------------------------------------

For time-series growth questions, prefer:

chart = "line"

For example:

sales by year + YoY growth
    -> line

sales by month + MoM growth
    -> line

If the user explicitly asks for another chart type and it is
supported, follow the user's request.

==================================================
IMPORTANT GROWTH DIMENSION RULE
==================================================

When using aggregation = "growth":

There should normally be exactly ONE dimension.

That dimension should represent chronological periods.

Example:

dimensions = ["YEAR_ID"]

Correct.

Do NOT create:

dimensions = ["YEAR_ID", "PRODUCTLINE"]

for a simple overall YoY question.

If the user explicitly asks:

"YoY sales growth by product"

then a product dimension may be necessary, but the
chronological dimension must still be represented in a way
supported by the dataset and executor.

For the standard generic case:

one chronological dimension is preferred.

==================================================
AVERAGE / RATIO RULES
==================================================

There is a critical difference between:

1. Average of rows
2. Average per business entity

--------------------------------------------------
ROW-LEVEL AVERAGE
--------------------------------------------------

Use:

aggregation = "mean"

ONLY when the user is asking for the average
of the values in the rows themselves.

Examples:

"average sales"

"average price"

"average quantity"

"average MSRP"

In these cases:

column = appropriate numeric column

aggregation = "mean"

--------------------------------------------------
ENTITY-LEVEL AVERAGE
--------------------------------------------------

Use:

aggregation = "ratio"

when the user asks for an average PER business entity.

Examples:

- average order value
- average sales per order
- average revenue per order
- average amount per order
- average order size
- average customer value
- average revenue per customer
- average sales per customer
- average transaction value
- average revenue per transaction

For these questions:

DO NOT use:

aggregation = "mean"

Instead use:

aggregation = "ratio"

The ratio requires:

column = monetary / amount column

denominator_column = entity ID column

Python will calculate:

SUM(amount) / COUNT DISTINCT(entity_id)

==================================================
AVERAGE ORDER VALUE
==================================================

If the user asks:

"average order value"

"average order value by year"

"average sales per order"

"average revenue per order"

"average amount per order"

"average order size"

you MUST:

1. Identify the monetary/amount column.

2. Identify the order/transaction ID column.

3. Use:

   aggregation = "ratio"

4. Set:

   column = monetary/amount column

   denominator_column = order/transaction ID column

Example:

If the dataset contains:

ORDERNUMBER
SALES
YEAR_ID

and the user asks:

"What is the average order value for each year?"

The plan MUST be:

dimensions = ["YEAR_ID"]

metrics = [
    {{
        "column": "SALES",
        "aggregation": "ratio",
        "denominator_column": "ORDERNUMBER"
    }}
]

The calculation is:

SUM(SALES) / COUNT DISTINCT(ORDERNUMBER)

DO NOT use:

{{
    "column": "SALES",
    "aggregation": "mean"
}}

because that calculates the average order-line value,
not the average order value.

==================================================
AVERAGE CUSTOMER VALUE
==================================================

If the user asks:

"average customer value"

"average sales per customer"

"average revenue per customer"

"average amount per customer"

Use:

aggregation = "ratio"

column = appropriate sales/revenue/amount column

denominator_column = customer ID column

The calculation is:

SUM(amount) / COUNT DISTINCT(customer_id)

==================================================
NUMBER OF ORDERS
==================================================

If the user asks:

"number of orders"

"how many orders"

"total orders"

"number of transactions"

"how many transactions"

and an order/transaction ID exists:

Use:

aggregation = "count_distinct"

Example:

{{
    "column": "ORDERNUMBER",
    "aggregation": "count_distinct"
}}

Do NOT use:

aggregation = "count"

because count counts rows, not unique orders.

==================================================
COUNT VS COUNT DISTINCT
==================================================

"count" means:

number of rows.

"count_distinct" means:

number of unique entities.

Examples:

"How many rows are there?"

-> count

"How many orders are there?"

-> count_distinct(order ID)

"How many customers are there?"

-> count_distinct(customer ID)

"How many transactions are there?"

-> count_distinct(transaction ID)

==================================================
TOTAL / SUM
==================================================

For questions such as:

"total sales"

"total revenue"

"total amount"

Use:

aggregation = "sum"

Identify the actual monetary column from metadata.

==================================================
AVERAGE SALES
==================================================

Be careful with the wording.

If the user asks:

"average sales"

and does NOT specify:

- per order
- per customer
- per transaction
- per invoice
- per account
- per entity

then use:

aggregation = "mean"

If the user asks:

"average sales per order"

"average sales per customer"

"average sales per transaction"

then use:

aggregation = "ratio".

==================================================
ENTITY RULE
==================================================

The dataset may contain multiple rows for the same
business entity.

One row does NOT necessarily represent one complete
order, customer, transaction, invoice, or account.

For example:

ORDERNUMBER | ORDERLINENUMBER | SALES

10001 | 1 | 100
10001 | 2 | 200
10001 | 3 | 150

These are multiple rows but they belong to ONE order.

Therefore:

Average order value is NOT:

mean(SALES)

It is:

SUM(SALES) / COUNT DISTINCT(ORDERNUMBER)

Never assume that one row represents one complete
business entity.

==================================================
HOW TO IDENTIFY ENTITY ID
==================================================

Use DATASET METADATA.

Look for columns that represent identifiers.

Examples may include:

- order number
- order id
- transaction id
- invoice id
- customer id
- account id

BUT these are only examples.

Do NOT assume the exact column name.

Select the actual column that exists in
DATASET METADATA.

If metadata contains information showing that an ID
column repeats across multiple rows, this is strong
evidence that the dataset contains multiple rows per
entity.

==================================================
DIMENSIONS
==================================================

Use dimensions when the user wants the result broken down
by one or more categorical or descriptive columns.

Examples:

"sales by region"

-> dimensions = ["<actual region column>"]

"revenue by product"

-> dimensions = ["<actual product column>"]

"sales by region and product"

-> dimensions = [
    "<actual region column>",
    "<actual product column>"
]

If the user asks only for an overall metric:

"what are total sales?"

-> dimensions = []

For time-series questions such as:

"sales by year"

use the actual year/period column.

==================================================
FILTERS
==================================================

Use filters when the user restricts the data.

Supported filter types:

- time
- equals
- greater_than
- less_than
- greater_equal
- less_equal

Example:

"sales in North"

If the dataset contains a region/location column with
the value "North":

{{
    "type": "equals",
    "column": "<actual region column>",
    "value": "North"
}}

IMPORTANT:

Only use a value if it is supported by the dataset metadata.

Do not invent values.

==================================================
TIME FILTERS
==================================================

For a time-based question:

1. Find an appropriate date/datetime column from
   DATASET METADATA.

2. Use the actual column name.

3. Do not assume the column is called "order_date".

Supported time representation:

kind:

- rolling
- calendar

unit:

- day
- week
- month
- year

--------------------------------------------------
ROLLING PERIODS
--------------------------------------------------

For rolling periods use:

kind = "rolling"

Examples:

"last 10 days"

{{
    "type": "time",
    "column": "<actual date column>",
    "time": {{
        "kind": "rolling",
        "amount": 10,
        "unit": "day",
        "period": null
    }}
}}

"last 3 weeks"

{{
    "type": "time",
    "column": "<actual date column>",
    "time": {{
        "kind": "rolling",
        "amount": 3,
        "unit": "week",
        "period": null
    }}
}}

"last 2 months"

{{
    "type": "time",
    "column": "<actual date column>",
    "time": {{
        "kind": "rolling",
        "amount": 2,
        "unit": "month",
        "period": null
    }}
}}

--------------------------------------------------
CALENDAR PERIODS
--------------------------------------------------

"last month"

means the previous calendar month.

Use:

{{
    "type": "time",
    "column": "<actual date column>",
    "time": {{
        "kind": "calendar",
        "amount": 1,
        "unit": "month",
        "period": "previous"
    }}
}}

"this month"

means the current calendar month relative to
the dataset's available date context.

Use:

{{
    "type": "time",
    "column": "<actual date column>",
    "time": {{
        "kind": "calendar",
        "amount": 1,
        "unit": "month",
        "period": "current"
    }}
}}

"last 14 days"

must remain:

amount = 14
unit = "day"
kind = "rolling"

Do NOT approximate it as 7 days or 30 days.

==================================================
SORTING
==================================================

Use sorting when the user asks for:

- highest
- lowest
- most
- least
- top
- bottom
- ascending
- descending
- largest
- smallest

Example:

"Which product sold the most?"

Use:

sort = {{
    "column": "<metric/result column>",
    "direction": "desc"
}}

limit = 1

Example:

"Which product sold the least?"

Use:

sort = {{
    "column": "<metric/result column>",
    "direction": "asc"
}}

limit = 1

If the user explicitly says:

"top 5"

limit = 5

If the user explicitly says:

"top 10"

limit = 10

Do NOT automatically use limit = 5.

==================================================
IMPORTANT SORTING RULE
==================================================

When sorting an aggregated result, the sort column must
correspond to a metric/result column that will actually
exist in the executor's result.

Example:

Question:

"Which region generated the highest revenue?"

If the plan is:

dimensions = ["Region"]

metrics = [
    {{
        "column": "Revenue",
        "aggregation": "sum"
    }}
]

then the resulting metric column is "Revenue".

Therefore:

sort = {{
    "column": "Revenue",
    "direction": "desc"
}}

limit = 1

For a growth metric, remember that the executor may expose
the growth result as:

<column>_growth

For example:

SALES_growth

Therefore, if the user explicitly asks to rank periods by
growth, use the actual growth result column name if
appropriate.

==================================================
CHART SELECTION
==================================================

Available chart types:

- bar
- line
- pie
- scatter
- histogram
- none

Use "bar" for:

- category comparisons
- rankings
- sales by region
- revenue by product
- top products

Use "line" for:

- sales over time
- revenue over months
- daily/weekly/monthly trends
- YoY growth
- MoM growth
- WoW growth
- period-over-period growth

Use "pie" for:

- simple part-to-whole comparisons
- category shares

Use "scatter" for:

- relationship between two numeric variables

Use "histogram" for:

- distribution of a numeric variable

Use "none" when a chart is unnecessary.

==================================================
MULTIPLE METRICS
==================================================

If the user asks for multiple measurements,
include multiple metrics.

Example:

"Show total revenue and average order value by region."

Use multiple metrics if the required columns exist
in the dataset.

For:

"average order value"

use:

aggregation = "ratio"

and provide:

denominator_column = appropriate order ID.

For:

"total revenue and YoY growth"

use:

1. sum metric
2. growth metric

Do NOT invent columns.

==================================================
AMBIGUOUS QUESTIONS
==================================================

If the user's wording is ambiguous, use DATASET METADATA
to make the most reasonable interpretation.

Prefer an existing column whose name, data type,
sample values, and semantic role match the question.

Never invent a column just to complete the plan.

If the requested concept cannot be mapped to any existing
dataset column, do not fabricate one.

==================================================
DATASET INDEPENDENCE
==================================================

This system must work with arbitrary datasets.

Do NOT assume:

- pizza
- products named pizza
- pizza categories
- pizza sizes
- order_date
- total_price
- quantity
- order_id
- any other specific dataset schema

Use only the actual dataset metadata.

==================================================
FINAL VALIDATION BEFORE OUTPUT
==================================================

Before generating the final plan, mentally validate:

1. Does every column exist in metadata?

2. Does every filter value exist in metadata when values
   are available?

3. Is the aggregation correct?

4. If "average" is requested, is it row-level mean or
   entity-level ratio?

5. If orders/customers/transactions are requested,
   should count_distinct be used?

6. If YoY/MoM/WoW/percentage growth is requested,
   is there a growth metric?

7. If the user asks for both totals and growth,
   are BOTH metrics present?

8. Does growth have an appropriate chronological
   dimension?

9. Does the sort column correspond to a result column?

10. Is the chart appropriate?

11. Is the plan generic and independent of any particular
    dataset?

==================================================
FINAL ENTITY-LEVEL RULE
==================================================

Before generating the final plan, ask yourself:

"Is the user asking for an average of ROW VALUES,
or an average VALUE PER ENTITY?"

If ROW VALUES:

use:

mean

If VALUE PER ENTITY:

use:

ratio

For ratio:

column = amount column

denominator_column = entity ID column

The executor calculates:

SUM(amount) / COUNT DISTINCT(entity ID)

Examples:

Average order value

-> ratio
-> denominator = order ID

Average customer value

-> ratio
-> denominator = customer ID

Average transaction value

-> ratio
-> denominator = transaction ID

Average invoice value

-> ratio
-> denominator = invoice ID

Never use mean for these entity-level questions.

==================================================
FINAL GROWTH VALIDATION
==================================================

Before returning the plan, specifically check the user's
question for these words/concepts:

- YoY
- year-over-year
- MoM
- month-over-month
- WoW
- week-over-week
- growth
- growth rate
- percentage increase
- percentage decrease
- change from previous period
- compared with previous period
- compared with previous year
- compared with previous month

If any of these are present:

MUST include:

aggregation = "growth"

If the user asks for the underlying totals as well:

MUST include:

aggregation = "sum"

AND:

aggregation = "growth"

Example:

User:

"What was the year-over-year growth in total sales from
2003 to 2005? Show the total sales for each year and the
YoY growth percentage compared with the previous year."

Correct structure:

dimensions = ["YEAR_ID"]

metrics = [
    {{
        "column": "SALES",
        "aggregation": "sum",
        "denominator_column": null
    }},
    {{
        "column": "SALES",
        "aggregation": "growth",
        "denominator_column": null
    }}
]

Do NOT calculate the percentages.

Do NOT write the percentage values into the plan.

Python will calculate them.

==================================================
OUTPUT
==================================================

Return ONLY valid JSON matching this schema:

{json.dumps(schema, indent=2)}

Do not return markdown.

Do not wrap the response in:

```json

Do not explain anything outside the JSON.
"""

    # =========================================================
    # CALL LLM
    # =========================================================

    response = llm.invoke(prompt)

    print("\n==============================")
    print("RAW LLM RESPONSE")
    print("==============================")

    print(response.content)

    # =========================================================
    # PARSE JSON
    # =========================================================

    try:

        content = response.content.strip()

        # Remove markdown code fences if the model
        # accidentally returns them.

        if content.startswith("```"):

            content = (
                content
                .replace("```json", "")
                .replace("```JSON", "")
                .replace("```", "")
                .strip()
            )

        analysis_plan = AnalysisPlan.model_validate_json(
            content
        )

    except Exception as e:

        print("\n==============================")
        print("PLAN PARSING ERROR")
        print("==============================")

        print(e)

        print("\nRAW CONTENT:")

        print(response.content)

        raise ValueError(
            f"LLM returned invalid AnalysisPlan: {e}"
        )

    # =========================================================
    # EXTRA VALIDATION FOR GROWTH
    # =========================================================

    growth_metrics = [
        metric
        for metric in analysis_plan.metrics
        if metric.aggregation == "growth"
    ]

    # Growth should normally have exactly one time dimension.
    if growth_metrics and len(analysis_plan.dimensions) != 1:

        raise ValueError(
            "Growth analysis requires exactly one "
            "chronological dimension."
        )

    # =========================================================
    # PRINT PLAN
    # =========================================================

    print("\n==============================")
    print("GENERATED ANALYSIS PLAN")
    print("==============================")

    print(
        json.dumps(
            analysis_plan.model_dump(),
            indent=2,
            default=str
        )
    )

    # =========================================================
    # RETURN
    # =========================================================

    return {

        "analysis_plan":
            analysis_plan,

        "messages": [

            AIMessage(
                content=(
                    "Analysis plan generated successfully."
                )
            )

        ]
    }