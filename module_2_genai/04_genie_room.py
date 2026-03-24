# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 04 — Metric Views & Genie Room
# MAGIC
# MAGIC **Time**: ~10 min
# MAGIC
# MAGIC **Metric Views** let you define business metrics once — as governed objects in Unity Catalog —
# MAGIC and query them flexibly across any dimension. A **Genie Room** powered by metric views gives
# MAGIC business users a natural-language interface to those same KPIs, with consistent answers guaranteed.
# MAGIC
# MAGIC In this notebook you will:
# MAGIC 1. Create metric views for customer churn KPIs and support ticket metrics
# MAGIC 2. Query them using the `MEASURE()` function
# MAGIC 3. Set up a Genie Room that lets anyone ask questions about churn in plain English
# MAGIC
# MAGIC | Notebook | Topic |
# MAGIC |----------|-------|
# MAGIC | 01 AI Functions | FMAPI, ai_query(), ai_extract() |
# MAGIC | 02 Create Tools | UC functions as agent tools |
# MAGIC | 03 Agent Eval | mlflow.genai.evaluate() |
# MAGIC | **04 Metric Views & Genie Room** | Governed metrics + NL analytics |
# MAGIC | 05 Build & Deploy Agent | AI Playground + Databricks Apps |

# COMMAND ----------

# MAGIC %pip install databricks-sdk>=0.74.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Customer Metric View
# MAGIC
# MAGIC A **metric view** separates *what* to measure from *how* to slice it.
# MAGIC - **Measures** define aggregations (churn rate, ARPU, etc.)
# MAGIC - **Dimensions** define how users can group and filter
# MAGIC
# MAGIC Unlike a regular view that locks in a `GROUP BY` at creation time, a metric view lets
# MAGIC each query choose its own dimensions — and the engine rewrites the SQL automatically.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.{schema}.churn_customer_metrics
WITH METRICS
LANGUAGE YAML
COMMENT 'Customer churn KPIs for telecom retention analysis'
AS $$
  version: 1.1
  source: {catalog}.{schema}.customers
  dimensions:
    - name: contract_type
      expr: contract_type
      comment: "Contract type: Month-to-month, One year, Two year"
    - name: internet_service
      expr: internet_service
      comment: "Internet service type: Fiber optic, DSL, No"
    - name: payment_method
      expr: payment_method
      comment: "Payment method used by the customer"
    - name: churn_status
      expr: churn
      comment: "Whether the customer churned: Yes or No"
    - name: tenure_bucket
      expr: >
        CASE
          WHEN tenure_months <= 12 THEN '0-12 months'
          WHEN tenure_months <= 24 THEN '13-24 months'
          WHEN tenure_months <= 48 THEN '25-48 months'
          ELSE '49+ months'
        END
      comment: "Customer tenure grouped into buckets"
    - name: senior_citizen
      expr: CASE WHEN senior_citizen = 1 THEN 'Yes' ELSE 'No' END
      comment: "Whether the customer is a senior citizen"
  measures:
    - name: total_customers
      expr: COUNT(*)
      comment: "Total number of customers"
    - name: churned_customers
      expr: COUNT(*) FILTER (WHERE churn = 'Yes')
      comment: "Number of customers who churned"
    - name: churn_rate
      expr: AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END)
      comment: "Percentage of customers who churned"
    - name: arpu
      expr: AVG(monthly_charges)
      comment: "Average Revenue Per User (monthly charges)"
    - name: avg_tenure
      expr: AVG(tenure_months)
      comment: "Average customer tenure in months"
    - name: avg_total_charges
      expr: AVG(total_charges)
      comment: "Average lifetime charges across customers"
$$
""")
print(f"✓ Created metric view: {catalog}.{schema}.churn_customer_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query with `MEASURE()`
# MAGIC
# MAGIC The `MEASURE()` function tells the engine to apply the aggregation defined in the metric view.
# MAGIC You pick the dimensions in `GROUP BY` — the metric view handles the rest.

# COMMAND ----------

display(spark.sql(f"""
SELECT
  contract_type,
  MEASURE(total_customers) AS total_customers,
  ROUND(MEASURE(churn_rate) * 100, 1) AS churn_rate_pct,
  ROUND(MEASURE(arpu), 2) AS arpu,
  ROUND(MEASURE(avg_tenure), 1) AS avg_tenure_months
FROM {catalog}.{schema}.churn_customer_metrics
GROUP BY ALL
ORDER BY churn_rate_pct DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **Notice**: month-to-month customers have the highest churn rate — consistent with the
# MAGIC patterns we saw in Module 1. The metric view gives us a single, governed definition
# MAGIC of "churn rate" that every downstream consumer shares.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Slice by a different dimension — same measures, no new code

# COMMAND ----------

display(spark.sql(f"""
SELECT
  tenure_bucket,
  churn_status,
  MEASURE(total_customers) AS customers,
  ROUND(MEASURE(arpu), 2) AS arpu
FROM {catalog}.{schema}.churn_customer_metrics
GROUP BY ALL
ORDER BY tenure_bucket, churn_status
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Ticket Metric View
# MAGIC
# MAGIC Service tickets live at a different grain than customers. We first create a join view,
# MAGIC then define a metric view on top of it.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.{schema}.tickets_with_customers AS
SELECT
  t.ticket_id,
  t.customer_id,
  t.created_date,
  t.category,
  t.priority,
  t.status,
  t.resolution_time_hours,
  c.contract_type,
  c.internet_service,
  c.churn
FROM {catalog}.{schema}.service_tickets t
JOIN {catalog}.{schema}.customers c ON t.customer_id = c.customer_id
""")
print(f"✓ Created join view: {catalog}.{schema}.tickets_with_customers")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.{schema}.churn_ticket_metrics
WITH METRICS
LANGUAGE YAML
COMMENT 'Service ticket KPIs for telecom support analysis'
AS $$
  version: 1.1
  source: {catalog}.{schema}.tickets_with_customers
  dimensions:
    - name: ticket_category
      expr: category
      comment: "Ticket category: billing, technical, cancellation, upgrade, general"
    - name: ticket_priority
      expr: priority
      comment: "Ticket priority: low, medium, high, critical"
    - name: ticket_status
      expr: status
      comment: "Current ticket status: open, resolved, escalated"
    - name: churn_status
      expr: churn
      comment: "Whether the customer churned: Yes or No"
    - name: contract_type
      expr: contract_type
      comment: "Customer contract type"
    - name: ticket_month
      expr: DATE_TRUNC('MONTH', created_date)
      comment: "Month the ticket was created"
  measures:
    - name: ticket_volume
      expr: COUNT(*)
      comment: "Total number of service tickets"
    - name: resolved_tickets
      expr: COUNT(*) FILTER (WHERE status = 'resolved')
      comment: "Number of resolved tickets"
    - name: resolution_rate
      expr: AVG(CASE WHEN status = 'resolved' THEN 1.0 ELSE 0.0 END)
      comment: "Percentage of tickets that were resolved"
    - name: avg_resolution_hours
      expr: AVG(resolution_time_hours)
      comment: "Average resolution time in hours"
    - name: escalation_rate
      expr: AVG(CASE WHEN status = 'escalated' THEN 1.0 ELSE 0.0 END)
      comment: "Percentage of tickets that were escalated"
$$
""")
print(f"✓ Created metric view: {catalog}.{schema}.churn_ticket_metrics")

# COMMAND ----------

display(spark.sql(f"""
SELECT
  ticket_category,
  churn_status,
  MEASURE(ticket_volume) AS tickets,
  ROUND(MEASURE(resolution_rate) * 100, 1) AS resolution_rate_pct,
  ROUND(MEASURE(escalation_rate) * 100, 1) AS escalation_rate_pct
FROM {catalog}.{schema}.churn_ticket_metrics
GROUP BY ALL
ORDER BY tickets DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Permissions

# COMMAND ----------

for view_name in ["churn_customer_metrics", "churn_ticket_metrics", "tickets_with_customers"]:
    spark.sql(f"GRANT SELECT ON VIEW {catalog}.{schema}.{view_name} TO `account users`")
    print(f"  ✓ Granted SELECT on {view_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Create a Genie Room
# MAGIC
# MAGIC A **Genie Room** gives business users a natural-language chat interface to your data.
# MAGIC When powered by metric views, every answer uses the same governed metric definitions —
# MAGIC no risk of "churn rate" meaning different things to different people.
# MAGIC
# MAGIC We'll use the **Genie Space API** (`WorkspaceClient.genie.create_space`) to create the
# MAGIC room programmatically — complete with data sources, sample questions, and instructions.

# COMMAND ----------

import json
import uuid
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pick a SQL warehouse
# MAGIC
# MAGIC Genie Rooms require a **Pro or Serverless SQL warehouse**. We'll find one automatically.

# COMMAND ----------

from databricks.sdk.service.sql import EndpointInfoWarehouseType

warehouses = [
    wh for wh in w.warehouses.list()
    if wh.warehouse_type in (EndpointInfoWarehouseType.PRO, EndpointInfoWarehouseType.TYPE_UNSPECIFIED)
]
if not warehouses:
    raise RuntimeError("No Pro or Serverless SQL warehouse found. Create one before running this notebook.")

warehouse_id = warehouses[0].id
print(f"Using warehouse: {warehouses[0].name} ({warehouse_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Genie Room configuration
# MAGIC
# MAGIC The `serialized_space` parameter is a JSON string that specifies:
# MAGIC - **Data sources** — our two metric views
# MAGIC - **Sample questions** — suggested prompts shown to users
# MAGIC - **Instructions** — rules for consistent formatting and domain context

# COMMAND ----------

def _hex_id():
    """Generate a 32-char lowercase hex ID for sample questions."""
    return uuid.uuid4().hex

space_config = {
    "version": 2,
    "config": {
        "sample_questions": [
            {"id": _hex_id(), "question": ["What is the churn rate by contract type?"]},
            {"id": _hex_id(), "question": ["Show me average revenue per user for churned vs retained customers"]},
            {"id": _hex_id(), "question": ["How does ticket volume trend by month for churners?"]},
            {"id": _hex_id(), "question": ["What is the ticket resolution rate by category and priority?"]},
            {"id": _hex_id(), "question": ["Which tenure bucket has the highest churn rate?"]},
            {"id": _hex_id(), "question": ["Compare ARPU across internet service types"]},
        ],
    },
    "instructions": {
        "text_instructions": [
            {
                "id": _hex_id(),
                "content": [
                    "Churn rate should always be displayed as a percentage. "
                    "ARPU stands for Average Revenue Per User (average monthly charges). "
                    "Churned customers have churn = 'Yes', retained have churn = 'No'. "
                    "When showing rates, round to one decimal place. "
                    "Tenure buckets are: 0-12 months, 13-24 months, 25-48 months, 49+ months. "
                    "Contract types are: Month-to-month, One year, Two year. "
                    "Ticket categories are: billing, technical, cancellation, upgrade, general."
                ],
            },
        ],
    },
    "data_sources": {
        "metric_views": [
            {
                "identifier": f"{catalog}.{schema}.churn_customer_metrics",
                "description": ["Customer churn KPIs: churn rate, ARPU, tenure, segmented by contract, internet service, and tenure bucket"],
            },
            {
                "identifier": f"{catalog}.{schema}.churn_ticket_metrics",
                "description": ["Support ticket KPIs: volume, resolution rate, escalation rate, segmented by category, priority, and month"],
            },
        ],
    },
}

print(json.dumps(space_config, indent=2)[:800] + "\n...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Genie Room

# COMMAND ----------

genie_space = w.genie.create_space(
    warehouse_id=warehouse_id,
    title="Telecom Churn Analytics",
    description="Ask questions about customer churn, revenue, tenure, and support ticket patterns. Powered by governed metric views.",
    serialized_space=json.dumps(space_config),
)

space_url = f"{w.config.host}/genie/rooms/{genie_space.space_id}"
print(f"✓ Genie Room created: {genie_space.title}")
print(f"  Space ID: {genie_space.space_id}")
print(f"  URL:      {space_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Genie Room
# MAGIC
# MAGIC Open the URL above, or try asking questions right here via the API:

# COMMAND ----------

conversation = w.genie.start_conversation(
    space_id=genie_space.space_id,
    content="What is the churn rate by contract type?",
)
print(f"Started conversation: {conversation.conversation_id}")
print("Open the Genie Room URL above to see the response and continue the conversation.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Try these questions in the Genie Room**:
# MAGIC - *"What is our overall churn rate?"*
# MAGIC - *"Break down churn rate by contract type and internet service"*
# MAGIC - *"Show me the monthly ticket trend for cancellation tickets"*
# MAGIC - *"Which customer segment has the highest ARPU?"*
# MAGIC
# MAGIC Genie generates SQL using `MEASURE()` against your metric views and returns
# MAGIC results as tables or charts — all governed by the definitions you created above.

# COMMAND ----------

# MAGIC %md
# MAGIC # Key Takeaways
# MAGIC
# MAGIC | Concept | What You Learned |
# MAGIC |---------|-----------------|
# MAGIC | **Metric Views** | Define metrics once in YAML, query flexibly with `MEASURE()` |
# MAGIC | **Dimensions vs Measures** | Dimensions slice data; measures aggregate it |
# MAGIC | **Genie Space API** | Create and configure Genie Rooms programmatically via the SDK |
# MAGIC | **Governance** | Metric views are Unity Catalog objects with standard permissions |
# MAGIC
# MAGIC **Why this matters**: Metric views ensure that "churn rate" means the same thing
# MAGIC whether queried by an analyst in SQL, a dashboard, or a business user in a Genie Room.
# MAGIC One definition eliminates metric drift across teams.

# COMMAND ----------

# --- Uncomment to clean up ---
# w.genie.trash_space(space_id=genie_space.space_id)
# print(f"Trashed Genie Room: {genie_space.space_id}")
# spark.sql(f"DROP VIEW IF EXISTS {catalog}.{schema}.churn_customer_metrics")
# spark.sql(f"DROP VIEW IF EXISTS {catalog}.{schema}.churn_ticket_metrics")
# spark.sql(f"DROP VIEW IF EXISTS {catalog}.{schema}.tickets_with_customers")
# print("Cleaned up metric views")
