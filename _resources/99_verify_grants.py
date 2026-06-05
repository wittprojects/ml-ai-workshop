# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Workshop GRANTS — Access Verification
# MAGIC
# MAGIC Run this as a **regular workspace user** (not the admin identity that ran
# MAGIC setup) to confirm the grants are correct. It exercises every asset the
# MAGIC workshop creates with the same operations a normal user would perform. If any
# MAGIC check fails, the corresponding grant is missing.

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow lightgbm==4.6.0 databricks-vectorsearch -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

failures: list[str] = []
passes: list[str] = []

def _short_error(e: BaseException) -> str:
    """Trim Spark's noisy tail (SQLSTATE, JVM stacktrace, query plan) from error messages."""
    msg = str(e)
    for sentinel in (" SQLSTATE:", "\nJVM stacktrace:", "\n'", "\n+- "):
        idx = msg.find(sentinel)
        if idx != -1:
            msg = msg[:idx]
    return msg.strip()

def check(label: str, fn):
    try:
        fn()
        passes.append(label)
        print(f"  ✓ {label}")
    except Exception as e:
        msg = f"{label}: {type(e).__name__}: {_short_error(e)}"
        failures.append(msg)
        print(f"  ✗ {msg}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Catalog + schema + foundation tables

# COMMAND ----------

check(f"USE CATALOG {catalog}",   lambda: spark.sql(f"USE CATALOG {catalog}").collect())
check(f"USE SCHEMA  {schema}",    lambda: spark.sql(f"USE SCHEMA {schema}").collect())

foundation_tables = [
    "customers", "service_tickets", "call_transcripts",
    "plans", "policies", "product_knowledge", "churn_labels",
]
for t in foundation_tables:
    check(
        f"SELECT on {catalog}.{schema}.{t}",
        lambda t=t: spark.sql(f"SELECT COUNT(*) FROM {catalog}.{schema}.{t}").collect(),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Documents volume

# COMMAND ----------

check(
    f"READ VOLUME on {documents_volume}",
    lambda: dbutils.fs.ls(documents_volume_path),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Module 1 ML assets

# COMMAND ----------

check(
    f"SELECT on feature_table_name ({feature_table_name})",
    lambda: spark.sql(f"SELECT COUNT(*) FROM {feature_table_name}").collect(),
)

check(
    f"EXECUTE on avg_price_increase",
    lambda: spark.sql(
        f"SELECT {catalog}.{schema}.avg_price_increase(CAST(50.0 AS DOUBLE), CAST(12 AS BIGINT))"
    ).collect(),
)

def _load_champion_model():
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    mlflow.sklearn.load_model(f"models:/{model_name}@Champion")

check(f"EXECUTE on MODEL {model_name}@Champion", _load_champion_model)

def _invoke_serving_endpoint():
    import requests
    sample = (
        spark.table(feature_table_name).limit(1).toPandas()
        .drop(columns=["customer_id", "update_timestamp"], errors="ignore")
    )
    sample["avg_price_increase"] = sample.apply(
        lambda r: (r["monthly_charges"] / r["tenure_months"]) if r["tenure_months"] else 0.0,
        axis=1,
    )
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    host = spark.conf.get("spark.databricks.workspaceUrl")
    r = requests.post(
        f"https://{host}/serving-endpoints/{churn_model_serving_endpoint}/invocations",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"dataframe_records": sample.to_dict(orient="records")},
        timeout=120,
    )
    r.raise_for_status()

check(f"CAN_QUERY on serving endpoint {churn_model_serving_endpoint}", _invoke_serving_endpoint)

check(
    f"SELECT on {predictions_table_name}",
    lambda: spark.sql(f"SELECT COUNT(*) FROM {predictions_table_name}").collect(),
)
check(
    f"SELECT on {catalog}.{schema}.churn_predictions_baseline",
    lambda: spark.sql(f"SELECT COUNT(*) FROM {catalog}.{schema}.churn_predictions_baseline").collect(),
)
for suffix in ("profile_metrics", "drift_metrics"):
    check(
        f"SELECT on {catalog}.{schema}.churn_predictions_{suffix}",
        lambda s=suffix: spark.sql(
            f"SELECT COUNT(*) FROM {catalog}.{schema}.churn_predictions_{s}"
        ).collect(),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vector Search

# COMMAND ----------

def _vs_query():
    idx = w.vector_search_indexes.get_index(index_name=product_knowledge_index)
    # Just resolving the index exercises CAN_USE on the endpoint. Running a similarity
    # search exercises the actual retrieval path.
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(endpoint_name=vector_search_endpoint, index_name=product_knowledge_index)
    index.similarity_search(query_text="slow internet", columns=["title", "content"], num_results=1)

check(f"CAN_USE on VS endpoint {vector_search_endpoint} (+ index {product_knowledge_index})", _vs_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Module 2 GenAI assets

# COMMAND ----------

check(
    f"SELECT on {catalog}.{schema}.parsed_documents",
    lambda: spark.sql(f"SELECT COUNT(*) FROM {catalog}.{schema}.parsed_documents").collect(),
)

agent_functions = [
    "get_latest_ticket",
    "get_customer_profile",
    "get_ticket_history",
    "get_retention_policy",
    "get_churn_risk",
]
# Sample customer for parameterized functions
sample_customer = spark.table(f"{catalog}.{schema}.customers").select("customer_id").limit(1).collect()[0][0]

check(
    f"EXECUTE on {catalog}.{schema}.get_latest_ticket()",
    lambda: spark.sql(f"SELECT * FROM {catalog}.{schema}.get_latest_ticket()").collect(),
)
check(
    f"EXECUTE on {catalog}.{schema}.get_customer_profile()",
    lambda: spark.sql(
        f"SELECT * FROM {catalog}.{schema}.get_customer_profile('{sample_customer}')"
    ).collect(),
)
check(
    f"EXECUTE on {catalog}.{schema}.get_ticket_history()",
    lambda: spark.sql(
        f"SELECT * FROM {catalog}.{schema}.get_ticket_history('{sample_customer}')"
    ).collect(),
)
check(
    f"EXECUTE on {catalog}.{schema}.get_retention_policy()",
    lambda: spark.sql(f"SELECT * FROM {catalog}.{schema}.get_retention_policy()").collect(),
)
check(
    f"EXECUTE on {catalog}.{schema}.get_churn_risk()",
    lambda: spark.sql(
        f"SELECT {catalog}.{schema}.get_churn_risk('{sample_customer}')"
    ).collect(),
)

# Metric views
for view in ("churn_customer_metrics", "churn_ticket_metrics", "tickets_with_customers"):
    check(
        f"SELECT on view {catalog}.{schema}.{view}",
        lambda v=view: spark.sql(f"SELECT * FROM {catalog}.{schema}.{v} LIMIT 1").collect(),
    )

# Genie space — list spaces visible to caller and look for the one setup created.
# w.genie.list_spaces() returns a GenieListSpacesResponse with `.spaces` and an
# optional `.next_page_token`. Paginate until we either find the space or exhaust.
def _genie_visible():
    title = "Telecom Churn Analytics"
    token = None
    while True:
        resp = w.genie.list_spaces(page_token=token) if token else w.genie.list_spaces()
        for s in (getattr(resp, "spaces", None) or []):
            if getattr(s, "title", None) == title:
                return
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    raise PermissionError(f"Genie space '{title}' not visible to this user")

check("CAN_RUN on Genie space 'Telecom Churn Analytics'", _genie_visible)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print(f"  Passed:  {len(passes)}")
print(f"  Failed:  {len(failures)}")
print("=" * 60)

if failures:
    print("\nMissing grants / failed checks:")
    for f in failures:
        print(f"  ✗ {f}")
    raise AssertionError(f"{len(failures)} verification check(s) failed — see output above")

print("\n✓ All workshop assets are accessible to this user.")