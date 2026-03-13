# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Classical ML on Databricks
# MAGIC ## Notebook 05 — Model Serving with Online Tables
# MAGIC
# MAGIC **Time**: ~8 min
# MAGIC
# MAGIC We'll deploy the Champion model to a **Model Serving** endpoint backed by **Lakebase Online Tables**.
# MAGIC
# MAGIC Key concepts:
# MAGIC - Lakebase-backed online tables for low-latency feature lookup
# MAGIC - Model Serving endpoints with auto-capture
# MAGIC - Real-time predictions via REST API

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AutoCaptureConfigInput,
)
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
import time

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Enable Change Data Feed on Feature Table
# MAGIC
# MAGIC Online tables require CDF to sync changes from the offline feature table.

# COMMAND ----------

spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"✓ CDF enabled on {feature_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Lakebase-Backed Online Table
# MAGIC
# MAGIC The online table provides millisecond-latency lookups for feature serving at inference time.

# COMMAND ----------

online_table_name = f"{feature_table_name}_online"

try:
    w.online_tables.create(
        table=OnlineTable(
            name=online_table_name,
            spec=OnlineTableSpec(
                source_table_full_name=feature_table_name,
                primary_key_columns=["customer_id"],
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
            ),
        )
    )
    print(f"✓ Creating online table '{online_table_name}'...")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Online table '{online_table_name}' already exists")
    else:
        raise e

# COMMAND ----------

# Wait for online table to be ready
online_table = w.online_tables.get(name=online_table_name)
while online_table.status.detailed_state.value not in ("ONLINE", "ACTIVE", "ONLINE_TRIGGERED_UPDATE", "ONLINE_NO_PENDING_UPDATE"):
    print(f"  Status: {online_table.status.detailed_state} — waiting 30s...")
    time.sleep(30)
    online_table = w.online_tables.get(name=online_table_name)
print(f"✓ Online table '{online_table_name}' is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Model Serving Endpoint

# COMMAND ----------

try:
    w.serving_endpoints.create_and_wait(
        name=churn_model_serving_endpoint,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=model_name,
                    entity_version=None,  # Uses Champion alias
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                )
            ],
            auto_capture_config=AutoCaptureConfigInput(
                catalog_name=catalog,
                schema_name=schema,
                table_name_prefix="churn_model",
                enabled=True,
            ),
        ),
    )
    print(f"✓ Serving endpoint '{churn_model_serving_endpoint}' created and ready")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Serving endpoint '{churn_model_serving_endpoint}' already exists")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test the Serving Endpoint

# COMMAND ----------

import requests
import json

# Get workspace URL and token
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Test with a sample customer
test_payload = {
    "dataframe_records": [
        {"customer_id": "CUST-00001"}
    ]
}

response = requests.post(
    f"https://{workspace_url}/serving-endpoints/{churn_model_serving_endpoint}/invocations",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=test_payload,
)

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Batch Test with Multiple Customers

# COMMAND ----------

# Test a few customers to see prediction distribution
test_customers = ["CUST-00001", "CUST-00050", "CUST-00100", "CUST-00500", "CUST-01000"]

batch_payload = {
    "dataframe_records": [{"customer_id": cid} for cid in test_customers]
}

response = requests.post(
    f"https://{workspace_url}/serving-endpoints/{churn_model_serving_endpoint}/invocations",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=batch_payload,
)

if response.status_code == 200:
    predictions = response.json()["predictions"]
    for cid, pred in zip(test_customers, predictions):
        print(f"  {cid}: {pred}")
else:
    print(f"Error: {response.status_code} — {response.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We deployed the churn model for real-time serving:
# MAGIC - **Lakebase Online Table** provides ms-latency feature lookup from the feature table
# MAGIC - **Model Serving Endpoint** serves predictions via REST API with auto-scaling
# MAGIC - **Auto-capture** logs all inference requests and responses for monitoring
# MAGIC
# MAGIC This endpoint will be used in Module 2 as the `get_churn_risk()` tool in our retention agent.
# MAGIC
# MAGIC **Next**: [06 Batch Inference →](./06_batch_inference)
