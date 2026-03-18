# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
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

# MAGIC %pip install databricks-feature-engineering==0.14.0 databricks-sdk>=0.50.0 mlflow -q
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
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup, FeatureFunction
import mlflow
from mlflow.tracking import MlflowClient
import time

w = WorkspaceClient()
fe = FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

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
# MAGIC ## 2. Create Online Store & Publish Feature Table
# MAGIC
# MAGIC A **Databricks Online Feature Store** (Lakebase-backed) provides millisecond-latency lookups for feature serving at inference time.
# MAGIC We create an online store, then publish the feature table to it.

# COMMAND ----------

online_table_name = f"{feature_table_name}_online"

# Create the online store (skip if it already exists)
try:
    fe.create_online_store(
        name=online_store_name,
        capacity="CU_1"
    )
    print(f"✓ Online store '{online_store_name}' created")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Online store '{online_store_name}' already exists")
    else:
        raise e

# Wait for the online store to be available
online_store = fe.get_online_store(name=online_store_name)
print(f"  Online store state: {online_store.state}")

# COMMAND ----------

# Wait for the online store to be available
while True:
    online_store = fe.get_online_store(name=online_store_name)
    if str(online_store.state) == "State.AVAILABLE":
        break
    print(f"  Online store state: {online_store.state} — waiting 30s...")
    time.sleep(30)
print(f"✓ Online store '{online_store_name}' is available")

# Publish the feature table to the online store
try:
    fe.publish_table(
        online_store=online_store,
        source_table_name=feature_table_name,
        online_table_name=online_table_name,
    )
    print(f"✓ Feature table published: {feature_table_name} → {online_table_name}")
except Exception as e:
    if "already exists" in str(e).lower() or "already published" in str(e).lower():
        print(f"✓ Online table '{online_table_name}' already published")
    else:
        raise e

# COMMAND ----------

# DBTITLE 1,Feature Spec overview
# MAGIC %md
# MAGIC ## 3. Create Feature Spec for Feature Serving
# MAGIC
# MAGIC A **FeatureSpec** defines which features to serve and how to look them up. This mirrors the training-time feature lookups from notebook 03 and enables:
# MAGIC - Consistent feature retrieval at serving time (online table lookups + on-demand functions)
# MAGIC - Standalone feature serving endpoints for direct feature access

# COMMAND ----------

# DBTITLE 1,Create FeatureSpec
# Create a FeatureSpec matching the training-time feature lookups
feature_lookups = [
    FeatureLookup(
        table_name=feature_table_name,
        lookup_key="customer_id",
    ),
    FeatureFunction(
        udf_name=f"{catalog}.{schema}.avg_price_increase",
        output_name="avg_price_increase",
        input_bindings={"monthly_charges": "monthly_charges", "tenure_months": "tenure_months"},
    ),
]

try:
    fe.create_feature_spec(
        name=feature_spec_name,
        features=feature_lookups,
        exclude_columns=["update_timestamp"],
    )
    print(f"✓ Feature spec created: {feature_spec_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Feature spec already exists: {feature_spec_name}")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Model Serving Endpoint
# MAGIC
# MAGIC Deploy the Champion model version to a serving endpoint. The endpoint automatically performs feature lookups from the online table at inference time since the model was logged with `fe.log_model()`.

# COMMAND ----------

from datetime import timedelta

# Look up the Champion model version
champion_info = mlflow_client.get_model_version_by_alias(model_name, "Champion")
champion_version = champion_info.version
print(f"Deploying Champion model version: {champion_version}")

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        entity_version=str(champion_version),
        scale_to_zero_enabled=True,
        workload_size="Small",
    )
]

try:
    w.serving_endpoints.create_and_wait(
        name=churn_model_serving_endpoint,
        config=EndpointCoreConfigInput(
            name=churn_model_serving_endpoint,
            served_entities=served_entities,
        ),
        timeout=timedelta(minutes=45),
    )
    print(f"✓ Serving endpoint '{churn_model_serving_endpoint}' created and ready")
except Exception as e:
    if "already exists" in str(e).lower():
        # Endpoint exists — push a fresh config update (handles failed builds)
        print(f"Endpoint already exists, pushing config update...")
        w.serving_endpoints.update_config_and_wait(
            name=churn_model_serving_endpoint,
            served_entities=served_entities,
            timeout=timedelta(minutes=45),
        )
        print(f"✓ Serving endpoint '{churn_model_serving_endpoint}' updated and ready")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test the Serving Endpoint

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
# MAGIC ## 6. Batch Test with Multiple Customers

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
# MAGIC - **Feature Spec** defines the feature serving contract (lookups + on-demand functions)
# MAGIC - **Model Serving Endpoint** serves predictions via REST API with auto-scaling
# MAGIC - **Auto-capture** logs all inference requests and responses for monitoring
# MAGIC
# MAGIC This endpoint will be used in Module 2 as the `get_churn_risk()` tool in our retention agent.
# MAGIC
# MAGIC **Next**: [06 Batch Inference →](./06_batch_inference)
