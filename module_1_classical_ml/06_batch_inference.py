# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Classical ML on Databricks
# MAGIC ## Notebook 06 — Batch Inference
# MAGIC
# MAGIC **Time**: ~5 min
# MAGIC
# MAGIC Two approaches to batch scoring:
# MAGIC 1. **Python**: `fe.score_batch()` with Feature Store lineage
# MAGIC 2. **SQL**: `ai_query()` calling the serving endpoint (bridges to Module 2)

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
import mlflow

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 1: `fe.score_batch()` (Python)
# MAGIC
# MAGIC This automatically looks up features from the feature table and applies the model — all with lineage tracking.

# COMMAND ----------

# Load test labels (customers we haven't scored yet)
test_labels = spark.table("churn_labels").filter("split = 'test'").select("customer_id")
print(f"Test customers to score: {test_labels.count()}")

# COMMAND ----------

# Score batch using the Champion model
batch_predictions = fe.score_batch(
    model_uri=f"models:/{model_name}@Champion",
    df=test_labels,
)

display(batch_predictions.limit(10))

# COMMAND ----------

# Save predictions for monitoring
(
    batch_predictions
    .write
    .mode("overwrite")
    .saveAsTable(f"{catalog}.{schema}.churn_predictions")
)

print(f"✓ Predictions saved to {catalog}.{schema}.churn_predictions")
print(f"  Total predictions: {spark.table(f'{catalog}.{schema}.churn_predictions').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 2: `ai_query()` (SQL)
# MAGIC
# MAGIC `ai_query()` calls the serving endpoint from SQL — great for dashboards and pipelines.
# MAGIC This bridges to Module 2 where we'll use AI Functions extensively.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Score customers using ai_query() against the serving endpoint
# MAGIC SELECT
# MAGIC   customer_id,
# MAGIC   ai_query(
# MAGIC     'workshop-churn-model',
# MAGIC     named_struct('customer_id', customer_id)
# MAGIC   ) as churn_prediction
# MAGIC FROM churn_labels
# MAGIC WHERE split = 'test'
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare Approaches
# MAGIC
# MAGIC | Aspect | `fe.score_batch()` | `ai_query()` |
# MAGIC |--------|-------------------|--------------|
# MAGIC | Language | Python | SQL |
# MAGIC | Feature lookup | Automatic (Feature Store) | Via serving endpoint |
# MAGIC | Lineage | Full lineage tracking | Endpoint-level tracking |
# MAGIC | Use case | ML pipelines, batch jobs | Dashboards, SQL pipelines |
# MAGIC | Latency | Batch (Spark) | Per-row (endpoint call) |
# MAGIC
# MAGIC **Next**: [07 Monitoring →](./07_monitoring)
