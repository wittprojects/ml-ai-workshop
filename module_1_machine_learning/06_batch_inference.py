# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Module 1: Machine Learning on Databricks
# MAGIC ## Notebook 06 — Batch Inference
# MAGIC
# MAGIC **Time**: ~5 min
# MAGIC
# MAGIC Two approaches to batch scoring:
# MAGIC 1. **Python**: `fe.score_batch()` with Feature Store lineage
# MAGIC

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.14.0 databricks-sdk>=0.50.0 lightgbm mlflow -q
# MAGIC dbutils.library.restartPython()

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
