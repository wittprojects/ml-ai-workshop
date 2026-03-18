# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Module 1: Classical ML on Databricks
# MAGIC ## Notebook 07 — Lakehouse Monitoring
# MAGIC
# MAGIC **Time**: ~5 min
# MAGIC
# MAGIC Set up **Lakehouse Monitoring** to track model drift and data quality over time.
# MAGIC
# MAGIC Key concepts:
# MAGIC - Baseline vs. scoring table comparison
# MAGIC - Profile metrics (statistics) and drift metrics
# MAGIC - Alerting on drift

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.14.0 databricks-sdk>=0.50.0 mlflow -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Prepare the Inference Table
# MAGIC
# MAGIC The predictions we saved in the previous notebook serve as our monitoring target.

# COMMAND ----------

predictions_table = f"{catalog}.{schema}.churn_predictions"
display(spark.table(predictions_table).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create a Baseline Table
# MAGIC
# MAGIC We'll use the training data distribution as our baseline for drift detection.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# Score training set to create baseline
# train_labels = spark.table("churn_labels").filter("split = 'train'").select("customer_id")
# baseline = fe.score_batch(
#     model_uri=f"models:/{model_name}@Champion",
#     df=train_labels,
# )

baseline_table = f"{catalog}.{schema}.churn_predictions"
# baseline.write.mode("overwrite").saveAsTable(baseline_table)
# print(f"✓ Baseline table created: {baseline_table} ({spark.table(baseline_table).count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Lakehouse Monitor

# COMMAND ----------

monitor_name = predictions_table
username = spark.sql("SELECT current_user()").first()[0]

# Add required columns if they don't exist
existing_cols = spark.table(predictions_table).columns
if "inference_timestamp" not in existing_cols:
    spark.sql(f"ALTER TABLE {predictions_table} ADD COLUMNS (inference_timestamp TIMESTAMP)")
    spark.sql(f"UPDATE {predictions_table} SET inference_timestamp = current_timestamp()")
    print("Added inference_timestamp column")
if "model_id" not in existing_cols:
    spark.sql(f"ALTER TABLE {predictions_table} ADD COLUMNS (model_id STRING)")
    spark.sql(f"UPDATE {predictions_table} SET model_id = '{model_name}'")
    print("Added model_id column")

try:
    w.quality_monitors.create(
        table_name=monitor_name,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            label_col=None,
            model_id_col="model_id",
            timestamp_col="inference_timestamp",
            granularities=["1 day"],
        ),
        baseline_table_name=baseline_table,
        output_schema_name=f"{catalog}.{schema}",
        assets_dir=f"/Workspace/Users/{username}/databricks_lakehouse_monitoring/{predictions_table}",
    )
    print(f"✓ Monitor created for {monitor_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Monitor already exists for {monitor_name}")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Refresh the Monitor
# MAGIC
# MAGIC The first refresh computes profile and drift metrics.

# COMMAND ----------

try:
    w.quality_monitors.run_refresh(table_name=monitor_name)
    print("✓ Monitor refresh triggered")
except Exception as e:
    print(f"Note: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Explore Monitor Results
# MAGIC
# MAGIC After the refresh completes, two tables are created:
# MAGIC - `{table}_profile_metrics` — statistics for each column
# MAGIC - `{table}_drift_metrics` — drift scores comparing baseline vs. scoring

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Profile metrics (may take a minute to populate after first refresh)
# MAGIC SELECT column_name, median, percent_nan, min
# MAGIC FROM wittprojects.workshop.churn_predictions_profile_metrics
# MAGIC -- WHERE metric_name IN ('count', 'mean', 'stddev', 'min', 'max')
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Drift metrics
# MAGIC SELECT column_name, metric_name, metric_value
# MAGIC FROM ml_ai_workshop.workshop.churn_predictions_drift_metrics
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Alerting (Overview)
# MAGIC
# MAGIC In production, you'd set up alerts on drift metrics:
# MAGIC
# MAGIC ```python
# MAGIC # Example: Alert if prediction distribution shifts significantly
# MAGIC # This would be configured via Databricks SQL Alerts or Workflows
# MAGIC #
# MAGIC # SELECT metric_value
# MAGIC # FROM {monitor}_drift_metrics
# MAGIC # WHERE column_name = 'prediction'
# MAGIC #   AND metric_name = 'chi_squared_test'
# MAGIC #   AND metric_value > 0.05  -- p-value threshold
# MAGIC ```
# MAGIC
# MAGIC Lakehouse Monitoring integrates with:
# MAGIC - **Databricks SQL Alerts** — automated threshold-based notifications
# MAGIC - **Workflows** — trigger retraining pipelines on drift detection
# MAGIC - **Dashboards** — visual monitoring in Databricks SQL
