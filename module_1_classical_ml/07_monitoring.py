# Databricks notebook source
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
train_labels = spark.table("churn_labels").filter("split = 'train'").select("customer_id")
baseline = fe.score_batch(
    model_uri=f"models:/{model_name}@Champion",
    df=train_labels,
)

baseline_table = f"{catalog}.{schema}.churn_predictions_baseline"
baseline.write.mode("overwrite").saveAsTable(baseline_table)
print(f"✓ Baseline table created: {baseline_table} ({spark.table(baseline_table).count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Lakehouse Monitor

# COMMAND ----------

monitor_name = predictions_table

try:
    w.quality_monitors.create(
        table_name=monitor_name,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            label_col=None,  # Labels may not be available at inference time
            model_id_col=None,
            timestamp_col=None,
        ),
        baseline_table_name=baseline_table,
        output_schema_name=f"{catalog}.{schema}",
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
# MAGIC SELECT column_name, metric_name, metric_value
# MAGIC FROM ml_ai_workshop.workshop.churn_predictions_profile_metrics
# MAGIC WHERE metric_name IN ('count', 'mean', 'stddev', 'min', 'max')
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Module 1 Complete! 🎉
# MAGIC
# MAGIC We built an end-to-end ML pipeline:
# MAGIC
# MAGIC | Step | What We Did |
# MAGIC |------|-------------|
# MAGIC | 01 Overview | Explored the data and ML Runtime |
# MAGIC | 02 Feature Engineering | Created feature table in Unity Catalog |
# MAGIC | 03 Train Model | LightGBM + Optuna tuning + MLflow tracking |
# MAGIC | 04 Model Registry | Registered to UC with Champion alias |
# MAGIC | 05 Model Serving | Online tables + serving endpoint |
# MAGIC | 06 Batch Inference | `fe.score_batch()` + `ai_query()` |
# MAGIC | 07 Monitoring | Lakehouse Monitoring for drift detection |
# MAGIC
# MAGIC **The serving endpoint is now live** and will be used as a tool in Module 2's retention agent.
# MAGIC
# MAGIC **Next**: [Module 2: GenAI Development →](../module_2_genai/01_ai_functions)
