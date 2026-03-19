# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Module 1: Machine Learning on Databricks
# MAGIC ## Notebook 06 — Lakehouse Monitoring
# MAGIC
# MAGIC **Time**: ~15 min
# MAGIC
# MAGIC Build **production-grade model monitoring** using Databricks Lakehouse Monitoring. This capstone ties together everything we've built — scoring the model, detecting drift, tracking model quality, and setting up alerting.
# MAGIC
# MAGIC ### What we'll cover
# MAGIC | Concept | What it does |
# MAGIC |---------|-------------|
# MAGIC | **Baseline comparison** | Compare production predictions against training-distribution baseline |
# MAGIC | **Drift detection** | KS, chi-squared, PSI, Wasserstein, Jensen-Shannon, total variation |
# MAGIC | **Model quality** | Classification metrics when ground-truth labels arrive |
# MAGIC | **Custom metrics** | Business KPIs via Jinja SQL templates |
# MAGIC | **Alerting** | SQL-based threshold alerts on drift metrics |
# MAGIC | **Dashboard** | Auto-generated Lakeview dashboard |

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering==0.14.0 databricks-sdk>=0.50.0 mlflow -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorMetric,
    MonitorMetricType,
)
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F
import mlflow
from mlflow.tracking import MlflowClient
import time

w = WorkspaceClient()
fe = FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

predictions_table = f"{catalog}.{schema}.churn_predictions"
spark.sql(f"ALTER TABLE {predictions_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

baseline_table = f"{catalog}.{schema}.churn_predictions_baseline"

# Model ID for monitoring must match the serving endpoint's model identifier
# Format: <model_name> version <version> (matches inference table convention)
champion_version = mlflow_client.get_model_version_by_alias(model_name, "Champion").version
monitor_model_id = f"{model_name} version {champion_version}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Baseline Table
# MAGIC
# MAGIC A monitor needs a **baseline** — the distribution the model was trained on. If we compare a table against itself, drift is always zero.
# MAGIC
# MAGIC We score the **training split** with the Champion model to capture the training-time distribution. This becomes our reference point.

# COMMAND ----------

# Score training split to create baseline
train_labels = spark.table("churn_labels").filter("split = 'train'").select("customer_id")

baseline_df = fe.score_batch(
    model_uri=f"models:/{model_name}@Champion",
    df=train_labels,
)

# Add monitoring columns — fixed timestamp for the baseline window
baseline_df = (
    baseline_df
    .withColumn("inference_timestamp", F.lit("2025-06-01").cast("timestamp"))
    .withColumn("model_id", F.lit(monitor_model_id))
)

baseline_df.write.mode("overwrite").saveAsTable(baseline_table)
print(f"Baseline table created: {baseline_table} ({spark.table(baseline_table).count()} rows)")

# COMMAND ----------

# Baseline distribution summary
display(
    spark.table(baseline_table)
    .groupBy("prediction")
    .agg(
        F.count("*").alias("count"),
        F.round(F.avg("monthly_charges"), 2).alias("avg_monthly_charges"),
        F.round(F.avg("tenure_months"), 1).alias("avg_tenure"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare Inference Table
# MAGIC
# MAGIC The predictions saved in notebook 05 (test split) represent our "current" production scoring window.

# COMMAND ----------

# Add monitoring columns to predictions table if not present
existing_cols = [c.lower() for c in spark.table(predictions_table).columns]

if "inference_timestamp" not in existing_cols:
    spark.sql(f"ALTER TABLE {predictions_table} ADD COLUMNS (inference_timestamp TIMESTAMP)")

if "model_id" not in existing_cols:
    spark.sql(f"ALTER TABLE {predictions_table} ADD COLUMNS (model_id STRING)")

spark.sql(f"""
    UPDATE {predictions_table}
    SET inference_timestamp = CAST('2025-07-01' AS TIMESTAMP),
        model_id = '{monitor_model_id}'
    WHERE inference_timestamp IS NULL
""")

print(f"Inference table ready: {spark.table(predictions_table).count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Simulate Drift
# MAGIC
# MAGIC In production, drift happens naturally — pricing changes, customer behavior shifts, seasonal patterns.
# MAGIC Here we **programmatically shift features** to demonstrate detection:
# MAGIC
# MAGIC | Feature | Shift | Simulated cause |
# MAGIC |---------|-------|-----------------|
# MAGIC | `monthly_charges` | +15% | Pricing increase |
# MAGIC | `total_charges` | +20% | Cumulative effect |
# MAGIC | `tenure_months` | +6 | Customer base aging |
# MAGIC | `contract_type` | 70% month-to-month | Market shift |

# COMMAND ----------

# Build drifted dataset from current predictions
current_df = spark.table(predictions_table).filter("inference_timestamp = '2025-07-01'")

drifted_df = (
    current_df
    .withColumn("monthly_charges", F.round(F.col("monthly_charges") * 1.15, 2))
    .withColumn("total_charges", F.round(F.col("total_charges") * 1.20, 2))
    .withColumn("tenure_months", F.col("tenure_months") + 6)
    .withColumn(
        "contract_type",
        F.when(F.rand() < 0.70, F.lit("Month-to-month")).otherwise(F.col("contract_type"))
    )
    .withColumn("inference_timestamp", F.lit("2025-08-01").cast("timestamp"))
)

# Idempotent: remove any existing drifted rows before appending
spark.sql(f"DELETE FROM {predictions_table} WHERE inference_timestamp = CAST('2025-08-01' AS TIMESTAMP)")
drifted_df.write.mode("append").saveAsTable(predictions_table)

print(f"Drifted rows appended: {drifted_df.count()}")
print(f"Total rows in predictions table: {spark.table(predictions_table).count()}")

# Verify we have multiple time windows
display(spark.sql(f"SELECT inference_timestamp, COUNT(*) as row_count FROM {predictions_table} GROUP BY 1 ORDER BY 1"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Lakehouse Monitor
# MAGIC
# MAGIC We create an **InferenceLog** monitor that:
# MAGIC - Compares each time window against the baseline
# MAGIC - Computes per-column statistics and drift tests
# MAGIC - Tracks a custom business metric

# COMMAND ----------

# Delete existing monitor for idempotency
try:
    w.quality_monitors.delete(table_name=predictions_table)
    print(f"Deleted existing monitor on {predictions_table}")
    time.sleep(5)
except Exception:
    print("No existing monitor to delete")

# COMMAND ----------

# Define custom metric: percentage of high-risk (churn=1) predictions
high_risk_metric = MonitorMetric(
    type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
    name="high_risk_percentage",
    input_columns=[":table"],
    definition="100.0 * SUM(CASE WHEN {{prediction_col}} = 1 THEN 1 ELSE 0 END) / COUNT(*)",
    output_data_type="DOUBLE",
)

# COMMAND ----------

# Create the monitor
username = spark.sql("SELECT current_user()").first()[0]

monitor_info = w.quality_monitors.create(
    table_name=predictions_table,
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        label_col=None,
        model_id_col="model_id",
        timestamp_col="inference_timestamp",
        granularities=["1 month"],
    ),
    baseline_table_name=baseline_table,
    custom_metrics=[high_risk_metric],
    output_schema_name=f"{catalog}.{schema}",
    assets_dir=f"/Workspace/Users/{username}/databricks_lakehouse_monitoring/{predictions_table}",
)

print(f"Monitor created for {predictions_table}")
print(f"Dashboard: {monitor_info.assets_dir}")

# COMMAND ----------

# Trigger refresh and wait for completion
w.quality_monitors.run_refresh(table_name=predictions_table)
print("Monitor refresh triggered — this takes 1-3 minutes...")

while True:
    refreshes_response = w.quality_monitors.list_refreshes(table_name=predictions_table)
    active = [r for r in refreshes_response.refreshes if r.state in ("PENDING", "RUNNING")]
    if not active:
        latest = sorted(refreshes_response.refreshes, key=lambda r: r.start_time_ms or 0, reverse=True)[0]
        print(f"Refresh complete — state: {latest.state}")
        if latest.state == "SUCCEEDED":
            print("Monitor refresh succeeded.")
            break
        elif latest.state in ("FAILED", "CANCELLED"):
            print(f"Monitor refresh ended with state: {latest.state}")
            break
    print(f"  Still running... ({time.strftime('%H:%M:%S')})")
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analyze Profile Metrics
# MAGIC
# MAGIC The monitor creates a `_profile_metrics` table with **per-column statistics for each time window** — count, mean, stddev, min, max, quantiles, percent nulls, and more.

# COMMAND ----------

# Check what windows were computed
display(spark.sql(f"""
    SELECT DISTINCT window, granularity, model_id
    FROM {catalog}.{schema}.churn_predictions_profile_metrics
    ORDER BY window
"""))

# COMMAND ----------

display(spark.sql(f"""
    SELECT window, column_name,
           count, mean, stddev, min, max,
           percent_zeros, percent_nulls
    FROM {catalog}.{schema}.churn_predictions_profile_metrics
    WHERE column_name IN ('monthly_charges', 'tenure_months', 'contract_type', 'prediction')
      AND slice_key IS NULL
      AND window IS NOT NULL
    ORDER BY column_name, window
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Analyze Drift Metrics
# MAGIC
# MAGIC The `_drift_metrics` table compares each window against the baseline using multiple statistical tests:
# MAGIC
# MAGIC | Test | Type | What it measures |
# MAGIC |------|------|-----------------|
# MAGIC | **KS test** | Numerical | Max CDF distance — are distributions shaped differently? |
# MAGIC | **Chi-squared** | Categorical | Frequency differences — has the category mix changed? |
# MAGIC | **PSI** | Both | Population Stability Index — overall distribution shift |
# MAGIC | **Wasserstein** | Numerical | Earth mover's distance — how much "work" to transform one distribution into another |
# MAGIC | **Jensen-Shannon** | Both | Symmetric KL divergence — information-theoretic distance |
# MAGIC | **Total variation** | Both | Max absolute probability difference across bins |

# COMMAND ----------

# Numerical feature drift
display(spark.sql(f"""
    SELECT window, column_name,
           wasserstein_distance,
           ks_test.statistic AS ks_statistic,
           ks_test.pvalue AS ks_pvalue,
           population_stability_index AS psi
    FROM {catalog}.{schema}.churn_predictions_drift_metrics
    WHERE column_name IN ('monthly_charges', 'total_charges', 'tenure_months')
      AND slice_key IS NULL
    ORDER BY column_name, window
"""))

# COMMAND ----------

# Categorical feature drift
display(spark.sql(f"""
    SELECT window, column_name,
           chi_squared_test.statistic AS chi2_statistic,
           chi_squared_test.pvalue AS chi2_pvalue,
           js_distance,
           tv_distance
    FROM {catalog}.{schema}.churn_predictions_drift_metrics
    WHERE column_name IN ('contract_type', 'internet_service', 'payment_method', 'prediction')
      AND slice_key IS NULL
    ORDER BY column_name, window
"""))

# COMMAND ----------

# Drift summary: which features drifted most?
display(spark.sql(f"""
    SELECT column_name, window,
           COALESCE(population_stability_index, 0) AS psi,
           COALESCE(wasserstein_distance, 0) AS wasserstein,
           COALESCE(js_distance, 0) AS js_distance,
           COALESCE(ks_test.pvalue, chi_squared_test.pvalue) AS test_pvalue,
           CASE
               WHEN COALESCE(population_stability_index, 0) > 0.2 THEN 'MAJOR DRIFT'
               WHEN COALESCE(population_stability_index, 0) > 0.1 THEN 'MODERATE DRIFT'
               WHEN COALESCE(ks_test.pvalue, chi_squared_test.pvalue, 1) < 0.05 THEN 'SIGNIFICANT'
               ELSE 'STABLE'
           END AS drift_status
    FROM {catalog}.{schema}.churn_predictions_drift_metrics
    WHERE slice_key IS NULL
      AND window IS NOT NULL
    ORDER BY COALESCE(population_stability_index, 0) DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Quality with Ground Truth
# MAGIC
# MAGIC In production, **labels arrive after predictions** — a customer churns days or weeks after being scored. Lakehouse Monitoring computes classification metrics (accuracy, precision, recall, F1) automatically when a label column is present.
# MAGIC
# MAGIC We'll MERGE ground-truth labels into the predictions table and re-run the monitor.

# COMMAND ----------

# Add label column as DOUBLE (must match prediction column type for monitoring)
existing_cols = [c.lower() for c in spark.table(predictions_table).columns]
if "churn_label" not in existing_cols:
    spark.sql(f"ALTER TABLE {predictions_table} ADD COLUMNS (churn_label DOUBLE)")

# MERGE ground-truth labels into predictions
# Model uses LabelEncoder: Yes=1, No=0 — match that encoding
spark.sql(f"""
    MERGE INTO {predictions_table} AS p
    USING (
        SELECT customer_id, CAST(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END AS DOUBLE) AS churn_label
        FROM {catalog}.{schema}.churn_labels
    ) AS l
    ON p.customer_id = l.customer_id
    WHEN MATCHED THEN UPDATE SET p.churn_label = l.churn_label
""")

label_coverage = spark.sql(f"""
    SELECT
        COUNT(*) AS total_rows,
        SUM(CASE WHEN churn_label IS NOT NULL THEN 1 ELSE 0 END) AS labeled_rows,
        ROUND(100.0 * SUM(CASE WHEN churn_label IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_labeled
    FROM {predictions_table}
""")
display(label_coverage)

# COMMAND ----------

# Update monitor to include label column, then refresh
w.quality_monitors.update(
    table_name=predictions_table,
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        label_col="churn_label",
        model_id_col="model_id",
        timestamp_col="inference_timestamp",
        granularities=["1 month"],
    ),
    baseline_table_name=baseline_table,
    custom_metrics=[high_risk_metric],
    output_schema_name=f"{catalog}.{schema}",
)

w.quality_monitors.run_refresh(table_name=predictions_table)
print("Refresh triggered with label column — model quality metrics will be computed")
print("This takes 1-3 minutes. You can continue reading while it runs.")

# COMMAND ----------

# Model quality metrics (run after refresh completes)
display(spark.sql(f"""
    SELECT window,
           accuracy, precision, recall, f1_score, log_loss
    FROM {catalog}.{schema}.churn_predictions_profile_metrics
    WHERE column_name = 'prediction'
      AND slice_key IS NULL
      AND accuracy IS NOT NULL
    ORDER BY window
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Custom Metrics
# MAGIC
# MAGIC The `high_risk_percentage` metric we defined uses a **Jinja SQL template**. Lakehouse Monitoring evaluates it per time window, so we can track how the business-level KPI changes over time.
# MAGIC
# MAGIC Template: `100.0 * SUM(CASE WHEN {{prediction_col}} = 1 THEN 1 ELSE 0 END) / COUNT(*)`
# MAGIC
# MAGIC The `{{prediction_col}}` placeholder is automatically replaced with the monitor's configured prediction column.

# COMMAND ----------

display(spark.sql(f"""
    SELECT window, column_name, high_risk_percentage
    FROM {catalog}.{schema}.churn_predictions_profile_metrics
    WHERE high_risk_percentage IS NOT NULL
      AND slice_key IS NULL
    ORDER BY window
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Alerting & Retraining Triggers
# MAGIC
# MAGIC In production, the monitoring pipeline looks like:
# MAGIC
# MAGIC ```
# MAGIC Monitor refresh (scheduled) → drift_metrics table → SQL Alert (threshold check)
# MAGIC    → Notification (email/Slack/PagerDuty) → Workflow (retrain pipeline)
# MAGIC ```
# MAGIC
# MAGIC Below is a **working alert query** that you'd plug into a Databricks SQL Alert.

# COMMAND ----------

alert_results = spark.sql(f"""
    SELECT column_name, window,
           COALESCE(population_stability_index, 0) AS psi,
           COALESCE(ks_test.pvalue, chi_squared_test.pvalue) AS test_pvalue,
           CASE
               WHEN COALESCE(population_stability_index, 0) > 0.2 THEN 'CRITICAL — major distribution shift'
               WHEN COALESCE(population_stability_index, 0) > 0.1 THEN 'WARNING — moderate drift detected'
               WHEN COALESCE(ks_test.pvalue, chi_squared_test.pvalue, 1) < 0.05 THEN 'ALERT — statistically significant drift'
               ELSE 'OK'
           END AS status
    FROM {catalog}.{schema}.churn_predictions_drift_metrics
    WHERE slice_key IS NULL
      AND (
          COALESCE(population_stability_index, 0) > 0.1
          OR COALESCE(ks_test.pvalue, chi_squared_test.pvalue, 1) < 0.05
      )
    ORDER BY COALESCE(population_stability_index, 0) DESC
""")

display(alert_results)

alert_count = alert_results.count()
print(f"\n{'='*60}")
print(f"  {alert_count} drift alert(s) fired across all windows")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Dashboard & Next Steps
# MAGIC
# MAGIC Lakehouse Monitoring **auto-generates a Lakeview dashboard** with:
# MAGIC - Profile distributions per column per window
# MAGIC - Drift test results with color-coded thresholds
# MAGIC - Model quality trends (when labels are present)
# MAGIC
# MAGIC You can share it via **scheduled PDF/email subscriptions**, embed in **Genie Spaces**, or extend with custom visualizations.

# COMMAND ----------

# Print dashboard location
try:
    info = w.quality_monitors.get(table_name=predictions_table)
    print(f"Dashboard assets: {info.assets_dir}")
    print(f"Dashboard ID: {info.dashboard_id}")
    print(f"\nOpen the dashboard in your Databricks workspace to see auto-generated visualizations.")
except Exception as e:
    print(f"Monitor info: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | What we built | Details |
# MAGIC |--------------|---------|
# MAGIC | **Baseline table** | Scored training split → `churn_predictions_baseline` |
# MAGIC | **Drift simulation** | Shifted `monthly_charges`, `total_charges`, `tenure_months`, `contract_type` |
# MAGIC | **Monitor** | InferenceLog with monthly granularity, baseline comparison |
# MAGIC | **Drift detection** | KS, chi-squared, PSI, Wasserstein, JS divergence, total variation |
# MAGIC | **Model quality** | Accuracy, precision, recall, F1 via ground-truth label MERGE |
# MAGIC | **Custom metric** | `high_risk_percentage` — business KPI per window |
# MAGIC | **Alerting** | SQL query with PSI/p-value thresholds → ready for SQL Alerts |
# MAGIC | **Dashboard** | Auto-generated Lakeview dashboard |
# MAGIC
# MAGIC ### Databricks Lakehouse Monitoring vs. Third-Party Tools
# MAGIC
# MAGIC | Capability | Lakehouse Monitoring | Evidently / Great Expectations |
# MAGIC |-----------|---------------------|-------------------------------|
# MAGIC | **Setup** | One API call — no infrastructure | Separate service / container to manage |
# MAGIC | **Data stays in place** | Reads from Delta tables directly | Requires data export or connector |
# MAGIC | **Drift tests** | 6 built-in (KS, chi², PSI, Wasserstein, JS, TV) | Varies by tool |
# MAGIC | **Custom metrics** | Jinja SQL templates — any SQL expression | Python functions |
# MAGIC | **Dashboard** | Auto-generated Lakeview | Manual setup |
# MAGIC | **Alerting** | Native SQL Alerts → email/Slack/PagerDuty | Separate alerting pipeline |
# MAGIC | **GenAI monitoring** | Text quality, toxicity, topic drift | Varies |
# MAGIC | **Cost** | Serverless SQL compute for refresh | Infrastructure + license costs |
# MAGIC | **Governance** | Unity Catalog lineage built-in | External metadata management |

# COMMAND ----------

# # Cleanup (uncomment to remove monitor and tables)
# w.quality_monitors.delete(table_name=predictions_table)
# spark.sql(f"DROP TABLE IF EXISTS {baseline_table}")
# spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.churn_predictions_profile_metrics")
# spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.churn_predictions_drift_metrics")
# print("Monitor and metric tables removed")
