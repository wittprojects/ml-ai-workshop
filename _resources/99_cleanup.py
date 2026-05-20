# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Workshop Cleanup — destructive
# MAGIC
# MAGIC Wipes every asset the workshop creates so the next end-to-end run starts on a
# MAGIC clean slate. Removes:
# MAGIC
# MAGIC - Lakehouse Monitor on `churn_predictions` (+ its workspace dashboard folder)
# MAGIC - Genie space(s) titled "Telecom Churn Analytics"
# MAGIC - Model serving endpoint `workshop-churn-model`
# MAGIC - Vector Search index `product_knowledge_index` and endpoint `workshop_vs_endpoint`
# MAGIC - `DROP SCHEMA wittprojects.workshop CASCADE` — every table, view, UC function, registered model, and volume
# MAGIC - MLflow experiments owned by the running user (`ml-ai-workshop-churn`, `ml-ai-workshop-genai`)
# MAGIC
# MAGIC The catalog `wittprojects` is **not** dropped (shared infra). Other participants'
# MAGIC MLflow experiments are **not** touched (theirs to clean).
# MAGIC
# MAGIC Reciprocal verify notebook: `99_verify_grants.py`.

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confirmation gate
# MAGIC
# MAGIC Type `DELETE` into the widget at the top of the notebook before running. A blank
# MAGIC widget raises and aborts — "Run all" without thought is a no-op.

# COMMAND ----------

dbutils.widgets.text("confirm", "", "Type DELETE to confirm")
if dbutils.widgets.get("confirm") != "DELETE":
    raise RuntimeError(
        "Cleanup aborted. Set the `confirm` widget at the top of the notebook to "
        "DELETE to run."
    )
print("✓ confirmed — proceeding with destructive cleanup")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
current_user = spark.sql("SELECT current_user()").first()[0]
print(f"Running as: {current_user}")

actions: list[str] = []

def _short_error(e: BaseException) -> str:
    msg = str(e)
    for sentinel in (" SQLSTATE:", "\nJVM stacktrace:", "\n'", "\n+- "):
        idx = msg.find(sentinel)
        if idx != -1:
            msg = msg[:idx]
    return msg.strip()

def step(label: str, fn):
    try:
        result = fn()
        if result == "skip":
            actions.append(f"(already gone) {label}")
            print(f"  · (already gone) {label}")
        else:
            actions.append(f"deleted {label}")
            print(f"  ✓ deleted {label}")
    except Exception as e:
        msg = f"{label}: {type(e).__name__}: {_short_error(e)}"
        actions.append(f"FAILED {msg}")
        print(f"  ✗ {msg}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Lakehouse Monitor on `churn_predictions`
# MAGIC
# MAGIC Must run before the schema drop — the monitor's auto-generated
# MAGIC `*_profile_metrics` and `*_drift_metrics` tables can't be dropped while the
# MAGIC monitor still owns them.

# COMMAND ----------

def _delete_monitor():
    try:
        w.quality_monitors.get(table_name=predictions_table_name)
    except Exception:
        return "skip"
    w.quality_monitors.delete(table_name=predictions_table_name)

step(f"Lakehouse Monitor on {predictions_table_name}", _delete_monitor)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Genie space(s) "Telecom Churn Analytics"
# MAGIC
# MAGIC Paginates `list_spaces` and trashes every match — handles duplicates left by
# MAGIC prior re-runs before the create cell was made idempotent.

# COMMAND ----------

def _delete_genie_spaces():
    title = "Telecom Churn Analytics"
    matches = []
    token = None
    while True:
        resp = w.genie.list_spaces(page_token=token) if token else w.genie.list_spaces()
        for s in (getattr(resp, "spaces", None) or []):
            if getattr(s, "title", None) == title:
                matches.append(s)
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    if not matches:
        return "skip"
    for s in matches:
        w.genie.trash_space(space_id=s.space_id)
        print(f"    · trashed Genie space {s.space_id}")
    print(f"  ({len(matches)} space(s) trashed)")

step("Genie spaces titled 'Telecom Churn Analytics'", _delete_genie_spaces)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model serving endpoint `workshop-churn-model`
# MAGIC
# MAGIC Fire-and-forget. Deletion is async and takes a few minutes to drain, but doesn't
# MAGIC block subsequent cleanup.

# COMMAND ----------

def _delete_serving_endpoint():
    try:
        w.serving_endpoints.get(name=churn_model_serving_endpoint)
    except Exception:
        return "skip"
    w.serving_endpoints.delete(name=churn_model_serving_endpoint)

step(f"serving endpoint {churn_model_serving_endpoint}", _delete_serving_endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vector Search index, then endpoint
# MAGIC
# MAGIC Index has to go first — deleting the endpoint with a live index fails.

# COMMAND ----------

def _delete_vs_index():
    try:
        w.vector_search_indexes.get_index(index_name=product_knowledge_index)
    except Exception:
        return "skip"
    w.vector_search_indexes.delete_index(index_name=product_knowledge_index)

step(f"vector search index {product_knowledge_index}", _delete_vs_index)

def _delete_vs_endpoint():
    try:
        w.vector_search_endpoints.get_endpoint(endpoint_name=vector_search_endpoint)
    except Exception:
        return "skip"
    w.vector_search_endpoints.delete_endpoint(endpoint_name=vector_search_endpoint)

step(f"vector search endpoint {vector_search_endpoint}", _delete_vs_endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `DROP SCHEMA ... CASCADE`
# MAGIC
# MAGIC One statement wipes every UC child: foundation tables, feature table, predictions
# MAGIC + baseline, monitor metric tables, parsed_documents, views, all 6 UC functions,
# MAGIC the `churn_model` registered model, and the `documents` volume (with contents).

# COMMAND ----------

def _drop_schema():
    rows = spark.sql(f"SHOW SCHEMAS IN {catalog} LIKE '{schema}'").collect()
    if not rows:
        return "skip"
    spark.sql(f"DROP SCHEMA {catalog}.{schema} CASCADE")

step(f"schema {catalog}.{schema} (CASCADE)", _drop_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MLflow experiments for the running user
# MAGIC
# MAGIC Only wipes experiments owned by whoever runs cleanup. Each participant manages
# MAGIC their own MLflow experiments.

# COMMAND ----------

import mlflow

for exp_suffix in ("ml-ai-workshop-churn", "ml-ai-workshop-genai"):
    path = f"/Users/{current_user}/{exp_suffix}"
    def _delete_exp(p=path):
        exp = mlflow.get_experiment_by_name(p)
        if exp is None:
            return "skip"
        mlflow.delete_experiment(exp.experiment_id)
    step(f"MLflow experiment {path}", _delete_exp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Monitor dashboard assets folder
# MAGIC
# MAGIC The Lakehouse Monitor leaves a workspace folder behind (dashboard + queries) that
# MAGIC `quality_monitors.delete` doesn't reap.

# COMMAND ----------

def _delete_monitor_assets():
    path = f"/Workspace/Users/{current_user}/databricks_lakehouse_monitoring/{predictions_table_name}"
    try:
        w.workspace.get_status(path)
    except Exception:
        return "skip"
    w.workspace.delete(path=path, recursive=True)

step(f"monitor dashboard folder for {predictions_table_name}", _delete_monitor_assets)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

failures = [a for a in actions if a.startswith("FAILED")]
deletions = [a for a in actions if a.startswith("deleted")]
skipped   = [a for a in actions if a.startswith("(already gone)")]

print("=" * 60)
print(f"  Deleted:        {len(deletions)}")
print(f"  Already gone:   {len(skipped)}")
print(f"  Failed:         {len(failures)}")
print("=" * 60)

if failures:
    print("\nFailures:")
    for f in failures:
        print(f"  ✗ {f}")
    raise AssertionError(f"{len(failures)} cleanup step(s) failed — see output above")

print("\n✓ workshop assets cleaned — safe to re-run _resources/01_setup.py")
