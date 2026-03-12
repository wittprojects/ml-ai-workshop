# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Classical ML on Databricks
# MAGIC ## Notebook 04 — Model Registry with Unity Catalog
# MAGIC
# MAGIC **Time**: ~8 min
# MAGIC
# MAGIC We'll register our best model to Unity Catalog and set up Champion/Challenger aliases.
# MAGIC
# MAGIC Key concepts:
# MAGIC - UC model registry (`mlflow.register_model()`)
# MAGIC - Model versions, aliases, and tags
# MAGIC - Lineage tracking in Catalog Explorer

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find the Best Run

# COMMAND ----------

experiment_path = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/ml-ai-workshop-churn"
experiment = mlflow.get_experiment_by_name(experiment_path)

# Search for the best final model run by F1 score
best_run = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'final_model'",
    order_by=["metrics.f1_score DESC"],
    max_results=1,
).iloc[0]

best_run_id = best_run.run_id
best_f1 = best_run["metrics.f1_score"]
best_auc = best_run["metrics.roc_auc"]

print(f"Best run: {best_run_id}")
print(f"  F1 Score: {best_f1:.4f}")
print(f"  ROC AUC:  {best_auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model to Unity Catalog

# COMMAND ----------

model_uri = f"runs:/{best_run_id}/model"
mv = mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"✓ Registered model: {model_name}")
print(f"  Version: {mv.version}")

# COMMAND ----------

# Set model version description and tags
client.update_model_version(
    name=model_name,
    version=mv.version,
    description=f"LightGBM churn prediction model. F1={best_f1:.4f}, AUC={best_auc:.4f}. "
                f"Trained with Optuna hyperparameter tuning (20 trials)."
)
client.set_model_version_tag(name=model_name, version=mv.version, key="f1_score", value=f"{best_f1:.4f}")
client.set_model_version_tag(name=model_name, version=mv.version, key="roc_auc", value=f"{best_auc:.4f}")
client.set_model_version_tag(name=model_name, version=mv.version, key="training_framework", value="lightgbm+optuna")

print(f"✓ Tags and description set for version {mv.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Model Aliases
# MAGIC
# MAGIC Aliases like `Champion` and `Challenger` enable deployment without hardcoding version numbers.

# COMMAND ----------

# First, set as Challenger for validation
client.set_registered_model_alias(name=model_name, alias="Challenger", version=mv.version)
print(f"✓ Version {mv.version} set as 'Challenger'")

# COMMAND ----------

# After validation, promote to Champion
client.set_registered_model_alias(name=model_name, alias="Champion", version=mv.version)
print(f"✓ Version {mv.version} promoted to 'Champion'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Registration

# COMMAND ----------

# Load the Champion model to verify
champion_version = client.get_model_version_by_alias(name=model_name, alias="Champion")
print(f"Champion model: {model_name} v{champion_version.version}")
print(f"  Status: {champion_version.status}")
print(f"  Description: {champion_version.description}")

# COMMAND ----------

# Load and test the model
champion_model = mlflow.sklearn.load_model(f"models:/{model_name}@Champion")
print(f"✓ Champion model loaded successfully")
print(f"  Type: {type(champion_model)}")
print(f"  Steps: {[step[0] for step in champion_model.steps]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog Lineage
# MAGIC
# MAGIC Navigate to **Catalog Explorer** → `{catalog}` → `{schema}` → **Models** → `churn_model` to see:
# MAGIC - Model versions with aliases
# MAGIC - **Lineage graph**: which feature tables and functions were used to train this model
# MAGIC - Tags and descriptions for governance
# MAGIC
# MAGIC This lineage was automatically captured because we used `fe.log_model()` in the training notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We registered our model to Unity Catalog with:
# MAGIC - **Version tracking** — each training run creates a new version
# MAGIC - **Aliases** — `Champion` for production, `Challenger` for validation
# MAGIC - **Tags** — performance metrics for governance
# MAGIC - **Lineage** — automatic tracking of feature tables and functions
# MAGIC
# MAGIC **Next**: [05 Model Serving →](./05_model_serving)
