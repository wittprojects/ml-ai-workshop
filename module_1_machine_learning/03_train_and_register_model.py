# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Module 1: Machine Learning on Databricks
# MAGIC ## Notebook 03 — Model Training with Optuna + MLflow
# MAGIC
# MAGIC **Time**: ~15 min
# MAGIC
# MAGIC We'll train a **LightGBM** classifier with **Optuna** hyperparameter tuning, tracked by **MLflow**.
# MAGIC
# MAGIC Key concepts:
# MAGIC - `fe.create_training_set()` for feature lineage
# MAGIC - sklearn preprocessing pipeline
# MAGIC - Optuna study with MLflow autologging
# MAGIC - SHAP feature importance

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering==0.14.0 databricks-sdk==0.102.0 optuna==4.8.0 lightgbm==4.6.0 shap==0.46.0 "numpy<2" uv==0.10.11 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup, FeatureFunction

fe = FeatureEngineeringClient()

# Set the MLflow experiment
experiment_path = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/ml-ai-workshop-churn"
mlflow.set_experiment(experiment_path)
print(f"MLflow experiment: {experiment_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training Set via Feature Store
# MAGIC
# MAGIC Using `fe.create_training_set()` ensures full lineage between features and the trained model.

# COMMAND ----------

# Labels table
labels_df = spark.table("churn_labels").filter("split IN ('train', 'val')")

# Define feature lookups
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

# Create training set
# Exclude split — it's a label-side column, not a serving input
training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churn",
    exclude_columns=["customer_id", "label_date", "update_timestamp", "split"],
)

training_df = training_set.load_df()
print(f"Training set shape: {training_df.count()} rows, {len(training_df.columns)} columns")
display(training_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing Pipeline
# MAGIC
# MAGIC Build a scikit-learn pipeline to handle categorical, boolean, and numerical features.

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Convert to pandas
pdf = training_df.toPandas()

# Encode target
le = LabelEncoder()
y = le.fit_transform(pdf["churn"])  # Yes=1, No=0
X = pdf.drop(columns=["churn"], errors='ignore')

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

# COMMAND ----------

# Get the split column from the original labels (excluded from training_set)
split_series = labels_df.select("split").toPandas()["split"]
train_mask = split_series == "train"

X_train, X_val = X[train_mask], X[~train_mask]
y_train, y_val = y[train_mask], y[~train_mask]

# Identify column types
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# COMMAND ----------

# Build preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning with Optuna
# MAGIC
# MAGIC Optuna provides efficient Bayesian optimization. Each trial is logged as an MLflow child run.

# COMMAND ----------

import optuna
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# COMMAND ----------

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgb.LGBMClassifier(**params, random_state=42, verbose=-1)),
    ])

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)

        mlflow.log_params(params)
        mlflow.log_metrics({"f1_score": f1, "roc_auc": auc})

    return f1

# COMMAND ----------

# Run the study
with mlflow.start_run(run_name="optuna_tuning") as parent_run:
    study = optuna.create_study(direction="maximize", study_name="churn_lgbm")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    # Log best params to parent run
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_f1_score", study.best_value)

    parent_run_id = parent_run.info.run_id
    print(f"\nBest F1 Score: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Final Model with Best Parameters

# COMMAND ----------

best_params = study.best_params

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)),
])

# Conda Env for a more deterministic build. 
conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.12.3",
        "pip<=25.0.1",
        {
            "pip": [
                "mlflow==3.8.1",
                "scikit-learn==1.6.1",
                "lightgbm==4.6.0",
                "pyarrow==21.0.0",
                "cloudpickle==3.0.0",
            ]
        },
    ],
    "name": "mlflow-env",
}

# Input example: a few rows of the feature vector the model expects at inference time.
input_example = X_val.head(5)

with mlflow.start_run(run_name="final_model") as run:
    final_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = final_pipeline.predict(X_val)
    y_proba = final_pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "f1_score": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
    }
    mlflow.log_metrics(metrics)
    print("Final model metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        artifact_path="final_model",
        input_example=input_example,
        conda_env=conda_env,
    )

    final_run_id = run.info.run_id
    print(f"\nModel logged: run_id={final_run_id}")

# COMMAND ----------

# DBTITLE 1,Smoke test model before registration
# Load the logged model and predict against a few rows of the validation set.
loaded_model = mlflow.sklearn.load_model(f"runs:/{final_run_id}/final_model")
preds = loaded_model.predict(X_val.head(5))
probs = loaded_model.predict_proba(X_val.head(5))[:, 1]
print("✓ Smoke test passed!")
display(pd.DataFrame({"prediction": preds, "churn_probability": probs}))

# COMMAND ----------

# DBTITLE 1,Register model header
# MAGIC %md
# MAGIC ## Register Model to Unity Catalog
# MAGIC
# MAGIC Register directly from the training run to ensure we're registering the model we just trained — not a stale historical run.

# COMMAND ----------

# DBTITLE 1,Register and promote model
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

# Register the model from the training run
model_uri = f"runs:/{final_run_id}/final_model"
mv = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"✓ Registered model: {model_name} v{mv.version}")

# Set description and tags
mlflow_client.update_model_version(
    name=model_name,
    version=mv.version,
    description=f"LightGBM churn prediction model. F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}. "
                f"Trained with Optuna hyperparameter tuning (20 trials)."
)
mlflow_client.set_model_version_tag(name=model_name, version=mv.version, key="f1_score", value=f"{metrics['f1_score']:.4f}")
mlflow_client.set_model_version_tag(name=model_name, version=mv.version, key="roc_auc", value=f"{metrics['roc_auc']:.4f}")
mlflow_client.set_model_version_tag(name=model_name, version=mv.version, key="training_framework", value="lightgbm+optuna")

# Promote to Champion
mlflow_client.set_registered_model_alias(name=model_name, alias="Challenger", version=mv.version)
mlflow_client.set_registered_model_alias(name=model_name, alias="Champion", version=mv.version)
print(f"✓ Version {mv.version} promoted to 'Champion'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SHAP Feature Importance
# MAGIC
# MAGIC SHAP (SHapley Additive exPlanations) shows which features drive predictions.

# COMMAND ----------

import shap

# Get preprocessed data for SHAP
X_val_processed = final_pipeline.named_steps["preprocessor"].transform(X_val)

# Get feature names after preprocessing
num_features = numerical_cols
cat_features = list(final_pipeline.named_steps["preprocessor"]
                    .named_transformers_["cat"]
                    .get_feature_names_out(categorical_cols))
all_features = num_features + cat_features

explainer = shap.TreeExplainer(final_pipeline.named_steps["classifier"])
shap_values = explainer.shap_values(X_val_processed)

# Handle both old SHAP (list of arrays per class) and new SHAP (single array)
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
else:
    shap_values_pos = shap_values

# COMMAND ----------

# Summary plot — shows top features and their impact direction
shap.summary_plot(shap_values_pos, X_val_processed, feature_names=all_features, show=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix & ROC Curve

# COMMAND ----------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, display_labels=["No Churn", "Churn"], ax=axes[0])
axes[0].set_title("Confusion Matrix")

# ROC curve
RocCurveDisplay.from_predictions(y_val, y_proba, ax=axes[1])
axes[1].set_title("ROC Curve")

plt.tight_layout()
plt.show()

# COMMAND ----------

# Save the run_id for the next notebook
spark.sql(f"""
CREATE OR REPLACE TEMPORARY VIEW best_run AS
SELECT '{final_run_id}' as run_id, {metrics['f1_score']} as f1_score
""")

print(f"Best run_id saved: {final_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We trained a LightGBM churn prediction model:
# MAGIC - **Optuna** explored 20 hyperparameter combinations
# MAGIC - All trials logged to **MLflow** with full tracking
# MAGIC - **Feature Store** lineage preserved via `fe.create_training_set()`
# MAGIC - **SHAP** revealed the most important churn drivers
# MAGIC
# MAGIC **Next**: [03a Genie Code Alternative →](./03a_genie_code_alternative) or [04 Model Serving →](./04_model_serving)
