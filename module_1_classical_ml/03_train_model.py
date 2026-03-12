# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Classical ML on Databricks
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
training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churn",
    exclude_columns=["customer_id", "label_date", "update_timestamp"],
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
X = pdf.drop(columns=["churn", "split"])

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

# COMMAND ----------

# Split using the pre-assigned split column from labels
train_mask = pdf["split"] == "train"
X_train, X_val = X[train_mask].drop(columns=["split"], errors="ignore"), X[~train_mask].drop(columns=["split"], errors="ignore")
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
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
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

    # Log with Feature Engineering client for lineage
    fe.log_model(
        model=final_pipeline,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=None,  # We'll register in the next notebook
    )

    final_run_id = run.info.run_id
    print(f"\nModel logged: run_id={final_run_id}")

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

# COMMAND ----------

# Summary plot — shows top features and their impact direction
shap.summary_plot(shap_values[1], X_val_processed, feature_names=all_features, show=True)

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
# MAGIC - **Feature Store** lineage preserved via `fe.log_model()`
# MAGIC - **SHAP** revealed the most important churn drivers
# MAGIC
# MAGIC **Next**: [03a Genie Code Alternative →](./03a_genie_code_alternative) or [04 Model Registry →](./04_model_registry)
