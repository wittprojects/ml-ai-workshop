# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Machine Learning on Databricks
# MAGIC ## Notebook 03a — Genie Code Alternative Track
# MAGIC
# MAGIC **Time**: Self-paced
# MAGIC
# MAGIC This notebook walks through how to use **Genie Code** (Databricks' AI coding assistant) to accomplish the same model training as Notebook 03 — but conversationally.
# MAGIC
# MAGIC > This is a **markdown-only** notebook. Follow the steps in the Databricks UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is Genie Code?
# MAGIC
# MAGIC Genie Code is Databricks' built-in coding agent. It can:
# MAGIC - Generate complete notebooks from natural language descriptions
# MAGIC - Access your Unity Catalog tables and understand your schema
# MAGIC - Write, execute, and iterate on code cells
# MAGIC - Log experiments to MLflow automatically
# MAGIC
# MAGIC **When to use Genie Code vs. manual notebooks:**
# MAGIC - ✅ Rapid prototyping and exploration
# MAGIC - ✅ When you know *what* you want but not the exact API
# MAGIC - ✅ Generating boilerplate (preprocessing, evaluation, visualization)
# MAGIC - ⚠️ Always verify data leakage, train/test splits, and metric calculations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step-by-Step: Build a Churn Model with Genie Code
# MAGIC
# MAGIC ### Step 1: Open Genie Code
# MAGIC 1. In the Databricks sidebar, click **New** → **Notebook**
# MAGIC 2. Click the **Genie Code** toggle (sparkle icon) at the top of the notebook
# MAGIC 3. You'll see a natural language input box
# MAGIC
# MAGIC ### Step 2: Describe Your Task
# MAGIC Start with a clear, specific prompt:
# MAGIC
# MAGIC ```
# MAGIC Load the churn_feature_table and churn_labels tables from the ml_ai_workshop.workshop schema.
# MAGIC Join them on customer_id. The target column is 'churn' (Yes/No).
# MAGIC Use the 'split' column in churn_labels for train/val/test partitioning.
# MAGIC Train a LightGBM classifier, optimize F1 score using cross-validation,
# MAGIC and log the model to MLflow. Show a confusion matrix and SHAP feature importance.
# MAGIC ```
# MAGIC
# MAGIC ### Step 3: Review Generated Code
# MAGIC Genie Code will generate multiple cells. **Before running**, check:
# MAGIC - [ ] Data is loaded from the correct tables
# MAGIC - [ ] Target encoding is correct (Yes=1, No=0)
# MAGIC - [ ] Train/test split uses the `split` column (not random)
# MAGIC - [ ] No data leakage (target not in features)
# MAGIC - [ ] MLflow experiment is set correctly
# MAGIC
# MAGIC ### Step 4: Iterate
# MAGIC Ask Genie Code to refine the output:
# MAGIC - *"Add Optuna hyperparameter tuning with 20 trials"*
# MAGIC - *"Use the FeatureEngineeringClient to log the model with feature lineage"*
# MAGIC - *"Show precision-recall curve alongside the ROC curve"*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tips & Tricks for Prompting Coding Agents
# MAGIC
# MAGIC ### 🎯 Be Specific About Data
# MAGIC
# MAGIC **Vague** ❌: *"Train a model on the customer data"*
# MAGIC
# MAGIC **Specific** ✅: *"Train a LightGBM classifier on `ml_ai_workshop.workshop.churn_feature_table` joined with `churn_labels` on customer_id. Target is `churn` (binary: Yes/No). Use the `split` column for partitioning."*
# MAGIC
# MAGIC ### 🎯 Specify the Metric
# MAGIC
# MAGIC **Vague** ❌: *"Make it accurate"*
# MAGIC
# MAGIC **Specific** ✅: *"Optimize for F1 score since the classes are imbalanced (~26% churn). Also report AUC, precision, and recall."*
# MAGIC
# MAGIC ### 🎯 Request Specific Libraries
# MAGIC
# MAGIC **Vague** ❌: *"Tune the hyperparameters"*
# MAGIC
# MAGIC **Specific** ✅: *"Use Optuna with 20 trials to tune LightGBM hyperparameters. Log each trial as a nested MLflow run."*
# MAGIC
# MAGIC ### 🎯 Ask for Explainability
# MAGIC
# MAGIC *"Add SHAP feature importance analysis. Show a summary plot of the top 15 features."*
# MAGIC
# MAGIC ### 🎯 Iterate in Small Steps
# MAGIC
# MAGIC Don't ask for everything at once. Build up:
# MAGIC 1. First: load data and explore
# MAGIC 2. Then: build preprocessing pipeline
# MAGIC 3. Then: train baseline model
# MAGIC 4. Then: add hyperparameter tuning
# MAGIC 5. Then: add evaluation and visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ## Common Pitfalls to Watch For
# MAGIC
# MAGIC | Pitfall | What to Check |
# MAGIC |---------|---------------|
# MAGIC | **Data leakage** | Is the `churn` column excluded from features? Are ticket features computed only from pre-churn data? |
# MAGIC | **Wrong split** | Is it using the `split` column or creating a random split? Random splits can leak information if data has temporal structure. |
# MAGIC | **Missing encoding** | Are categorical columns properly one-hot encoded? Is the target label-encoded? |
# MAGIC | **No random seed** | Results should be reproducible. Check for `random_state=42` in model and split. |
# MAGIC | **Overfitting** | Compare train vs. validation metrics. A large gap suggests overfitting. |
# MAGIC | **Wrong metric** | For imbalanced data, accuracy is misleading. F1 or AUC is more appropriate. |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparing Approaches
# MAGIC
# MAGIC | Aspect | Notebook 03 (Manual) | This Track (Genie Code) |
# MAGIC |--------|---------------------|------------------------|
# MAGIC | Time to first model | ~15 min | ~5 min |
# MAGIC | Customization | Full control | Iterative prompting |
# MAGIC | Reproducibility | Exact (checked into Git) | Depends on prompt consistency |
# MAGIC | Learning value | Deep understanding | Faster exploration |
# MAGIC | Production readiness | High | Needs review and refinement |
# MAGIC
# MAGIC **Both approaches produce the same output**: a registered MLflow model with Feature Store lineage.
# MAGIC The best approach depends on your use case and experience level.
# MAGIC
# MAGIC **Next**: [05 Model Serving →](./05_model_serving)
