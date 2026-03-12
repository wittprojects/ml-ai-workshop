# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Configuration
# MAGIC
# MAGIC **Instructor**: Set these values before cloning for participants.
# MAGIC All downstream notebooks reference these variables via `%run ../_resources/00_config`.

# COMMAND ----------

# ---- Catalog & Schema ----
catalog = "ml_ai_workshop"
schema = "workshop"

# ---- Foundation Model API Endpoints ----
llm_endpoint = "databricks-claude-sonnet-4-6"
embedding_endpoint = "databricks-gte-large-en"

# ---- Vector Search ----
vector_search_endpoint = "workshop_vs_endpoint"
product_knowledge_index = f"{catalog}.{schema}.product_knowledge_index"

# ---- Model Serving ----
churn_model_serving_endpoint = "workshop-churn-model"

# COMMAND ----------

# Derived references (do not edit)
feature_table_name = f"{catalog}.{schema}.churn_feature_table"
model_name = f"{catalog}.{schema}.churn_model"

# COMMAND ----------

# Set default catalog and schema for SQL (skip if not yet created by setup)
try:
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"USE SCHEMA {schema}")
except Exception:
    print(f"Note: catalog '{catalog}' or schema '{schema}' not yet created — run 01_setup first")

print(f"Catalog:            {catalog}")
print(f"Schema:             {schema}")
print(f"LLM Endpoint:       {llm_endpoint}")
print(f"Embedding Endpoint: {embedding_endpoint}")
print(f"Feature Table:      {feature_table_name}")
print(f"Model:              {model_name}")
