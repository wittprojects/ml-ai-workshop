# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Configuration
# MAGIC
# MAGIC **Instructor**: Set these values before cloning for participants.
# MAGIC All downstream notebooks reference these variables via `%run ../_resources/00_config`.

# COMMAND ----------

# ---- Catalog & Schema ----
catalog = "wittprojects"
schema = "workshop"

# ---- Foundation Model API Endpoints ----
llm_endpoint = "databricks-claude-sonnet-4-6"
embedding_endpoint = "databricks-gte-large-en"

# ---- Vector Search ----
vector_search_endpoint = "workshop_vs_endpoint"
product_knowledge_index = f"{catalog}.{schema}.product_knowledge_index"

# ---- Model Serving ----
churn_model_serving_endpoint = "workshop-churn-model"

# ---- Lakebase Online Store ----
online_store_name = "workshop-online-store"

# COMMAND ----------

# Derived references (do not edit)
feature_table_name = f"{catalog}.{schema}.churn_feature_table"
model_name = f"{catalog}.{schema}.churn_model"
feature_spec_name = f"{catalog}.{schema}.churn_feature_spec"
documents_volume = f"{catalog}.{schema}.documents"
documents_volume_path = f"/Volumes/{catalog}/{schema}/documents"
documents_source_path = f"{documents_volume_path}/source"
parsed_images_path = f"{documents_volume_path}/parsed_images"

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
