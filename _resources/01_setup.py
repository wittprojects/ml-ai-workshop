# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Workshop Setup
# MAGIC
# MAGIC **Run once by the instructor** to provision data and resources.
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Creates the catalog and schema
# MAGIC 2. Generates all synthetic data tables
# MAGIC 3. Creates a Vector Search endpoint and index on `product_knowledge`
# MAGIC 4. Grants permissions to `account users`

# COMMAND ----------

# MAGIC %pip install faker databricks-sdk==0.50.0 reportlab==4.2.5 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %run ./data_generators

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Catalog & Schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

print(f"✓ Catalog '{catalog}' and schema '{schema}' ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate and Write Data Tables

# COMMAND ----------

# Generate customers first — other tables depend on it
customers_pdf = generate_customers(n=5000, seed=42)
print(f"Generated {len(customers_pdf)} customers, churn rate: {customers_pdf['churn'].value_counts(normalize=True)['Yes']:.1%}")

# COMMAND ----------

# Write customers table
customers_sdf = spark.createDataFrame(customers_pdf)
customers_sdf.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.customers")
print(f"✓ customers table: {spark.table(f'{catalog}.{schema}.customers').count()} rows")

# COMMAND ----------

# Generate and write service_tickets
tickets_pdf = generate_service_tickets(customers_pdf, n=15000, seed=42)
spark.createDataFrame(tickets_pdf).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.service_tickets")
print(f"✓ service_tickets table: {spark.table(f'{catalog}.{schema}.service_tickets').count()} rows")

# COMMAND ----------

# Generate and write call_transcripts
transcripts_pdf = generate_call_transcripts(customers_pdf, n=3000, seed=42)
spark.createDataFrame(transcripts_pdf).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.call_transcripts")
print(f"✓ call_transcripts table: {spark.table(f'{catalog}.{schema}.call_transcripts').count()} rows")

# COMMAND ----------

# Generate and write plans
plans_pdf = generate_plans()
spark.createDataFrame(plans_pdf).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.plans")
print(f"✓ plans table: {spark.table(f'{catalog}.{schema}.plans').count()} rows")

# COMMAND ----------

# Generate and write policies
policies_pdf = generate_policies()
spark.createDataFrame(policies_pdf).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.policies")
print(f"✓ policies table: {spark.table(f'{catalog}.{schema}.policies').count()} rows")

# COMMAND ----------

# Generate and write product_knowledge (with CDC for vector search)
knowledge_pdf = generate_product_knowledge()
spark.createDataFrame(knowledge_pdf).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.product_knowledge")
print(f"✓ product_knowledge table: {spark.table(f'{catalog}.{schema}.product_knowledge').count()} rows")

# COMMAND ----------

# Enable Change Data Feed on product_knowledge for delta sync index
spark.sql(f"ALTER TABLE {catalog}.{schema}.product_knowledge SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("✓ CDC enabled on product_knowledge")

# COMMAND ----------

# Generate and write churn_labels
labels_pdf = generate_churn_labels(customers_pdf, seed=42)
spark.createDataFrame(labels_pdf).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.churn_labels")
print(f"✓ churn_labels table: {spark.table(f'{catalog}.{schema}.churn_labels').count()} rows")
print(f"  Split distribution:\n{labels_pdf['split'].value_counts().to_string()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2b. Generate Sample PDFs (for `ai_parse_document` demo)
# MAGIC
# MAGIC Renders a small portfolio of telecom documents (5 bills, 1 contract, 1 complaint
# MAGIC letter) into a Unity Catalog Volume. These power the document intelligence section
# MAGIC of `module_2_genai/01_ai_functions.py`.

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {documents_volume}")
print(f"✓ Volume {documents_volume} ready at {documents_volume_path}")

# COMMAND ----------

pdf_files = generate_sample_pdfs(customers_pdf, plans_pdf, seed=42)
for filename, content in pdf_files:
    out_path = f"{documents_volume_path}/{filename}"
    with open(out_path, "wb") as f:
        f.write(content)
print(f"✓ Wrote {len(pdf_files)} PDFs to {documents_volume_path}/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Vector Search Endpoint & Index

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
)

w = WorkspaceClient()

# COMMAND ----------

# Create vector search endpoint (if it doesn't already exist)
try:
    w.vector_search_endpoints.create_endpoint(
        name=vector_search_endpoint,
        endpoint_type=EndpointType.STANDARD,
    )
    print(f"✓ Creating vector search endpoint '{vector_search_endpoint}'...")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Vector search endpoint '{vector_search_endpoint}' already exists")
    else:
        raise e

# COMMAND ----------

# Wait for endpoint to be ready
import time

endpoint = w.vector_search_endpoints.get_endpoint(vector_search_endpoint)
while str(endpoint.endpoint_status.state) != "EndpointStatusState.ONLINE":
    print(f"  Endpoint status: {endpoint.endpoint_status.state} — waiting 30s...")
    time.sleep(30)
    endpoint = w.vector_search_endpoints.get_endpoint(vector_search_endpoint)
print(f"✓ Vector search endpoint '{vector_search_endpoint}' is ONLINE")

# COMMAND ----------

# Create delta sync vector index on product_knowledge
source_table = f"{catalog}.{schema}.product_knowledge"
try:
    w.vector_search_indexes.create_index(
        name=product_knowledge_index,
        endpoint_name=vector_search_endpoint,
        primary_key="article_id",
        index_type=PipelineType.TRIGGERED,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name=embedding_endpoint,
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
        ),
    )
    print(f"✓ Creating vector search index '{product_knowledge_index}'...")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Vector search index '{product_knowledge_index}' already exists")
    else:
        raise e

# COMMAND ----------

# Sync the index
try:
    w.vector_search_indexes.sync_index(index_name=product_knowledge_index)
    print(f"✓ Triggered sync for '{product_knowledge_index}'")
except Exception as e:
    print(f"  Note: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Grant Permissions

# COMMAND ----------

spark.sql(f"GRANT USE CATALOG ON CATALOG {catalog} TO `account users`")
spark.sql(f"GRANT USE SCHEMA ON SCHEMA {catalog}.{schema} TO `account users`")

# Grant SELECT on all tables
tables = ["customers", "service_tickets", "call_transcripts", "plans", "policies", "product_knowledge", "churn_labels"]
for table in tables:
    spark.sql(f"GRANT SELECT ON TABLE {catalog}.{schema}.{table} TO `account users`")
    print(f"  ✓ SELECT granted on {table}")

# Grant READ on the documents volume
spark.sql(f"GRANT READ VOLUME ON VOLUME {documents_volume} TO `account users`")
print(f"  ✓ READ VOLUME granted on {documents_volume}")

# Grant EXECUTE on functions (will be created later in Module 2)
# These grants will be applied when the functions are created

print(f"\n✓ All permissions granted to `account users`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete
# MAGIC
# MAGIC All data tables, vector search index, and permissions are provisioned.
# MAGIC
# MAGIC | Resource | Status |
# MAGIC |----------|--------|
# MAGIC | Catalog & Schema | ✓ Created |
# MAGIC | customers (5,000 rows) | ✓ Written |
# MAGIC | service_tickets (15,000 rows) | ✓ Written |
# MAGIC | call_transcripts (3,000 rows) | ✓ Written |
# MAGIC | plans (15 rows) | ✓ Written |
# MAGIC | policies (8 rows) | ✓ Written |
# MAGIC | product_knowledge (50 rows) | ✓ Written |
# MAGIC | churn_labels (5,000 rows) | ✓ Written |
# MAGIC | documents volume (7 PDFs) | ✓ Written |
# MAGIC | Vector Search Endpoint | ✓ Online |
# MAGIC | Vector Search Index | ✓ Synced |
# MAGIC | Permissions | ✓ Granted |
# MAGIC
# MAGIC **Next**: Participants can now work through Module 1 and Module 2 notebooks sequentially.
