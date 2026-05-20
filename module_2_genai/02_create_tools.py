# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 02 — Create Agent Tools
# MAGIC
# MAGIC **Time**: ~10 min
# MAGIC
# MAGIC We'll create **Unity Catalog functions** that serve as tools for our retention agent.
# MAGIC
# MAGIC Key concepts:
# MAGIC - UC SQL functions as agent tools
# MAGIC - UC Python functions via `DatabricksFunctionClient`
# MAGIC - VectorSearchRetrieverTool for RAG
# MAGIC - Parameterized SQL with `IDENTIFIER()`

# COMMAND ----------

# MAGIC %pip install databricks-langchain databricks-sdk>=0.50.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 1: `get_latest_ticket()`
# MAGIC
# MAGIC Retrieve the most recent escalated service ticket.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_latest_ticket()
RETURNS TABLE(ticket_id STRING, customer_id STRING, created_date TIMESTAMP, category STRING, priority STRING, status STRING, description STRING)
LANGUAGE SQL
COMMENT 'Returns the most recent escalated or high-priority service ticket. Use this to find the next customer to help.'
RETURN
  SELECT ticket_id, customer_id, created_date, category, priority, status, description
  FROM IDENTIFIER(:catalog_name || '.' || :schema_name || '.' || 'service_tickets')
  WHERE priority IN ('high', 'critical') AND status IN ('open', 'escalated')
  ORDER BY created_date DESC
  LIMIT 1
""", args={"catalog_name": catalog, "schema_name": schema})

print(f"✓ Created {catalog}.{schema}.get_latest_ticket()")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalog}.{schema}.get_latest_ticket()"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 2: `get_customer_profile()`

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_customer_profile(customer_id_param STRING)
RETURNS TABLE(customer_id STRING, name STRING, tenure_months INT, contract_type STRING, monthly_charges DOUBLE, total_charges DOUBLE, internet_service STRING, phone_service STRING, payment_method STRING, senior_citizen INT, partner STRING, dependents STRING)
LANGUAGE SQL
COMMENT 'Looks up a customer profile by customer_id. Returns demographics, plan details, and service information.'
RETURN
  SELECT customer_id, name, tenure_months, contract_type, monthly_charges, total_charges,
         internet_service, phone_service, payment_method, senior_citizen, partner, dependents
  FROM IDENTIFIER(:catalog_name || '.' || :schema_name || '.' || 'customers')
  WHERE customer_id = customer_id_param
""", args={"catalog_name": catalog, "schema_name": schema})

print(f"✓ Created {catalog}.{schema}.get_customer_profile()")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalog}.{schema}.get_customer_profile('CUST-00001')"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 3: `get_ticket_history()`

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_ticket_history(customer_id_param STRING)
RETURNS TABLE(category STRING, ticket_count BIGINT, latest_ticket TIMESTAMP, example_description STRING)
LANGUAGE SQL
COMMENT 'Returns ticket history summary for a customer grouped by category, including count and most recent ticket per category.'
RETURN
  SELECT category,
         COUNT(*) as ticket_count,
         MAX(created_date) as latest_ticket,
         FIRST(description) as example_description
  FROM IDENTIFIER(:catalog_name || '.' || :schema_name || '.' || 'service_tickets')
  WHERE customer_id = customer_id_param
  GROUP BY category
  ORDER BY ticket_count DESC
""", args={"catalog_name": catalog, "schema_name": schema})

print(f"✓ Created {catalog}.{schema}.get_ticket_history()")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalog}.{schema}.get_ticket_history('CUST-00001')"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 4: `get_retention_policy()`

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_retention_policy()
RETURNS TABLE(policy_name STRING, category STRING, policy_text STRING)
LANGUAGE SQL
COMMENT 'Returns company retention and cancellation policies. Use this to understand what offers and actions are available for customer retention.'
RETURN
  SELECT policy_name, category, policy_text
  FROM IDENTIFIER(:catalog_name || '.' || :schema_name || '.' || 'policies')
  WHERE category IN ('retention', 'cancellation')
""", args={"catalog_name": catalog, "schema_name": schema})

print(f"✓ Created {catalog}.{schema}.get_retention_policy()")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalog}.{schema}.get_retention_policy()"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 5: `get_churn_risk()`
# MAGIC
# MAGIC Looks up the pre-computed churn probability for a customer from the batch-scored
# MAGIC `churn_predictions` table built in Module 1, notebook 05.
# MAGIC
# MAGIC **Standalone fallback:** if Module 1 hasn't been run, we synthesize a stand-in
# MAGIC `churn_predictions` table from `customers` so this module works on its own.

# COMMAND ----------

if not spark.catalog.tableExists(predictions_table_name):
    print(f"⚠ {predictions_table_name} not found — Module 1 wasn't run.")
    print(f"  Generating a synthetic stand-in so the agent's churn-risk tool still works.")
    spark.sql(f"""
        CREATE OR REPLACE TABLE {predictions_table_name} AS
        SELECT
          customer_id,
          CAST(LEAST(0.99, GREATEST(0.01,
            0.15
            + CASE WHEN contract_type = 'Month-to-month' THEN 0.35 ELSE 0.0 END
            + CASE WHEN tenure_months < 12 THEN 0.20 ELSE 0.0 END
            + (rand(42) - 0.5) * 0.30
          )) AS DOUBLE) AS churn_probability,
          CURRENT_TIMESTAMP() AS scored_at
        FROM {catalog}.{schema}.customers
    """)
    n = spark.table(predictions_table_name).count()
    print(f"✓ Wrote synthetic {predictions_table_name} ({n} rows)")
else:
    print(f"✓ Using {predictions_table_name} from Module 1")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_churn_risk(customer_id_param STRING)
RETURNS DOUBLE
LANGUAGE SQL
COMMENT 'Returns the pre-computed churn probability (0.0-1.0) for the given customer_id from the batch predictions table.'
RETURN
  SELECT MAX(churn_probability)
  FROM {predictions_table_name}
  WHERE customer_id = customer_id_param
""")
print(f"✓ Created {catalog}.{schema}.get_churn_risk()")

# COMMAND ----------

# Test against a customer that exists in the predictions table.
sample_customer_id = spark.table(predictions_table_name).limit(1).collect()[0]["customer_id"]
display(spark.sql(
    f"SELECT '{sample_customer_id}' as customer_id, "
    f"{catalog}.{schema}.get_churn_risk('{sample_customer_id}') as churn_risk"
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool 6: Vector Search Retriever
# MAGIC
# MAGIC The vector search endpoint and index on `product_knowledge` were created during setup.
# MAGIC We wrap it as a tool using `VectorSearchRetrieverTool`.

# COMMAND ----------

from databricks_langchain import VectorSearchRetrieverTool

product_search_tool = VectorSearchRetrieverTool(
    index_name=product_knowledge_index,
    tool_name="search_knowledge_base",
    tool_description="Search the product knowledge base for troubleshooting guides, plan comparisons, feature information, and FAQs. Use this when the customer has a question about products, services, or technical issues.",
    columns=["article_id", "title", "content", "category"],
    num_results=3,
)

# Test the tool
results = product_search_tool.invoke("How to troubleshoot slow internet speeds?")
print(results[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We created 6 tools for the retention agent:
# MAGIC
# MAGIC | Tool | Type | Purpose |
# MAGIC |------|------|---------|
# MAGIC | `get_latest_ticket()` | UC SQL | Find next customer to help |
# MAGIC | `get_customer_profile()` | UC SQL | Look up customer details |
# MAGIC | `get_ticket_history()` | UC SQL | Review support history |
# MAGIC | `get_retention_policy()` | UC SQL | Check available retention offers |
# MAGIC | `get_churn_risk()` | UC SQL + ai_query | Get ML churn prediction |
# MAGIC | `search_knowledge_base` | Vector Search | RAG for product info |
# MAGIC
# MAGIC All UC functions are registered with descriptions that the agent uses to decide when to call each tool.
# MAGIC
# MAGIC **Next**: [03 Agent Eval →](./03_agent_eval)
