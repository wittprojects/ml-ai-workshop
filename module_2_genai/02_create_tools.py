# Databricks notebook source
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

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %pip install databricks-langchain databricks-sdk==0.50.0 -q
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
# MAGIC This calls the Module 1 serving endpoint to get a real-time churn prediction!

# COMMAND ----------

try:
    spark.sql(f"""
    CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_churn_risk(customer_id_param STRING)
    RETURNS STRING
    LANGUAGE SQL
    COMMENT 'Calls the churn prediction model serving endpoint and returns the churn risk prediction for a given customer_id.'
    RETURN
      SELECT ai_query(
        '{churn_model_serving_endpoint}',
        named_struct('customer_id', customer_id_param)
      )
    """)
    print(f"✓ Created {catalog}.{schema}.get_churn_risk()")
except Exception as e:
    print(f"Note: get_churn_risk creation skipped — serving endpoint '{churn_model_serving_endpoint}' not active yet")
    print(f"  Run Module 1 notebook 05 first, then re-run this cell")

# COMMAND ----------

# Test (requires the serving endpoint from Module 1 to be active)
try:
    display(spark.sql(f"SELECT {catalog}.{schema}.get_churn_risk('CUST-00001') as churn_risk"))
except Exception as e:
    print(f"Note: get_churn_risk test skipped — serving endpoint not active")

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
# MAGIC ## Grant Execute Permissions on Functions

# COMMAND ----------

functions = [
    "get_latest_ticket",
    "get_customer_profile",
    "get_ticket_history",
    "get_retention_policy",
    "get_churn_risk",
]

for func in functions:
    try:
        spark.sql(f"GRANT EXECUTE ON FUNCTION {catalog}.{schema}.{func} TO `account users`")
        print(f"  ✓ EXECUTE granted on {func}")
    except Exception as e:
        print(f"  ⊘ Skipped {func} (not yet created)")

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
# MAGIC **Next**: [03 Build Agent →](./03_build_agent)
