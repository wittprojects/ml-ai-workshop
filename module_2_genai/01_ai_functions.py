# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 01 — AI Functions
# MAGIC
# MAGIC **Time**: ~8 min
# MAGIC
# MAGIC Databricks AI Functions bring LLM capabilities directly into SQL. No infrastructure, no prompts engineering frameworks — just SQL.
# MAGIC
# MAGIC | Notebook | Topic |
# MAGIC |----------|-------|
# MAGIC | **01 AI Functions** | FMAPI, ai_query(), ai_extract() |
# MAGIC | 02 Create Tools | UC functions as agent tools |
# MAGIC | 03 Build Agent | LangGraph retention agent + MCP |
# MAGIC | 04 MLflow Tracing | Agent observability |
# MAGIC | 05 Agent Eval | mlflow.genai.evaluate() |
# MAGIC | 06 Deploy to Apps | Databricks Apps deployment |
# MAGIC | 07 AI Gateway | Routing, guardrails, governance |

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Foundation Model APIs (FMAPI)
# MAGIC
# MAGIC Databricks hosts popular LLMs as pay-per-token endpoints. No provisioning required.
# MAGIC
# MAGIC Our workshop uses:
# MAGIC - **Chat**: `databricks-claude-sonnet-4-6`
# MAGIC - **Embeddings**: `databricks-gte-large-en`

# COMMAND ----------

# Quick test of FMAPI via Python SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()
response = w.serving_endpoints.query(
    name=llm_endpoint,
    messages=[ChatMessage(role=ChatMessageRole.USER, content="In one sentence, what causes customer churn in telecom?")],
    max_tokens=100,
)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `ai_query()` — LLM Queries in SQL
# MAGIC
# MAGIC Call any FMAPI endpoint with a custom prompt. Powerful for classification, summarization, and enrichment.

# COMMAND ----------

# DBTITLE 1,Classification
# MAGIC %sql
# MAGIC SELECT
# MAGIC   transcript_id,
# MAGIC   SUBSTRING(transcript_text, 1, 80) as preview,
# MAGIC   ai_query(
# MAGIC     'databricks-claude-sonnet-4-6',
# MAGIC     CONCAT(
# MAGIC       'Classify this telecom call transcript into exactly one category: ',
# MAGIC       'cancellation, billing_dispute, technical_issue, upgrade_inquiry, general_inquiry. ',
# MAGIC       'Return only the category name.\n\nTranscript: ', transcript_text
# MAGIC     )
# MAGIC   ) as intent
# MAGIC FROM call_transcripts
# MAGIC LIMIT 5

# COMMAND ----------

# DBTITLE 1,Summarize Call Transcripts
# MAGIC %sql
# MAGIC SELECT
# MAGIC   transcript_id,
# MAGIC   ai_query(
# MAGIC     'databricks-claude-sonnet-4-6',
# MAGIC     CONCAT(
# MAGIC       'Summarize this telecom customer service call in one sentence. ',
# MAGIC       'Include the customer issue and resolution.\n\n',
# MAGIC       transcript_text
# MAGIC     )
# MAGIC   ) as summary
# MAGIC FROM call_transcripts
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explain Churn Risk
# MAGIC
# MAGIC Combine ML predictions with LLM reasoning — the ML model says *what*, the LLM explains *why*.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   c.customer_id,
# MAGIC   c.name,
# MAGIC   c.churn,
# MAGIC   ai_query(
# MAGIC     'databricks-claude-sonnet-4-6',
# MAGIC     CONCAT(
# MAGIC       'This telecom customer has the following profile. Explain in 2-3 sentences why they might be at risk of churning.\n\n',
# MAGIC       'Tenure: ', c.tenure_months, ' months\n',
# MAGIC       'Contract: ', c.contract_type, '\n',
# MAGIC       'Monthly charges: $', c.monthly_charges, '\n',
# MAGIC       'Internet: ', c.internet_service, '\n',
# MAGIC       'Payment: ', c.payment_method, '\n',
# MAGIC       'Churn status: ', c.churn
# MAGIC     )
# MAGIC   ) as churn_explanation
# MAGIC FROM customers c
# MAGIC WHERE c.churn = 'Yes'
# MAGIC LIMIT 3

# COMMAND ----------

# MAGIC %md
# MAGIC ## `ai_sentiment()` — Built-in Sentiment Analysis
# MAGIC
# MAGIC The simplest AI Function: no model name, no prompt — just pass text and get sentiment.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   transcript_id,
# MAGIC   SUBSTRING(transcript_text, 1, 100) as transcript_preview,
# MAGIC   ai_analyze_sentiment(transcript_text) as sentiment
# MAGIC FROM call_transcripts
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## `ai_extract()` — Structured Extraction
# MAGIC
# MAGIC Extract structured fields from unstructured text. Returns a struct that can be unpacked into columns.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   transcript_id,
# MAGIC   ai_extract(
# MAGIC     transcript_text,
# MAGIC     ARRAY('competitor_mentioned', 'key_complaint', 'customer_sentiment', 'resolution_outcome')
# MAGIC   ) as extracted
# MAGIC FROM call_transcripts
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Unpack the extracted fields into separate columns
# MAGIC SELECT
# MAGIC   transcript_id,
# MAGIC   extracted.competitor_mentioned,
# MAGIC   extracted.key_complaint,
# MAGIC   extracted.customer_sentiment,
# MAGIC   extracted.resolution_outcome
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     transcript_id,
# MAGIC     ai_extract(
# MAGIC       transcript_text,
# MAGIC       ARRAY('competitor_mentioned', 'key_complaint', 'customer_sentiment', 'resolution_outcome')
# MAGIC     ) as extracted
# MAGIC   FROM call_transcripts
# MAGIC   LIMIT 5
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract from Service Tickets

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ticket_id,
# MAGIC   category,
# MAGIC   description,
# MAGIC   ai_extract(
# MAGIC     description,
# MAGIC     ARRAY('product_referenced', 'resolution_type', 'urgency_level')
# MAGIC   ) as extracted
# MAGIC FROM service_tickets
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC AI Functions bring LLM capabilities to SQL with zero infrastructure:
# MAGIC
# MAGIC | Function | Use Case | Example |
# MAGIC |----------|----------|---------|
# MAGIC | `ai_sentiment()` | Built-in sentiment | No config needed |
# MAGIC | `ai_query()` | Custom LLM prompts | Classification, summarization, explanation |
# MAGIC | `ai_extract()` | Structured extraction | Pull fields from free text |
# MAGIC
# MAGIC These work in **dashboards, pipelines, and scheduled queries** — not just notebooks.
# MAGIC
# MAGIC **Next**: [02 Create Tools →](./02_create_tools)
