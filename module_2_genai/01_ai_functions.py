# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 01 — AI Functions
# MAGIC
# MAGIC **Time**: ~13 min
# MAGIC
# MAGIC Databricks AI Functions bring LLM capabilities directly into SQL. No infrastructure, no prompts engineering frameworks — just SQL.
# MAGIC
# MAGIC | Notebook | Topic |
# MAGIC |----------|-------|
# MAGIC | **01 AI Functions** | FMAPI, ai_query(), ai_extract(), ai_parse_document() |
# MAGIC | 02 Create Tools | UC functions as agent tools |
# MAGIC | 03 Agent Eval | mlflow.genai.evaluate() |
# MAGIC | 04 Metric Views & Genie Room | Governed metrics + NL analytics |
# MAGIC | 05 Build & Deploy Agent | AI Playground + Databricks Apps |

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
# MAGIC ## Document Intelligence: `ai_parse_document` → `ai_extract` → `ai_classify`
# MAGIC
# MAGIC The examples above all assume the input is *already* clean text. In the real world,
# MAGIC most enterprise documents arrive as **PDFs, scanned images, or DOCX files** — bills,
# MAGIC contracts, complaint letters. Historically, getting these into a SQL-ready shape meant
# MAGIC stitching together OCR services, layout detection APIs, and figure captioning models.
# MAGIC
# MAGIC The new **`ai_parse_document`** function (GA, schema v2.0) collapses that whole stack
# MAGIC into one SQL call. It returns a `VARIANT` containing text, tables (preserved as
# MAGIC structures, not flattened), figure descriptions, and bounding-box metadata.
# MAGIC
# MAGIC The big upgrade is the **handoff**: the parsed `VARIANT` is the native input to
# MAGIC `ai_extract`, `ai_classify`, and `ai_query` — no glue code, no manual JSON wrangling.
# MAGIC
# MAGIC > **Requirements**: Serverless env v3+ or DBR 17.3+. Max 500 pages / 100 MB per file.

# COMMAND ----------

# DBTITLE 1,Read PDFs from the UC Volume
documents_df = (
    spark.read.format("binaryFile")
    .load(documents_volume_path)
    .select("path", "length", "content")
)
print(f"Loaded {documents_df.count()} documents from {documents_volume_path}")
display(documents_df.select("path", "length"))

# COMMAND ----------

# DBTITLE 1,ai_parse_document — single SQL call replaces an OCR + layout stack
from pyspark.sql.functions import expr

parsed_df = documents_df.withColumn(
    "parsed",
    expr("ai_parse_document(content, map('version', '2.0'))"),
)

# Materialize so downstream cells reuse the parse output instead of re-running it
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.parsed_documents")
(parsed_df
    .select("path", "parsed")
    .write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.parsed_documents"))

display(spark.table(f"{catalog}.{schema}.parsed_documents").limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC The `parsed` column is a versioned `VARIANT` with this shape:
# MAGIC
# MAGIC ```text
# MAGIC parsed.document.pages     — page id + image_uri
# MAGIC parsed.document.elements  — typed elements: text, table, figure, title, page_header, ...
# MAGIC parsed.metadata           — file path, name, size, modification time
# MAGIC parsed.error_status       — non-empty only if a page failed to parse
# MAGIC ```
# MAGIC
# MAGIC Tables come back **structured**, not flattened to raw text — which is the difference
# MAGIC between an LLM extraction that hallucinates a total and one that quotes the actual
# MAGIC line item.

# COMMAND ----------

# DBTITLE 1,Inspect element types — tables are preserved as structured elements
spark.sql(f"""
    SELECT
        element:type::string  AS element_type,
        COUNT(*)              AS count
    FROM {catalog}.{schema}.parsed_documents,
         LATERAL view explode(from_json(to_json(parsed:document:elements), 'array<variant>')) AS t AS element
    GROUP BY 1
    ORDER BY 2 DESC
""").display()

# COMMAND ----------

# DBTITLE 1,ai_extract — pull structured fields directly from the parsed VARIANT
# MAGIC %md
# MAGIC The reworked `ai_extract` (PuPr) takes the parsed `VARIANT` directly. The new
# MAGIC `instructions` parameter lets you tell the extractor what kind of document it's
# MAGIC looking at, which dramatically improves accuracy on telecom-specific fields like
# MAGIC `account_number`, `billing_period`, and `total_due`.

# COMMAND ----------

extracted_df = spark.sql(f"""
    SELECT
        path,
        ai_extract(
            parsed,
            ARRAY('account_number', 'billing_period', 'plan_name', 'total_due',
                  'payment_due_date', 'overage_charges'),
            MAP('instructions',
                'These are telecom monthly billing statements from Northstar Telecom. ' ||
                'total_due and overage_charges are USD amounts. payment_due_date is the date the customer must pay by.')
        ) AS bill_fields
    FROM {catalog}.{schema}.parsed_documents
    WHERE path LIKE '%/bill_%'
""")

display(extracted_df.select(
    "path",
    "bill_fields.account_number",
    "bill_fields.billing_period",
    "bill_fields.plan_name",
    "bill_fields.total_due",
    "bill_fields.payment_due_date",
    "bill_fields.overage_charges",
))

# COMMAND ----------

# DBTITLE 1,ai_classify — route documents by type
classified_df = spark.sql(f"""
    SELECT
        regexp_extract(path, '/([^/]+)$', 1) AS filename,
        ai_classify(
            parsed,
            ARRAY('bill', 'contract', 'complaint_letter')
        ) AS doc_type
    FROM {catalog}.{schema}.parsed_documents
""")
display(classified_df)

# COMMAND ----------

# DBTITLE 1,ai_query — free-form Q&A over the same parsed output
# MAGIC %md
# MAGIC The same `VARIANT` flows into `ai_query` for narrative answers. Here, we summarize the
# MAGIC dispute reason from the complaint letter — the LLM reads the structured parse, not
# MAGIC raw OCR noise.

# COMMAND ----------

spark.sql(f"""
    SELECT
        regexp_extract(path, '/([^/]+)$', 1) AS filename,
        ai_query(
            '{llm_endpoint}',
            CONCAT(
                'Read this customer-submitted document. In one sentence, state the customer''s ',
                'primary complaint and the resolution they are requesting. Document content: ',
                to_json(parsed:document:elements)
            )
        ) AS dispute_summary
    FROM {catalog}.{schema}.parsed_documents
    WHERE path LIKE '%/complaint_%'
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC AI Functions bring LLM capabilities to SQL with zero infrastructure:
# MAGIC
# MAGIC | Function | Use Case | Example |
# MAGIC |----------|----------|---------|
# MAGIC | `ai_analyze_sentiment()` | Built-in sentiment | No config needed |
# MAGIC | `ai_query()` | Custom LLM prompts | Classification, summarization, explanation |
# MAGIC | `ai_extract()` | Structured extraction | Pull fields from free text **or parsed PDFs** |
# MAGIC | `ai_classify()` | Bucket routing | Route docs by type |
# MAGIC | `ai_parse_document()` | PDF / image → `VARIANT` | Replaces OCR + layout + caption stack |
# MAGIC
# MAGIC The big shift in 2026: `ai_parse_document` outputs a `VARIANT` that flows **natively**
# MAGIC into `ai_extract`, `ai_classify`, and `ai_query`. One SQL chain takes you from a raw
# MAGIC PDF in a UC Volume to governed, structured columns in a Delta table — no glue code.
# MAGIC
# MAGIC These work in **dashboards, pipelines, and scheduled queries** — not just notebooks.
# MAGIC
# MAGIC **Next**: [02 Create Tools →](./02_create_tools)
