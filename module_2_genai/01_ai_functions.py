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
# MAGIC The new **`ai_parse_document`** function collapses that whole stack
# MAGIC into one SQL call. It returns a `VARIANT` containing text, tables, figure descriptions, and bounding-box metadata.
# MAGIC
# MAGIC The big upgrade is the **handoff**: the parsed `VARIANT` is the native input to
# MAGIC `ai_extract`, `ai_classify`, and `ai_query` — no glue code, no manual JSON wrangling.

# COMMAND ----------

# DBTITLE 1,Read PDFs from the UC Volume
documents_df = (
    spark.read.format("binaryFile")
    .load(documents_source_path)
    .select("path", "length", "content")
)
print(f"Loaded {documents_df.count()} documents from {documents_source_path}")
display(documents_df.select("path", "length"))

# COMMAND ----------

# DBTITLE 1,ai_parse_document — single SQL call replaces an OCR + layout stack
# MAGIC %md
# MAGIC The `imageOutputPath` option asks the parser to render each page as a PNG into a UC
# MAGIC Volume. We reuse those PNGs further down to overlay bounding boxes for visual
# MAGIC debugging. `descriptionElementTypes='*'` enables AI-generated descriptions for
# MAGIC figure elements as well.

# COMMAND ----------

from pyspark.sql.functions import expr, lit

parsed_df = documents_df.withColumn(
    "parsed",
    expr(f"""ai_parse_document(
        content,
        map(
            'version', '2.0',
            'imageOutputPath', '{parsed_images_path}',
            'descriptionElementTypes', '*'
        )
    )"""),
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
    FROM {catalog}.{schema}.parsed_documents
    LATERAL VIEW explode(from_json(to_json(parsed:document:elements), 'array<variant>')) t AS element
    GROUP BY 1
    ORDER BY 2 DESC
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visual debugging — bounding-box overlay
# MAGIC
# MAGIC Because we passed `imageOutputPath`, the parser dropped a PNG of every page next to the
# MAGIC source PDFs. Each element in `parsed.document.elements` carries a `bbox` array with
# MAGIC pixel coordinates that map directly onto those PNGs. Overlaying them is the fastest
# MAGIC way to confirm what the model actually segmented — useful when an extraction looks
# MAGIC off and you need to figure out whether the parse or the prompt is at fault.

# COMMAND ----------

# DBTITLE 1,Overlay element bboxes on the rendered page
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Pull one parsed bill: page 0 image URI + the full elements array
row = spark.sql(f"""
    SELECT
        regexp_extract(path, '/([^/]+)$', 1)                AS filename,
        parsed:document:pages[0]:image_uri::string          AS page0_uri,
        to_json(parsed:document:elements)                   AS elements_json
    FROM {catalog}.{schema}.parsed_documents
    WHERE path LIKE '%/bill_%'
    ORDER BY path
    LIMIT 1
""").collect()[0]

img = Image.open(row.page0_uri)
elements = json.loads(row.elements_json)

# One color per element type so the overlay is self-explanatory
type_color = {
    "title":          "#d62728",
    "section_header": "#ff7f0e",
    "text":           "#1f77b4",
    "table":          "#2ca02c",
    "figure":         "#9467bd",
    "caption":        "#8c564b",
    "page_header":    "#7f7f7f",
    "page_footer":    "#7f7f7f",
    "page_number":    "#7f7f7f",
    "footnote":       "#bcbd22",
}

fig, ax = plt.subplots(figsize=(9, 12))
ax.imshow(img)
ax.set_axis_off()
ax.set_title(f"ai_parse_document — bounding boxes\n{row.filename} (page 1)", fontsize=11)

seen_types = set()
for el in elements:
    el_type = el.get("type", "text")
    color = type_color.get(el_type, "#000000")
    for bb in el.get("bbox") or []:
        if bb.get("page_id") != 0:
            continue
        x0, y0, x1, y1 = bb["coord"]
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.4, edgecolor=color, facecolor="none", alpha=0.85,
        ))
        seen_types.add(el_type)

legend_handles = [patches.Patch(edgecolor=type_color.get(t, "#000"), facecolor="none", label=t)
                  for t in sorted(seen_types)]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### `ai_extract` v2 — pull structured fields directly from the parsed VARIANT
# MAGIC
# MAGIC The reworked `ai_extract` (v2.1) accepts the parsed `VARIANT` directly. Two things
# MAGIC are different from the v1 call we used on raw text earlier:
# MAGIC
# MAGIC 1. The schema is a **JSON-encoded string** (`'["field_a", "field_b", ...]'`), not a SQL `ARRAY(...)`. That's how SQL routes the call to v2 instead of v1.
# MAGIC 2. The `instructions` option lets you describe the document so the extractor can disambiguate fields (e.g., distinguish `total_due` from `overage_charges`).
# MAGIC
# MAGIC v2 returns a `VARIANT` shaped `{"response": {field: value, ...}, "error_message": null}`, so we read fields as `bill_fields:response.field_name::string`.

# COMMAND ----------

# DBTITLE 1,ai_extract over the parsed VARIANT
extracted_df = spark.sql(f"""
    SELECT
        path,
        ai_extract(
            parsed,
            '["account_number", "billing_period", "plan_name", "total_due", "payment_due_date", "overage_charges"]',
            MAP(
                'version', '2.1',
                'instructions',
                'These are telecom monthly billing statements from Northstar Telecom. ' ||
                'total_due and overage_charges are USD amounts. payment_due_date is the date the customer must pay by.'
            )
        ) AS bill_fields
    FROM {catalog}.{schema}.parsed_documents
    WHERE path LIKE '%/bill_%'
""")

display(extracted_df.selectExpr(
    "path",
    "bill_fields:response.account_number::string  AS account_number",
    "bill_fields:response.billing_period::string  AS billing_period",
    "bill_fields:response.plan_name::string       AS plan_name",
    "bill_fields:response.total_due::string       AS total_due",
    "bill_fields:response.payment_due_date::string AS payment_due_date",
    "bill_fields:response.overage_charges::string AS overage_charges",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Citations + Confidence Scores (new in `ai_extract` v2.1)
# MAGIC
# MAGIC With two extra options the extractor returns, per field, **(a)** a confidence score
# MAGIC between 0 and 1 and **(b)** a citation pointing back to the source. For VARIANT
# MAGIC inputs from `ai_parse_document`, citations are bounding boxes — so we can draw them
# MAGIC straight onto the page image we already rendered.
# MAGIC
# MAGIC ```sql
# MAGIC MAP(
# MAGIC   'version', '2.1',
# MAGIC   'enableCitations', 'true',
# MAGIC   'enableConfidenceScores', 'true'
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC The output shape changes: `:response.<field>` is no longer a scalar but an object of
# MAGIC `{value, citation_ids, confidence_score}`. The bboxes themselves live under
# MAGIC `:metadata.citations[*]` and join back via `id`.
# MAGIC
# MAGIC >  Inference cost is higher when both flags are on
# MAGIC > as a secondary evaluator pass produces these signals. Best for high-stakes pipelines
# MAGIC > — financial services, healthcare, insurance — or for human-in-the-loop routing
# MAGIC > where you auto-approve high-confidence fields and queue the rest for review.

# COMMAND ----------

# DBTITLE 1,Re-run ai_extract on one bill with citations + confidence
audited_row = spark.sql(f"""
    SELECT
        regexp_extract(path, '/([^/]+)$', 1)        AS filename,
        parsed:document:pages[0]:image_uri::string  AS page0_uri,
        to_json(ai_extract(
            parsed,
            '["account_number", "billing_period", "plan_name", "total_due", "payment_due_date", "overage_charges"]',
            MAP(
                'version',                '2.1',
                'enableCitations',        'true',
                'enableConfidenceScores', 'true',
                'instructions',
                'These are telecom monthly billing statements from Northstar Telecom. ' ||
                'total_due and overage_charges are USD amounts. payment_due_date is the date the customer must pay by.'
            )
        )) AS audited
    FROM {catalog}.{schema}.parsed_documents
    WHERE path LIKE '%/bill_%'
    ORDER BY path
    LIMIT 1
""").collect()[0]

# Per-field summary: value + confidence + citation count
fields = ["account_number", "billing_period", "plan_name", "total_due",
          "payment_due_date", "overage_charges"]
audited_json = json.loads(audited_row.audited)
response = audited_json.get("response", {})

def _to_float(x):
    return float(x) if x is not None else None

field_rows = [
    {
        "field":       f,
        "value":       (response.get(f) or {}).get("value"),
        "confidence":  _to_float((response.get(f) or {}).get("confidence_score")),
        "citations":   len((response.get(f) or {}).get("citation_ids") or []),
    }
    for f in fields
]
display(spark.createDataFrame(field_rows))

# COMMAND ----------

# DBTITLE 1,Overlay citations on the page — color-coded by confidence
img = Image.open(audited_row.page0_uri)
citations_by_id = {c["id"]: c for c in audited_json.get("metadata", {}).get("citations", [])}

def conf_color(score):
    if score is None: return "#7f7f7f"
    if score >= 0.85: return "#2ca02c"   # high — green
    if score >= 0.60: return "#ff7f0e"   # medium — orange
    return "#d62728"                     # low — red

fig, ax = plt.subplots(figsize=(9, 12))
ax.imshow(img)
ax.set_axis_off()
ax.set_title(
    f"ai_extract citations + confidence\n{audited_row.filename} (page 1)",
    fontsize=11,
)

for fname in fields:
    f_obj = response.get(fname) or {}
    score = f_obj.get("confidence_score")
    color = conf_color(score)
    for cid in f_obj.get("citation_ids") or []:
        for bb in (citations_by_id.get(cid, {}).get("bbox") or []):
            if bb.get("page_id") != 0:
                continue
            x0, y0, x1, y1 = bb["coord"]
            ax.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2.0, edgecolor=color, facecolor="none", alpha=0.9,
            ))
            label = f"{fname} ({score:.2f})" if score is not None else fname
            ax.text(x0, max(0, y0 - 4), label, fontsize=7,
                    color="white", bbox=dict(facecolor=color, edgecolor="none", pad=1.5))

legend_handles = [
    patches.Patch(edgecolor="#2ca02c", facecolor="none", label="high   (≥ 0.85)"),
    patches.Patch(edgecolor="#ff7f0e", facecolor="none", label="medium (0.60–0.85)"),
    patches.Patch(edgecolor="#d62728", facecolor="none", label="low    (< 0.60)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,ai_classify v2 — route documents by type
# MAGIC %md
# MAGIC `ai_classify` v2 follows the same shape: VARIANT-friendly content + JSON-string label
# MAGIC list. Result is a VARIANT with `:response` holding the chosen label.

# COMMAND ----------

classified_df = spark.sql(f"""
    SELECT
        regexp_extract(path, '/([^/]+)$', 1) AS filename,
        ai_classify(
            parsed,
            '["bill", "contract", "complaint_letter"]',
            MAP('version', '2.0')
        ):response::string AS doc_type
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
# MAGIC Pair this with `ai_extract`'s new **citations + confidence scores** and you get an
# MAGIC auditable pipeline ready for FSI / healthcare / insurance — auto-approve the
# MAGIC high-confidence rows, route the rest to human review.
# MAGIC
# MAGIC These work in **dashboards, pipelines, and scheduled queries** — not just notebooks.
# MAGIC
# MAGIC **Next**: [02 Create Tools →](./02_create_tools)