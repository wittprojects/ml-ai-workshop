# Telecom Churn ML + GenAI Workshop

A self-service, hands-on Databricks workshop that builds an **end-to-end customer
retention solution** on a single synthetic telecom dataset. You work through two
modules in order:

- **Module 1 — Machine Learning**: build a classical ML churn-prediction pipeline
  — feature engineering, training, the model registry, serving, batch inference,
  and monitoring.
- **Module 2 — GenAI & Agents**: build a GenAI retention agent on the same data —
  AI functions, Unity Catalog tools, agent evaluation, a Genie room for natural-
  language analytics, and an optional Databricks App deployment.

The two modules share one Unity Catalog schema, so Module 2's agent can call the
churn model that Module 1 deploys. Plan ~2 hours for both.

Everything runs from notebooks — no live instructor required. Just configure,
run setup once, and work through the modules at your own pace.

---

## Prerequisites

- A **Databricks workspace** with **Unity Catalog** and **serverless** compute.
  The notebooks pin the serverless environment (`environment_version = "5"`).
- **Permission to create a schema** in some catalog (and to create the catalog
  itself if you don't reuse an existing one). Setup also grants access to
  `account users`, so the assets are usable by everyone in your workspace.
- Access to the **Foundation Model API** endpoints used by Module 2:
  `databricks-claude-sonnet-4-6` (LLM) and `databricks-gte-large-en` (embeddings).
  Override these in the config if your workspace exposes different endpoint names.
- A **Pro or Serverless SQL warehouse** (used by the Genie room in Module 2).
- The **Databricks CLI** is only needed for the *optional* app deployment at the
  end of Module 2.

---

## Quick start

1. **Configure.** Open `_resources/00_config.py` and set `catalog` and `schema`
   to a Unity Catalog location you can write to. Defaults are `main` /
   `churn_workshop`. Adjust the endpoint names in the same file if needed.

2. **Run setup once.** Run `_resources/01_setup.py` top to bottom. It creates the
   catalog/schema, generates all synthetic tables, builds the Vector Search
   endpoint and index, and grants access to `account users`. (Takes a few minutes
   — the Vector Search index sync is the slow part.)

3. **Run Module 1**, notebooks `01` → `06` in order.

4. **Run Module 2**, notebooks `01` → `05` in order.

5. *(Optional)* Run `_resources/99_verify_grants.py` as a regular workspace user
   to confirm everything is accessible, and `_resources/99_cleanup.py` to tear it
   all down and start fresh.

Every notebook loads the shared config with `%run ../_resources/00_config` at the
top, so you only set the catalog/schema in one place.

---

## Repository structure

```
ml-ai-workshop/
├── _resources/
│   ├── 00_config.py          # Catalog/schema/endpoint config + grant helpers (edit this first)
│   ├── 01_setup.py           # One-time provisioning: data, vector index, grants
│   ├── data_generators.py    # Synthetic telecom data generators (called by setup)
│   ├── 99_verify_grants.py   # Verify a non-admin user can access every asset
│   └── 99_cleanup.py         # Destructive reset (drops the schema + endpoints)
├── module_1_machine_learning/
│   ├── 01_overview.py
│   ├── 02_feature_engineering.py
│   ├── 03_train_and_register_model.py
│   ├── 03a_genie_code_alternative.py
│   ├── 04_model_serving.py
│   ├── 05_batch_inference.py
│   └── 06_monitoring.py
└── module_2_genai/
    ├── 01_ai_functions.py
    ├── 02_create_tools.py
    ├── 03_agent_eval.py
    ├── 04_genie_room.py
    └── 05_build_and_deploy_agent.py
```

---

## Module 1 — Machine Learning

Run in order. Each notebook builds on the previous one's outputs.

| Notebook | What it does |
|----------|--------------|
| `01_overview.py` | Introduces the use case and explores the `customers` and `service_tickets` data (churn rate, key signals). |
| `02_feature_engineering.py` | Builds a Unity Catalog **feature table** joining customer profiles with aggregated ticket features; adds an on-demand feature function. |
| `03_train_and_register_model.py` | Trains a **LightGBM** classifier with **Optuna** tuning, tracked by **MLflow**, and registers it to Unity Catalog with a Champion alias. |
| `03a_genie_code_alternative.py` | Optional: a guided walkthrough for reproducing notebook `03` using the Genie Code UI / a coding agent. |
| `04_model_serving.py` | Deploys the Champion model to a **Model Serving** endpoint (scale-to-zero) and calls it over REST. |
| `05_batch_inference.py` | Scores the test split and writes a `churn_predictions` table for downstream monitoring. |
| `06_monitoring.py` | Sets up **Lakehouse Monitoring** on the predictions table — baseline, drift, and quality metrics. |

## Module 2 — GenAI & Agents

Run in order. Uses the data from Module 1 and calls the serving endpoint it deployed.

| Notebook | What it does |
|----------|--------------|
| `01_ai_functions.py` | Uses the **Foundation Model API** (`ai_query`, `ai_extract`) to classify, summarize, and extract structure from call transcripts. |
| `02_create_tools.py` | Creates **Unity Catalog functions** as agent tools (customer profile, ticket history, retention policy, churn risk via the Module 1 endpoint) plus a vector-search retriever tool. |
| `03_agent_eval.py` | Builds a retention agent over those tools, instruments it with **MLflow tracing**, and evaluates it with `mlflow.genai.evaluate`. |
| `04_genie_room.py` | Defines metric views (churn KPIs) and creates a **Genie room** for natural-language analytics. |
| `05_build_and_deploy_agent.py` | Walks through testing the agent in the AI Playground, then *optionally* deploying it as a **Databricks App**. |

> **Note on the App deploy:** Notebook `05` describes an `agent_app/` directory
> (FastAPI app, agent module, `app.yaml`, `requirements.txt`). Those files are a
> template you create if you want to deploy — they are **not** included in this
> repo. The rest of Module 2 runs without them.

---

## Configuration reference

All editable in `_resources/00_config.py`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `catalog` | `main` | Unity Catalog catalog for all workshop assets. |
| `schema` | `churn_workshop` | Schema within the catalog. |
| `llm_endpoint` | `databricks-claude-sonnet-4-6` | Foundation Model API LLM endpoint. |
| `embedding_endpoint` | `databricks-gte-large-en` | Embedding endpoint for Vector Search. |
| `vector_search_endpoint` | `workshop_vs_endpoint` | Vector Search endpoint name. |
| `churn_model_serving_endpoint` | `workshop-churn-model` | Model Serving endpoint for the churn model. |

Table, model, volume, and index names are **derived** from `catalog`/`schema`
(e.g. `churn_feature_table`, `churn_model`, `churn_predictions`,
`product_knowledge_index`) — you don't set those individually.

---

## Data model

`01_setup.py` generates a self-contained synthetic telecom dataset:

| Table | Rows | Contents |
|-------|------|----------|
| `customers` | ~5,000 | Customer profiles, demographics, and the churn label. |
| `service_tickets` | ~15,000 | Support tickets per customer. |
| `call_transcripts` | ~3,000 | Call-center transcripts (used by Module 2 AI functions). |
| `plans` | 15 | Plan catalog. |
| `policies` | 8 | Cancellation / retention / billing policies. |
| `product_knowledge` | 50 | Troubleshooting / FAQ articles (indexed for vector search). |
| `churn_labels` | ~5,000 | Target variable with train / validation / test splits. |

Setup also creates a `documents` volume with sample PDFs (bills, contracts,
complaint letters) for document-processing demos.

---

## Multi-user access, verification, and cleanup

- **Access model:** setup grants schema-level UC privileges (`USE`, `SELECT`,
  `EXECUTE`, `READ/WRITE VOLUME`) to `account users`, which cascade to every table,
  view, function, model, and volume — including ones the module notebooks create
  later. Non-UC objects (serving, vector search, Genie) are granted as they're
  created. So once one person runs setup, anyone in the workspace can run the
  modules against the shared schema.
- **Verify:** `_resources/99_verify_grants.py` exercises every asset as the
  calling identity. Run it as a regular (non-admin) user to confirm the grants
  are correct — it fails loudly if anything is inaccessible.
- **Cleanup:** `_resources/99_cleanup.py` removes everything the workshop creates
  (drops the schema CASCADE, deletes the serving/vector-search endpoints, Genie
  space, monitor, and your own MLflow experiments). It requires typing `DELETE`
  into a confirmation widget. The catalog itself is **not** dropped, and other
  users' MLflow experiments are left untouched.
