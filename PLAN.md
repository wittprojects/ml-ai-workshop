# Data Science and AI Workshop - Implementation Plan

## Context

Building a combined 2-module workshop (~55 min each) for Databricks, merging and modernizing two existing archived workshops from the azure-east workspace:
- `data-science-on-databricks-workshop` (Classical ML / MLOps with churn prediction)
- `agents-workshop` (GenAI agent for customer service)

**Key modernizations**: Replace AutoML with Optuna + Genie Code alternative track, deploy agents to Databricks Apps instead of model serving, incorporate AI Functions, MCP Server, MLflow Tracing, and AI Gateway.

**Target workspace**: FEVM (`fevm` profile, `fevm-serverless-stable-goo4dg.cloud.databricks.com`)
**Deployment**: Asset Bundle for instructor setup + Git repo for participant access
**Catalog/Schema**: Configurable via config notebook (instructor sets before cloning for students)

---

## Project Structure

```
ml-ai-workshop/
├── CLAUDE.md
├── databricks.yml                        # Asset Bundle config
├── .gitignore
├── _resources/
│   ├── 00_config.py                      # Instructor-configurable: catalog, schema, endpoints
│   ├── 01_setup.py                       # Data generation, table creation, resource provisioning
│   └── data_generators.py                # Synthetic data generation functions (called by 01_setup)
├── module_1_classical_ml/
│   ├── 01_overview.py                    # Workshop intro, ML Runtime, data exploration
│   ├── 02_feature_engineering.py         # Feature Store with Unity Catalog
│   ├── 03_train_model.py                 # sklearn + LightGBM + Optuna tuning + MLflow tracking
│   ├── 03a_genie_code_alternative.py     # Markdown instructions: replicate with Genie Code. Include tips & tricks for prompting a coding agent
│   ├── 04_model_registry.py             # Register to UC, Champion/Challenger aliases, lineage
│   ├── 05_model_serving.py              # Online tables + Model Serving endpoint
│   ├── 06_batch_inference.py            # fe.score_batch() + ai_query() side-by-side
│   └── 07_monitoring.py                 # Lakehouse Monitoring for drift detection
├── module_2_genai/
│   ├── 01_ai_functions.py               # FMAPI, ai_query(), ai_extract() demos
│   ├── 02_create_tools.py               # UC SQL/Python functions as agent tools
│   ├── 03_build_agent.py                # LangGraph agent + UC tools + MCP Server
│   ├── 04_mlflow_tracing.py             # MLflow Tracing for agent observability
│   ├── 05_agent_eval.py                 # mlflow.genai.evaluate() with scorers
│   ├── 06_deploy_to_apps.py             # Deploy agent to Databricks Apps
│   └── 07_ai_gateway.py                 # AI Gateway: routing, guardrails, governance
└── agent_app/                            # Databricks App source for agent deployment
    ├── app.yaml
    ├── app.py                            # FastAPI server wrapping the agent
    ├── agent.py                          # Agent definition (extracted from notebook 03)
    └── requirements.txt
```

---

## Unified Data Model — Telecom Customer Retention

All data is synthetically generated to serve both modules from a single coherent domain: a telecom company's customer retention workflow. The ML model predicts churn, and the agent helps service reps act on those predictions.

### Tables

| Table | ~Rows | Description | Module 1 (ML) | Module 2 (GenAI) |
|-------|-------|-------------|----------------|-------------------|
| `customers` | 5,000 | Customer profiles: demographics, tenure, plan type, monthly/total charges, payment method, optional services (streaming, security, backup, etc.) | Primary feature source for churn model | Agent looks up customer profile |
| `service_tickets` | 15,000 | Support tickets: customer_id, timestamp, category (billing, technical, cancellation, upgrade), priority, status, resolution, free-text description | Feature: ticket count/frequency/recency as churn signals | Agent retrieves latest ticket, reviews history |
| `call_transcripts` | 3,000 | Call center transcripts: customer_id, timestamp, agent_name, transcript_text (~200-500 words), call_duration, disposition | Feature: sentiment score (derived via ai_extract in M2) | AI Functions demo: summarize, extract sentiment/entities, classify intent |
| `plans` | 15 | Telecom plan catalog: plan_name, monthly_cost, data_limit, features, contract_term | — | Agent looks up plan details for comparison/upsell |
| `policies` | 8 | Company policies: policy_name, policy_text, effective_date, category (cancellation, billing, upgrade, retention offers) | — | Agent retrieves relevant policy for retention decisions |
| `product_knowledge` | 50 | Knowledge base articles: title, content (~300-800 words), category, last_updated. Topics: troubleshooting, plan comparisons, feature guides, FAQ | — | Vector search RAG retrieval for agent |
| `churn_labels` | 5,000 | customer_id, churn (Yes/No), label_date. Separated from features to avoid leakage. | Target variable | Agent gets churn risk score from serving endpoint |

### Synthetic Data Generation Approach
<!-- COMMENTARY: -->

Generated via Python (pandas/faker) in `_resources/01_setup.py` — no external CSV dependencies. This ensures portability (no GitHub raw file dependencies) and lets us control correlations for a good ML signal.

**Key design constraints:**
- `customers` has realistic churn correlations: short tenure + high charges + month-to-month contract → higher churn
- `service_tickets` correlates with churn: churners have more tickets, especially "cancellation" and "billing" categories in the 90 days before churn
- `call_transcripts` contain realistic but synthetic dialogue. Churners' transcripts have more negative sentiment, mentions of competitors, complaints about pricing
- `product_knowledge` articles are detailed enough for meaningful RAG retrieval (not just one-liners)
- All tables share `customer_id` as the join key

**Generation uses:**
- `faker` for names, addresses, dates
- Controlled random distributions with seed for reproducibility
- Parameterized churn rate (~26%, matching IBM Telco)
- `call_transcripts` generated with templates + randomized fill (not LLM-generated, to avoid FMAPI dependency during setup)

### Cross-Module Data Flow

```
Module 1                                    Module 2
────────                                    ────────
customers ──┐
             ├──→ feature_table ──→ model ──→ serving_endpoint
service_tickets ┘       │                        │
                        │                        ▼
                  churn_labels            agent tool: get_churn_risk()
                                                 │
                                    ┌────────────┤
                                    │            │
                              call_transcripts   ├──→ retention_agent
                              (ai_functions)     │
                                    │      plans ┘
                                    │   policies ┘
                              ai_extract()  product_knowledge ──→ vector search
                              ai_query()
```

The churn model serving endpoint from Module 1 becomes a callable tool in the Module 2 agent. The agent workflow:
1. `get_latest_ticket()` → retrieve the most recent escalation
2. `get_customer_profile(customer_id)` → look up the customer
3. `get_churn_risk(customer_id)` → call the Module 1 serving endpoint
4. `get_ticket_history(customer_id)` → review support history
5. `search_knowledge_base(query)` → RAG for product/troubleshooting info
6. `get_retention_policy()` → check what offers/actions are allowed
7. Agent synthesizes a recommended retention action

---

## Initialization (`_resources/`)

### `00_config.py` - Instructor Configuration
<!-- COMMENTARY: -->

- Instructor sets `catalog`, `schema`, LLM endpoint, embedding endpoint before cloning
- Defaults: `catalog = "ml_ai_workshop"`, `schema = "workshop"`
- Endpoint defaults: `databricks-claude-sonnet-4-6`, `databricks-gte-large-en`
- All downstream notebooks `%run ../_resources/00_config`

### `01_setup.py` - Data & Resource Setup
<!-- COMMENTARY: -->

**Generates all data in-process** (no external CSV downloads):
- Generate `customers` table with controlled churn correlations (5,000 rows)
- Generate `service_tickets` correlated to churn behavior (15,000 rows)
- Generate `call_transcripts` with sentiment patterns (3,000 rows)
- Generate `plans`, `policies`, `product_knowledge` reference tables
- Extract `churn_labels` from `customers` into separate table with train/val/test split
- Create vector search endpoint + delta sync index on `product_knowledge`
- Enable CDC on `product_knowledge` table
- Grant permissions to `account users`

---

## Module 1: Classical ML (~55 min)

### `01_overview.py` (~5 min)
<!-- COMMENTARY: -->

**Reuse**: `00_mlops_end2end` structure and markdown (adapted for unified telecom domain)
- Workshop intro, architecture diagram of the 7-notebook flow + cross-module data flow
- ML Runtime exploration: `%pip list` to show pre-installed libs
- Load and explore `customers` and `service_tickets` tables
- Brief EDA: churn distribution, tenure vs. churn, ticket frequency vs. churn

### `02_feature_engineering.py` (~10 min)
<!-- COMMENTARY: -->

**Reuse**: `01_feature_engineering` patterns (adapted for new schema)
- Join `customers` with aggregated `service_tickets` features (ticket count, recency, category distribution)
- `compute_service_features()` PandasUDF for `num_optional_services`
- `clean_churn_features()` using Pandas on Spark API
- Create feature table via `FeatureEngineeringClient.create_table()` with primary keys + timeseries
- On-demand feature function `avg_price_increase` in SQL
- Write features with `fe.write_table()`
- Labels already in separate `churn_labels` table (created by setup)

### `03_train_model.py` (~15 min) - KEY CHANGE: Optuna replaces AutoML/hyperopt
<!-- COMMENTARY: -->

**Reuse**: Preprocessing pipeline from `02_automl_champion` (ColumnSelector, bool/numerical/categorical transformers, sklearn Pipeline), `fe.create_training_set()`, `fe.log_model()`, SHAP
**New**: Replace `hyperopt.fmin` with Optuna
- Load training set via `fe.create_training_set()` with FeatureLookup + FeatureFunction
- sklearn preprocessing pipeline (reuse ColumnSelector, transformers)
- **Optuna study** for LightGBM hyperparameter tuning:
  - `optuna.create_study(direction="maximize")` targeting F1 score
  - `study.optimize(objective, n_trials=20)`
  - MLflow callback: `mlflow.optuna.autolog()` or manual logging per trial
- Train final model with best params
- SHAP feature importance visualization
- Log model with `fe.log_model()` for feature lineage tracking
- Confusion matrix, ROC, Precision-Recall curves

### `03a_genie_code_alternative.py` - Alternative Track (markdown-only)
<!-- COMMENTARY: -->

**New**: Step-by-step markdown instructions with screenshots/descriptions:
1. Navigate to Genie Code in the Databricks UI
2. Point it at the `churn_labels` table joined with `churn_feature_table`
3. Describe how Genie Code generates a training notebook conversationally
4. Show how to export and customize the generated notebook
5. Compare output to what notebook `03` achieves manually

**Tips & tricks for prompting a coding agent** (new section):
- How to frame ML tasks as prompts (e.g., "Train a LightGBM classifier on this table, optimize F1, log to MLflow")
- Iterating on generated code: asking for SHAP, asking to tune hyperparameters, asking to change the metric
- Common pitfalls: under-specifying the target column, forgetting to split data, not setting a random seed
- When to trust vs. verify generated code (always check data leakage, always validate train/test split)

### `04_model_registry.py` (~8 min)
<!-- COMMENTARY: -->

**Reuse**: `03_from_notebook_to_models_in_uc` (heavy reuse)
- `mlflow.search_runs()` to find best run by F1 score
- Register model to UC: `mlflow.register_model()`
- Set description, version tags (F1 score)
- Set `Challenger` alias, promote to `Champion`
- Demonstrate UC model lineage (link to Catalog Explorer)

### `05_model_serving.py` (~8 min)
<!-- COMMENTARY: -->

**IMPORTANT**: Online Feature Store now uses Lakebase backend. Must use the current Lakebase-backed Online Tables API — NOT the legacy `w.online_tables` SDK. The archived `06_serve_features_and_model` notebook uses the old API and will need to be rewritten for the Lakebase backend.

**Adapted from**: `06_serve_features_and_model` (significant rewrite for Lakebase)
- Enable CDF on feature table
- Create Lakebase-backed online table via current Databricks SDK (`w.online_tables.create()` with Lakebase spec)
- Create Model Serving endpoint via SDK with `auto_capture_config` for inference logging
- Test endpoint with REST API call
- Show real-time prediction with feature lookup

### `06_batch_inference.py` (~5 min)
<!-- COMMENTARY: -->

**Reuse**: `05_batch_inference` + `07_batch_inference_pipelines`
- **Python approach**: `fe.score_batch()` with Champion model
- **SQL approach**: `ai_query()` calling the serving endpoint (bridges to Module 2)
- Save predictions to inference table for monitoring

### `07_monitoring.py` (~5 min)
<!-- COMMENTARY: -->

**Reuse**: `extra_08_model_monitoring` (streamlined)
- Create baseline table from offline inference
- Create Lakehouse Monitor with `w.quality_monitors.create()`
- Review `profile_metrics` and `drift_metrics` tables
- Brief note on alerting setup

---

## Module 2: GenAI Development & Deployment (~55 min)

### `01_ai_functions.py` (~8 min) - NEW
<!-- COMMENTARY: -->

- Intro to Foundation Model APIs (FMAPI) on Databricks
- **`ai_sentiment()` demo**:
  - Perform sentiment analysis on `call_transcripts` — returns sentiment label + score
  - Show as a simple, no-config AI function that works out of the box
- **`ai_query()` demos** (all on the telecom data from Module 1):
  - Classify `call_transcripts` by intent (cancellation, billing dispute, technical issue, upgrade inquiry)
  - Summarize call transcripts into one-line disposition notes
  - Explain churn predictions in natural language: "Why is this customer at risk?"
- **`ai_extract()` demos**:
  - Extract structured fields from `call_transcripts`: competitor_mentioned, key_complaint
  - Extract entities from `service_tickets`: product_referenced, resolution_type
- Show these in pure SQL — usable in dashboards, pipelines, and as enrichment for the ML model

### `02_create_tools.py` (~10 min)
<!-- COMMENTARY: -->

**Vector search**: Yes — the endpoint + index on `product_knowledge` is created during `_resources/01_setup.py`. This notebook just creates the `VectorSearchRetrieverTool` that wraps the existing index. No new infrastructure deployed here.

**Reuse**: `agents-workshop/01_create_tools` patterns (adapted for telecom domain)
- Create UC SQL functions for the retention agent workflow:
  - `get_latest_ticket()` — retrieve most recent escalated service ticket
  - `get_customer_profile(customer_id)` — look up customer demographics, plan, tenure
  - `get_ticket_history(customer_id)` — return ticket count by category + recent tickets
  - `get_retention_policy()` — retrieve cancellation/retention offer policies
  - `get_churn_risk(customer_id)` — call the Module 1 serving endpoint via ai_query()
- Test each function with SQL queries
- Register a Python function to UC via `DatabricksFunctionClient`
- Create `VectorSearchRetrieverTool` wrapping the `product_knowledge` index (endpoint/index already provisioned by setup)
- Link to AI Playground for interactive testing with tools

### `03_build_agent.py` (~10 min)
<!-- COMMENTARY: -->

**Reuse**: Agent pattern from agents-workshop `agent.py` (LangGraph + VectorSearchRetrieverTool + UnityCatalogTool)
**New**: MCP Server integration
- Define LangGraph **retention agent** with system prompt: "You are a telecom retention specialist. When a customer calls to cancel, look up their profile, assess churn risk, review their history, search for relevant solutions, and recommend a retention action."
- Bind all UC functions from notebook 02 + vector search as tools
- **MCP Server integration**:
  - Show agent consuming tools exposed via MCP protocol
  - `databricks-langchain` MCP tool integration
  - Example: UC functions exposed as MCP tools
  - Include fallback code path if MCP not yet GA
- Test the agent with realistic scenarios:
  - "Customer John Smith just called wanting to cancel. Help me retain them."
  - "What retention offers can I make to a customer who's been with us 2 years?"
- Show tool call traces inline — demonstrate the agent calling the churn model endpoint

### `04_mlflow_tracing.py` (~7 min) - NEW
<!-- COMMENTARY: -->

- Enable MLflow Tracing: `mlflow.langchain.autolog()`
- Run the agent with tracing enabled
- Explore trace UI: spans, parent/child relationships, tool calls, LLM calls
- Query traces programmatically: `mlflow.search_traces()`
- Show how traces feed into evaluation and debugging

### `05_agent_eval.py` (~8 min)
<!-- COMMENTARY: -->

**Reuse**: `agents-workshop/02_agent_eval/driver` (adapted)
- Log agent as MLflow model with `mlflow.pyfunc.log_model()`
- Create evaluation dataset (5+ examples)
- Run `mlflow.genai.evaluate()` with scorers:
  - `RelevanceToQuery`, `RetrievalGroundedness`, `Safety`, custom `Guidelines`
- Iterate: change system prompt, re-run eval, compare in MLflow UI
- Register improved model to UC

### `06_deploy_to_apps.py` (~8 min) - NEW (replaces agents.deploy())
<!-- COMMENTARY: -->

- Walk through `agent_app/` directory structure:
  - `app.yaml`: command config
  - `app.py`: FastAPI server with `/chat` endpoint
  - `agent.py`: Agent definition (same as notebook 03, extracted as module)
  - `requirements.txt`
- Deploy via CLI: `databricks apps create` + `databricks apps deploy`
- Alternative: deploy via SDK (`w.apps.create()`, `w.apps.deploy()`)
- Test the deployed app endpoint
- Show the app URL accessible to users

### `07_ai_gateway.py` (~5 min) - NEW
<!-- COMMENTARY: -->

- Introduction to AI Gateway (Mosaic AI Gateway)
- Create/configure AI Gateway route for FMAPI endpoint
- Configure: rate limiting, usage tracking, guardrails (content filtering, PII)
- Route agent LLM calls through the gateway
- Demonstrate guardrail triggers
- Review gateway logs and metrics

---

## Agent App (`agent_app/`)

### `app.yaml`
<!-- COMMENTARY: -->

```yaml
command: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `app.py` - FastAPI server
<!-- COMMENTARY: -->

- `/chat` POST endpoint: `{"messages": [{"role": "user", "content": "..."}]}`
- Loads LangGraph agent from `agent.py`
- `/health` GET for health checks
- Optional: simple HTML chat interface

### `agent.py` - Retention agent definition
<!-- COMMENTARY: -->

- Same LangGraph retention agent as `module_2_genai/03_build_agent.py`, extracted as importable module
- Tools: get_customer_profile, get_churn_risk (calls Module 1 endpoint), get_ticket_history, get_retention_policy, search_knowledge_base (vector search)
- Configured LLM endpoint from environment/config

### `requirements.txt`
<!-- COMMENTARY: -->

```
fastapi
uvicorn
databricks-sdk
databricks-langchain
langgraph>=0.3.4
mlflow-skinny[databricks]
```

---

## Deployment

### `databricks.yml` (Asset Bundle)
<!-- COMMENTARY: -->

```yaml
bundle:
  name: ml-ai-workshop

workspace:
  root_path: /Workspace/Users/${workspace.current_user.userName}/ml-ai-workshop

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: https://fevm-serverless-stable-goo4dg.cloud.databricks.com/
```

**Instructor workflow**:
1. `databricks bundle deploy --profile fevm` to push all notebooks
2. Run `_resources/01_setup.py` to provision data and resources
3. Clone the deployed folder for each participant (or use Git folder sync)

**Participant workflow** (Git):
- Clone the Git repo as a Databricks Git folder
- Run `_resources/00_config` (already configured by instructor)
- Work through modules sequentially

---

## Timing Summary

| Module 1 | Time | Module 2 | Time |
|----------|------|----------|------|
| 01 Overview + ML Runtime | 5 min | 01 AI Functions | 8 min |
| 02 Feature Engineering | 10 min | 02 Create Tools | 10 min |
| 03 Train Model (Optuna) | 15 min | 03 Build Agent + MCP | 10 min |
| 03a Genie Code Alt | (self-paced) | 04 MLflow Tracing | 7 min |
| 04 Model Registry | 8 min | 05 Agent Eval | 8 min |
| 05 Model Serving | 8 min | 06 Deploy to Apps | 8 min |
| 06 Batch Inference | 5 min | 07 AI Gateway | 5 min |
| 07 Monitoring | 5 min | | |
| **Total** | **~56 min** | **Total** | **~56 min** |

---

## Verification

1. **Deploy**: `databricks bundle deploy --profile fevm` succeeds
2. **Setup**: Run `01_setup.py` - all tables created, vector search index online
3. **Module 1**: Run notebooks 01-07 sequentially on ML Runtime cluster - all cells pass
4. **Module 2**: Run notebooks 01-07 sequentially - agent builds, evals pass
5. **App deployment**: `databricks apps deploy` succeeds, `/chat` endpoint returns responses
6. **Portability**: Change config notebook catalog/schema, redeploy to different workspace
