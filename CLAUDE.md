# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

A portable, two-module Databricks workshop (~55 min each) covering Machine Learning and GenAI/Agents. Notebooks are Databricks-format Python files (header `# Databricks notebook source`, cells separated by `# COMMAND ----------`). The workshop must run on any Databricks workspace — all catalog/schema/endpoint references come from `_resources/00_config.py`.

## Architecture

- **`_resources/00_config.py`** — Single source of truth for catalog, schema, endpoint names. Every notebook starts with `%run ../_resources/00_config`. Instructor configures this before cloning for students.
- **`_resources/01_setup.py`** — Run once by instructor. Provisions data (IBM Telco Churn for Module 1, customer service CSVs for Module 2), creates tables, vector search endpoint/index, and grants permissions.
- **Module 1 (`module_1_machine_learning/`)** — Sequential notebooks 01-06: data exploration → feature engineering (Feature Store + UC) → model training (LightGBM + Optuna) → model serving → batch inference → Lakehouse Monitoring. Notebook `03a` is a markdown-only alternative track for Genie Code.
- **Module 2 (`module_2_genai/`)** — Sequential notebooks 01-03, 05: AI Functions (ai_query/ai_extract) → UC tool creation → AI Playground + Databricks Apps deployment → agent evaluation.
- **`agent_app/`** — Standalone FastAPI application deployed as a Databricks App. `agent.py` is the same agent from Module 2 notebook 03, extracted as an importable module.

## Databricks CLI

Use the full path to avoid the legacy CLI (v0.18.0) in conda:

```bash
/opt/homebrew/bin/databricks --profile <profile> <command>
```

**Primary workspace**: `fevm` profile (`fevm-serverless-stable-goo4dg.cloud.databricks.com`)

**Reference workshops** (azure-east workspace, read-only source material):
- `/Users/alex.witt@databricks.com/archive/data-science-on-databricks-workshop/`
- `/Users/alex.witt@databricks.com/archive/agents-workshop/`

## Deployment

### Asset Bundle (instructor setup)
```bash
/opt/homebrew/bin/databricks bundle deploy --profile fevm
```

### Git Folder (participant access)
Participants clone the Git repo as a Databricks Git folder and work through modules sequentially.

### Databricks App (Module 2)
```bash
/opt/homebrew/bin/databricks apps create --name workshop-agent-app --profile fevm
/opt/homebrew/bin/databricks apps deploy workshop-agent-app --source-code-path agent_app/ --profile fevm
```

## Key Conventions

- Never hardcode catalog, schema, or endpoint names — always reference variables from `00_config.py`
- Notebook format: `# Databricks notebook source` header, `# COMMAND ----------` cell separators, `# MAGIC %md` for markdown, `# MAGIC %sql` for SQL cells
- Use `IDENTIFIER(:catalog_name || '.' || :schema_name || '.' || 'function_name')` pattern for parameterized SQL function creation
- Default FMAPI models: `databricks-claude-sonnet-4-6` (chat), `databricks-gte-large-en` (embeddings)
- Training uses Optuna (not hyperopt or AutoML) for hyperparameter tuning
- Agent deployment targets Databricks Apps (not `agents.deploy()` to model serving)
