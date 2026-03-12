# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 06 — Deploy to Databricks Apps
# MAGIC
# MAGIC **Time**: ~8 min
# MAGIC
# MAGIC Deploy the retention agent as a **Databricks App** — a managed web application with built-in auth.
# MAGIC
# MAGIC Key concepts:
# MAGIC - Databricks Apps architecture
# MAGIC - FastAPI server wrapping the agent
# MAGIC - CLI and SDK deployment
# MAGIC - Testing the deployed endpoint

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.50.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## The `agent_app/` Directory
# MAGIC
# MAGIC Our app consists of 4 files:
# MAGIC
# MAGIC ```
# MAGIC agent_app/
# MAGIC ├── app.yaml           # App configuration
# MAGIC ├── app.py             # FastAPI server with /chat endpoint
# MAGIC ├── agent.py           # LangGraph agent (extracted from notebook 03)
# MAGIC └── requirements.txt   # Python dependencies
# MAGIC ```
# MAGIC
# MAGIC Let's walk through each file.

# COMMAND ----------

# MAGIC %md
# MAGIC ### `app.yaml`
# MAGIC ```yaml
# MAGIC command: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# MAGIC ```
# MAGIC
# MAGIC ### `app.py` — FastAPI Server
# MAGIC The server exposes:
# MAGIC - `POST /chat` — send messages to the agent
# MAGIC - `GET /health` — health check endpoint
# MAGIC
# MAGIC ### `agent.py` — Agent Definition
# MAGIC The same LangGraph agent from notebook 03, extracted as an importable module.
# MAGIC
# MAGIC ### `requirements.txt`
# MAGIC ```
# MAGIC fastapi
# MAGIC uvicorn
# MAGIC databricks-sdk
# MAGIC databricks-langchain
# MAGIC langgraph>=0.3.4
# MAGIC mlflow-skinny[databricks]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy via Databricks SDK

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import time

w = WorkspaceClient()
app_name = "workshop-agent-app"

# COMMAND ----------

# Create the app (if it doesn't exist)
try:
    app = w.apps.create_and_wait(
        name=app_name,
        description="Telecom customer retention agent — ML AI Workshop",
    )
    print(f"✓ App '{app_name}' created")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ App '{app_name}' already exists")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the App
# MAGIC
# MAGIC The deploy step uploads the `agent_app/` source code and starts the application.
# MAGIC
# MAGIC **Via CLI** (run in terminal):
# MAGIC ```bash
# MAGIC databricks apps deploy workshop-agent-app --source-code-path agent_app/ --profile fevm
# MAGIC ```
# MAGIC
# MAGIC **Via SDK** (below):

# COMMAND ----------

# Deploy using SDK
import os

# Get the workspace path for the source code
current_user = spark.sql("SELECT current_user()").first()[0]
source_path = f"/Workspace/Users/{current_user}/ml-ai-workshop/agent_app"

try:
    deployment = w.apps.deploy_and_wait(
        app_name=app_name,
        source_code_path=source_path,
    )
    print(f"✓ App deployed successfully")
    print(f"  Status: {deployment.status.state}")
except Exception as e:
    print(f"Deployment note: {e}")
    print("\nIf SDK deployment doesn't work, use the CLI:")
    print(f"  databricks apps deploy {app_name} --source-code-path agent_app/ --profile fevm")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get App URL

# COMMAND ----------

try:
    app_info = w.apps.get(name=app_name)
    app_url = app_info.url
    print(f"App URL: {app_url}")
    print(f"Status: {app_info.status.state if app_info.status else 'unknown'}")
except Exception as e:
    print(f"Note: {e}")
    print("The app URL will be available in the Databricks Apps UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Deployed App

# COMMAND ----------

import requests
import json

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Test health endpoint
try:
    health_response = requests.get(
        f"{app_url}/health",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    print(f"Health check: {health_response.status_code} — {health_response.json()}")
except Exception as e:
    print(f"Health check: {e}")

# COMMAND ----------

# Test chat endpoint
try:
    chat_response = requests.post(
        f"{app_url}/chat",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"messages": [{"role": "user", "content": "Help me retain customer CUST-00050 who wants to cancel."}]},
        timeout=60,
    )
    print(f"Status: {chat_response.status_code}")
    result = chat_response.json()
    print(f"\nAgent response:\n{result.get('response', result)}")
except Exception as e:
    print(f"Chat test: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We deployed the retention agent as a Databricks App:
# MAGIC
# MAGIC | Aspect | Detail |
# MAGIC |--------|--------|
# MAGIC | Framework | FastAPI + Uvicorn |
# MAGIC | Endpoints | `/chat` (POST), `/health` (GET) |
# MAGIC | Auth | Built-in Databricks authentication |
# MAGIC | Scaling | Managed by Databricks |
# MAGIC | Monitoring | MLflow tracing still active |
# MAGIC
# MAGIC The app is accessible to anyone with workspace access at the app URL.
# MAGIC
# MAGIC **Next**: [07 AI Gateway →](./07_ai_gateway)
