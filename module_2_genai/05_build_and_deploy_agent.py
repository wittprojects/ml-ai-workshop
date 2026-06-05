# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 05 — Build & Deploy an Agent
# MAGIC
# MAGIC **Time**: ~12 min
# MAGIC
# MAGIC In this notebook we go from **prototype to production** using two Databricks capabilities:
# MAGIC 1. **AI Playground** — visually prototype an agent with tools (no code)
# MAGIC 2. **Databricks Apps** — deploy the agent as a production FastAPI service
# MAGIC
# MAGIC | Notebook | Topic |
# MAGIC |----------|-------|
# MAGIC | 01 AI Functions | FMAPI, ai_query(), ai_extract(), ai_parse_document() |
# MAGIC | 02 Create Tools | UC functions as agent tools |
# MAGIC | 03 Agent Eval | mlflow.genai.evaluate() |
# MAGIC | 04 Metric Views & Genie Room | Governed metrics + NL analytics |
# MAGIC | **05 Build & Deploy Agent** | AI Playground + Databricks Apps |

# COMMAND ----------

# MAGIC %pip install databricks-sdk>=0.50.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: AI Playground
# MAGIC
# MAGIC The **AI Playground** is Databricks' built-in prototyping environment for LLM-powered agents.
# MAGIC It lets you:
# MAGIC - Select any Foundation Model API endpoint
# MAGIC - Attach **Unity Catalog functions** as tools
# MAGIC - Set a system prompt
# MAGIC - Test multi-turn conversations — with tool calls visible in real time
# MAGIC
# MAGIC No code required. When you're happy with the behavior, export the agent as Python code.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Open the Playground
# MAGIC
# MAGIC 1. In the left sidebar, click **Playground**
# MAGIC 2. Select a model — we recommend **databricks-claude-sonnet-4-6** (or any chat model on your FMAPI)
# MAGIC 3. Toggle **"Tools"** on in the top-right

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Set the System Prompt
# MAGIC
# MAGIC Paste the following system prompt into the **System Prompt** field:
# MAGIC
# MAGIC ```
# MAGIC You are a telecom customer retention specialist assistant. Your job is to help
# MAGIC service representatives retain customers who are at risk of churning.
# MAGIC
# MAGIC IMPORTANT RULES:
# MAGIC - Always look up the customer's profile and churn risk before making recommendations
# MAGIC - Always check the retention policy to ensure your offers are within guidelines
# MAGIC - For high-risk customers (churn score > 0.7), prioritize aggressive retention offers
# MAGIC - For billing complaints, always offer to review and adjust the plan
# MAGIC - For technical issues, search the knowledge base first, then offer service credits
# MAGIC - Always address the customer by name
# MAGIC - Structure your response as: 1) Situation Summary, 2) Key Findings, 3) Recommended Action
# MAGIC
# MAGIC When a customer calls in, follow this workflow:
# MAGIC 1. Look up the latest escalated ticket to identify the customer and issue
# MAGIC 2. Get the customer's profile to understand their account
# MAGIC 3. Check their churn risk score from the ML model
# MAGIC 4. Review their ticket history for patterns
# MAGIC 5. Search the knowledge base for relevant solutions
# MAGIC 6. Check retention policies to know what offers you can make
# MAGIC 7. Synthesize all information into a recommended retention action
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Add UC Function Tools
# MAGIC
# MAGIC Click **+ Add Tool** and add these five Unity Catalog functions:

# COMMAND ----------

# Print the fully-qualified function names for easy copy-paste into the Playground
functions = [
    "get_latest_ticket",
    "get_customer_profile",
    "get_ticket_history",
    "get_retention_policy",
    "get_churn_risk",
]
print("Add these UC functions as tools in the Playground:\n")
for fn in functions:
    print(f"  {catalog}.{schema}.{fn}")

print(f"\nNote: Vector Search (search_knowledge_base) cannot be added via the Playground UI.")
print(f"It is included in the deployed app code as a VectorSearchRetrieverTool.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Test the Agent
# MAGIC
# MAGIC Try these prompts in the Playground chat:
# MAGIC
# MAGIC 1. **"A customer just called in about an escalated ticket. Help me retain them."**
# MAGIC    - Watch the agent call `get_latest_ticket` → `get_customer_profile` → `get_churn_risk` → `get_retention_policy`
# MAGIC
# MAGIC 2. **"What's the churn risk for customer C-1042?"**
# MAGIC    - Single tool call to `get_churn_risk`
# MAGIC
# MAGIC 3. **"Show me the ticket history for the customer and suggest a retention offer."**
# MAGIC    - Multi-step: `get_ticket_history` → `get_retention_policy` → synthesized recommendation
# MAGIC
# MAGIC **What to observe**:
# MAGIC - The tool calls appear inline — you can see exactly what the agent decided to call and what it got back
# MAGIC - The system prompt guides the agent's workflow and response structure
# MAGIC - The agent chains multiple tools together autonomously

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Iterate
# MAGIC
# MAGIC The Playground is great for rapid iteration:
# MAGIC - **Tweak the system prompt** — make it more/less structured, add constraints
# MAGIC - **Try a different model** — compare Claude vs. Meta Llama vs. DBRX responses
# MAGIC - **Add/remove tools** — see how the agent adapts
# MAGIC
# MAGIC This is the fastest way to prototype agent behavior before writing any code.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Export
# MAGIC
# MAGIC When satisfied, click **Export** → **Export to notebook** in the Playground.
# MAGIC
# MAGIC The exported code uses `databricks-langchain` and `langgraph` — the same libraries a deployed app uses.
# MAGIC A deployed `agent_app/agent.py` (see the optional deploy section below) follows the same pattern:
# MAGIC - `ChatDatabricks` for the LLM
# MAGIC - `UCFunctionToolkit` for UC function tools
# MAGIC - `create_react_agent` from LangGraph

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Deploy as a Databricks App
# MAGIC
# MAGIC **Databricks Apps** let you deploy any Python web application (FastAPI, Streamlit, Dash, Gradio)
# MAGIC directly on Databricks with:
# MAGIC - **Built-in authentication** — inherits workspace identity
# MAGIC - **Unity Catalog access** — the app can call UC functions, query tables, and use model serving
# MAGIC - **Managed infrastructure** — no Docker, no Kubernetes, just push code
# MAGIC
# MAGIC > **Note**: This section walks through deploying the agent as a Databricks App. You'll review the app structure below and can optionally deploy it yourself.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code Walkthrough
# MAGIC
# MAGIC A Databricks App for this agent lives in an `agent_app/` directory with four files.
# MAGIC These are a template you create — they aren't shipped in this repo:
# MAGIC
# MAGIC ### `agent.py` — Agent logic
# MAGIC The same agent you prototyped in the Playground, as importable Python:
# MAGIC - System prompt defining the retention specialist workflow
# MAGIC - `UCFunctionToolkit` with our 5 UC functions
# MAGIC - `VectorSearchRetrieverTool` for knowledge base RAG (not available in Playground)
# MAGIC - `create_react_agent()` from LangGraph to wire it all together
# MAGIC
# MAGIC ### `app.py` — FastAPI server
# MAGIC ```python
# MAGIC @app.post("/chat")
# MAGIC def chat(request: ChatRequest):
# MAGIC     result = agent.invoke({"messages": request.messages})
# MAGIC     return ChatResponse(response=result["messages"][-1].content)
# MAGIC ```
# MAGIC A single `/chat` endpoint that accepts messages and returns the agent's response.
# MAGIC
# MAGIC ### `app.yaml` — App configuration
# MAGIC ```yaml
# MAGIC command: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# MAGIC ```
# MAGIC Tells Databricks how to start the app.
# MAGIC
# MAGIC ### `requirements.txt` — Dependencies
# MAGIC ```
# MAGIC fastapi, uvicorn, databricks-sdk, databricks-langchain, langgraph, mlflow-skinny
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the App (Optional)
# MAGIC
# MAGIC The cells below show how to deploy the app. This step is optional and requires
# MAGIC creating the `agent_app/` files above first (not included in this repo) and a
# MAGIC configured Databricks CLI.

# COMMAND ----------

# --- Optional: deploy the app ---
# Uncomment and run to deploy (requires the Databricks CLI configured and an
# agent_app/ directory containing the files described above).
#
# import subprocess
# result = subprocess.run(
#     ["databricks", "apps", "deploy", app_name,
#      "--source-code-path", "agent_app/"],
#     capture_output=True, text=True
# )
# print(result.stdout)
# if result.returncode != 0:
#     print(f"Error: {result.stderr}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Deployed App

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# import requests, json

# w = WorkspaceClient()

# # Get the app URL
# app_info = w.apps.get(app_name)
# app_url = f"https://{app_info.url}"
# print(f"App URL: {app_url}")

# # Get a token for authentication
# token = w.tokens.create(comment="workshop-test", lifetime_seconds=600)

# # Test the /chat endpoint
# response = requests.post(
#     f"{app_url}/chat",
#     headers={
#         "Authorization": f"Bearer {token.token_value}",
#         "Content-Type": "application/json"
#     },
#     json={
#         "messages": [
#             {"role": "user", "content": "A customer just called in about an escalated ticket. Help me retain them."}
#         ]
#     }
# )

# print(f"\nStatus: {response.status_code}")
# print(f"\nAgent Response:\n{response.json()['response']}")

# # Clean up the temporary token
# w.tokens.delete(token.token_info.token_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We followed the **Prototype → Export → Deploy → Evaluate** workflow:
# MAGIC
# MAGIC | Step | Tool | What You Did |
# MAGIC |------|------|-------------|
# MAGIC | Prototype | AI Playground | Tested agent with tools, iterated on system prompt |
# MAGIC | Export | Playground Export | Generated LangGraph code from the prototype |
# MAGIC | Deploy | Databricks Apps | Shipped agent as a FastAPI service with UC access |
# MAGIC | Evaluate | *(Next notebook)* | Systematically evaluate agent quality |
# MAGIC
# MAGIC **Next**: [03 Agent Eval →](./03_agent_eval)
