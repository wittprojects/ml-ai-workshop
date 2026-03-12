# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 04 — MLflow Tracing
# MAGIC
# MAGIC **Time**: ~7 min
# MAGIC
# MAGIC **MLflow Tracing** provides observability for LLM applications — see every LLM call, tool invocation, and retrieval step.
# MAGIC
# MAGIC Key concepts:
# MAGIC - Auto-tracing with `mlflow.langchain.autolog()`
# MAGIC - Trace UI: spans, parent/child relationships
# MAGIC - Programmatic trace queries

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %pip install databricks-langchain langgraph mlflow databricks-sdk==0.50.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

import mlflow

# Enable MLflow Tracing for LangChain/LangGraph
mlflow.langchain.autolog()

# Set experiment for traces
experiment_path = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/ml-ai-workshop-genai"
mlflow.set_experiment(experiment_path)
print(f"MLflow experiment: {experiment_path}")
print("✓ LangChain autologging enabled — all agent calls will be traced")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rebuild the Agent (with tracing enabled)

# COMMAND ----------

from databricks_langchain import ChatDatabricks, UCFunctionToolkit, VectorSearchRetrieverTool
from langgraph.prebuilt import create_react_agent

llm = ChatDatabricks(endpoint=llm_endpoint)

uc_toolkit = UCFunctionToolkit(
    function_names=[
        f"{catalog}.{schema}.get_latest_ticket",
        f"{catalog}.{schema}.get_customer_profile",
        f"{catalog}.{schema}.get_ticket_history",
        f"{catalog}.{schema}.get_retention_policy",
        f"{catalog}.{schema}.get_churn_risk",
    ]
)

vs_tool = VectorSearchRetrieverTool(
    index_name=product_knowledge_index,
    tool_name="search_knowledge_base",
    tool_description="Search the product knowledge base for troubleshooting guides, plan comparisons, feature info, and FAQs.",
    columns=["article_id", "title", "content", "category"],
    num_results=3,
)

SYSTEM_PROMPT = """You are a telecom customer retention specialist assistant. Your job is to help service representatives retain customers who are at risk of churning.

When a customer calls in, follow this workflow:
1. Look up the latest escalated ticket to identify the customer and issue
2. Get the customer's profile to understand their account
3. Check their churn risk score from the ML model
4. Review their ticket history for patterns
5. Search the knowledge base for relevant solutions
6. Check retention policies to know what offers you can make
7. Synthesize all information into a recommended retention action

Always be empathetic and solution-oriented. Provide specific, actionable recommendations."""

agent = create_react_agent(
    model=llm,
    tools=uc_toolkit.tools + [vs_tool],
    prompt=SYSTEM_PROMPT,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Agent with Tracing

# COMMAND ----------

# This call is fully traced — every LLM call, tool invocation, and response is captured
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Help me retain customer CUST-00100. They called about cancellation."}]}
)

print(response["messages"][-1].content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore Traces in the UI
# MAGIC
# MAGIC Navigate to the **MLflow Experiment** page:
# MAGIC 1. Click the experiment link above or go to **Experiments** in the sidebar
# MAGIC 2. Click on the latest run
# MAGIC 3. Open the **Traces** tab
# MAGIC
# MAGIC You'll see:
# MAGIC - **Span hierarchy**: parent → child relationships showing the full execution flow
# MAGIC - **LLM calls**: model, prompt, response, tokens used, latency
# MAGIC - **Tool calls**: function name, input arguments, output
# MAGIC - **Retrieval**: vector search queries and retrieved documents

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query Traces Programmatically

# COMMAND ----------

# Search for recent traces
traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(experiment_path).experiment_id],
    max_results=5,
)

display(traces)

# COMMAND ----------

# Look at the latest trace details
if len(traces) > 0:
    latest_trace = traces.iloc[0]
    print(f"Trace ID: {latest_trace['request_id']}")
    print(f"Status: {latest_trace['status']}")
    print(f"Execution time: {latest_trace.get('execution_time_ms', 'N/A')}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Another Scenario for Comparison

# COMMAND ----------

# Different scenario to compare traces
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "A long-time customer (CUST-00500) is complaining about slow internet speeds. What should I do?"}]}
)

print(response2["messages"][-1].content)

# COMMAND ----------

# Compare the two traces
traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(experiment_path).experiment_id],
    max_results=5,
)

print("Recent traces:")
for _, trace in traces.iterrows():
    print(f"  {trace['request_id'][:12]}... | Status: {trace['status']} | Time: {trace.get('execution_time_ms', 'N/A')}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC MLflow Tracing gives you full observability into agent behavior:
# MAGIC
# MAGIC | Feature | What You See |
# MAGIC |---------|-------------|
# MAGIC | Span hierarchy | Full execution flow with parent/child |
# MAGIC | LLM calls | Model, prompt, response, tokens, latency |
# MAGIC | Tool calls | Function name, inputs, outputs |
# MAGIC | Retrieval | Vector search queries and results |
# MAGIC | Programmatic access | `mlflow.search_traces()` for analysis |
# MAGIC
# MAGIC Traces feed directly into **agent evaluation** (next notebook) for systematic quality assessment.
# MAGIC
# MAGIC **Next**: [05 Agent Eval →](./05_agent_eval)
