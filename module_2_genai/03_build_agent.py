# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 03 — Build LangGraph Retention Agent
# MAGIC
# MAGIC **Time**: ~10 min
# MAGIC
# MAGIC We'll build a **LangGraph agent** that uses the UC tools from Notebook 02 to help telecom service reps retain at-risk customers.
# MAGIC
# MAGIC Key concepts:
# MAGIC - LangGraph agent with tool binding
# MAGIC - UC functions as tools via `UCFunctionToolkit`
# MAGIC - VectorSearchRetrieverTool for RAG
# MAGIC - MCP Server integration

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %pip install databricks-langchain langgraph mlflow databricks-sdk==0.50.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Agent

# COMMAND ----------

from databricks_langchain import ChatDatabricks, UCFunctionToolkit, VectorSearchRetrieverTool
from langgraph.prebuilt import create_react_agent

# COMMAND ----------

# LLM
llm = ChatDatabricks(endpoint=llm_endpoint)

# UC Function tools — load available tools, skip any that don't exist yet
uc_function_names = [
    f"{catalog}.{schema}.get_latest_ticket",
    f"{catalog}.{schema}.get_customer_profile",
    f"{catalog}.{schema}.get_ticket_history",
    f"{catalog}.{schema}.get_retention_policy",
    f"{catalog}.{schema}.get_churn_risk",
]

loaded_names = []
for fn in uc_function_names:
    try:
        UCFunctionToolkit(function_names=[fn])
        loaded_names.append(fn)
    except Exception as e:
        print(f"  Skipping {fn.split('.')[-1]}: {e}")

uc_toolkit = UCFunctionToolkit(function_names=loaded_names) if loaded_names else None
uc_tools = uc_toolkit.tools if uc_toolkit else []

# Vector search tool
vs_tool = VectorSearchRetrieverTool(
    index_name=product_knowledge_index,
    tool_name="search_knowledge_base",
    tool_description="Search the product knowledge base for troubleshooting guides, plan comparisons, feature info, and FAQs.",
    columns=["article_id", "title", "content", "category"],
    num_results=3,
)

all_tools = uc_tools + [vs_tool]
print(f"Agent tools ({len(all_tools)}):")
for t in all_tools:
    print(f"  - {t.name}")

# COMMAND ----------

# System prompt
SYSTEM_PROMPT = """You are a telecom customer retention specialist assistant. Your job is to help service representatives retain customers who are at risk of churning.

When a customer calls in (especially about cancellation or billing issues), follow this workflow:
1. Look up the latest escalated ticket to identify the customer and issue
2. Get the customer's profile to understand their account
3. Check their churn risk score from the ML model
4. Review their ticket history for patterns
5. Search the knowledge base for relevant solutions or product information
6. Check retention policies to know what offers you can make
7. Synthesize all information into a recommended retention action

Always be empathetic and solution-oriented. Provide specific, actionable recommendations with concrete offers based on the retention policy guidelines. Include the customer's name when making recommendations."""

# COMMAND ----------

# Create the agent
agent = create_react_agent(
    model=llm,
    tools=all_tools,
    prompt=SYSTEM_PROMPT,
)

print("✓ Retention agent created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Agent

# COMMAND ----------

# Scenario 1: Handle the latest escalated ticket
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Check the latest escalated ticket and help me retain this customer."}]}
)

print(response["messages"][-1].content)

# COMMAND ----------

# Scenario 2: Specific customer lookup
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Customer CUST-00050 just called wanting to cancel. They say the price is too high. What retention strategy should I use?"}]}
)

print(response["messages"][-1].content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Tool Call Traces
# MAGIC
# MAGIC Let's examine which tools the agent called and in what order.

# COMMAND ----------

# Show the full message chain from the last response
for msg in response["messages"]:
    role = msg.type if hasattr(msg, 'type') else 'unknown'
    if role == "ai" and hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"🔧 Tool call: {tc['name']}({tc['args']})")
    elif role == "tool":
        print(f"  ↳ Result: {str(msg.content)[:200]}...")
    elif role == "ai":
        print(f"🤖 Agent: {msg.content[:300]}...")
    elif role == "human":
        print(f"👤 User: {msg.content}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MCP Server Integration
# MAGIC
# MAGIC The Model Context Protocol (MCP) allows agents to discover and use tools via a standardized protocol.
# MAGIC UC functions can be exposed as MCP tools.

# COMMAND ----------

# MCP integration — expose UC functions as MCP-compatible tools
# This demonstrates the MCP protocol for tool discovery and invocation

from databricks_langchain import UCFunctionToolkit

# The UCFunctionToolkit already wraps UC functions in a way that's compatible
# with the MCP tool interface. When deployed as an MCP server, these tools
# become discoverable by any MCP-compatible client.

# To serve these tools via MCP, you would deploy an MCP server:
# from databricks_langchain.mcp import DatabricksMCPServer
# mcp_server = DatabricksMCPServer(tools=all_tools)

# For now, let's verify our tools follow the MCP tool schema
for tool in all_tools:
    print(f"Tool: {tool.name}")
    print(f"  Description: {tool.description[:100]}...")
    if hasattr(tool, 'args_schema'):
        print(f"  Schema: {tool.args_schema.schema() if tool.args_schema else 'None'}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We built a LangGraph retention agent that:
# MAGIC - Uses **6 tools** (5 UC functions + 1 vector search) to gather information
# MAGIC - Follows a structured workflow to assess churn risk and recommend actions
# MAGIC - Combines **ML predictions** (churn model) with **LLM reasoning** (retention strategy)
# MAGIC - Tools are **MCP-compatible** for standardized discovery
# MAGIC
# MAGIC The agent definition will be extracted into `agent_app/agent.py` for deployment in Notebook 06.
# MAGIC
# MAGIC **Next**: [04 MLflow Tracing →](./04_mlflow_tracing)
