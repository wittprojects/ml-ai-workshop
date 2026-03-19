# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 05 — Agent Evaluation
# MAGIC
# MAGIC **Time**: ~8 min
# MAGIC
# MAGIC Evaluate agent quality with `mlflow.genai.evaluate()` using built-in and custom scorers.
# MAGIC
# MAGIC Key concepts:
# MAGIC - Evaluation datasets with expected outputs
# MAGIC - Built-in scorers: relevance, groundedness, safety
# MAGIC - Custom guideline scorers
# MAGIC - Comparing agent versions in MLflow UI

# COMMAND ----------

# MAGIC %pip install databricks-langchain langgraph "mlflow[genai]" databricks-sdk>=0.50.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

import mlflow

experiment_path = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/ml-ai-workshop-genai"
mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build an Agent

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
# MAGIC ## Create Evaluation Dataset

# COMMAND ----------

import pandas as pd

eval_data = pd.DataFrame([
    {
        "inputs": {"messages": [{"role": "user", "content": "Customer CUST-00050 wants to cancel because the price is too high. What should I do?"}]},
        "expected_facts": ["should look up customer profile", "should check churn risk", "should recommend a retention offer", "should reference retention policy"],
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "Help me with the latest escalated ticket."}]},
        "expected_facts": ["should retrieve the latest ticket", "should identify the customer", "should provide a recommended action"],
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "CUST-00100 is complaining about slow internet. How can I help them?"}]},
        "expected_facts": ["should look up customer profile", "should search knowledge base for troubleshooting", "should provide technical solution"],
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "What retention offers can I make to a customer who's been with us for 2 years?"}]},
        "expected_facts": ["should reference retention policy", "should mention discount or loyalty benefits", "should mention tenure-based offers"],
    },
    {
        "inputs": {"messages": [{"role": "user", "content": "Customer CUST-00200 has filed 5 billing tickets this quarter. They're fed up. Help me save this account."}]},
        "expected_facts": ["should review ticket history", "should check churn risk", "should recommend billing resolution", "should offer a retention incentive"],
    },
])

print(f"Evaluation dataset: {len(eval_data)} examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Predict Function

# COMMAND ----------

def predict_fn(inputs):
    response = agent.invoke(inputs)
    return response["messages"][-1].content

# Quick test
test_result = predict_fn(eval_data.iloc[0]["inputs"])
print(f"Test prediction: {test_result[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation with Scorers

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines

# Custom guidelines for our retention use case
retention_guidelines = Guidelines(
    name="retention_quality",
    guidelines=[
        "The response should include a specific, actionable retention recommendation",
        "The response should reference customer data (profile, history, or churn risk) to justify the recommendation",
        "The response should be empathetic and professional in tone",
        "The response should mention specific offers or discounts when applicable",
    ],
)

# COMMAND ----------

# Run evaluation
mlflow.langchain.autolog()

# Wrap inputs to match predict_fn parameter name
eval_data_wrapped = eval_data.copy()
eval_data_wrapped["inputs"] = eval_data_wrapped["inputs"].apply(lambda x: {"inputs": x})

with mlflow.start_run(run_name="agent_eval_v1"):
    eval_results = mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data_wrapped,
        scorers=[
            RelevanceToQuery(),
            Safety(),
            retention_guidelines,
        ],
    )

print("✓ Evaluation complete")

# COMMAND ----------

# Display results
display(eval_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Iterate: Improved System Prompt

# COMMAND ----------

# Create an improved agent with a more detailed system prompt
IMPROVED_PROMPT = """You are a telecom customer retention specialist assistant. Your job is to help service representatives retain customers who are at risk of churning.

IMPORTANT RULES:
- Always look up the customer's profile and churn risk before making recommendations
- Always check the retention policy to ensure your offers are within guidelines
- For high-risk customers (churn score > 0.7), prioritize aggressive retention offers
- For billing complaints, always offer to review and adjust the plan
- For technical issues, search the knowledge base first, then offer service credits
- Always address the customer by name
- Structure your response as: 1) Situation Summary, 2) Key Findings, 3) Recommended Action

When a customer calls in, follow this workflow:
1. Look up the latest escalated ticket to identify the customer and issue
2. Get the customer's profile to understand their account
3. Check their churn risk score from the ML model
4. Review their ticket history for patterns
5. Search the knowledge base for relevant solutions
6. Check retention policies to know what offers you can make
7. Synthesize all information into a recommended retention action"""

improved_agent = create_react_agent(
    model=llm,
    tools=uc_toolkit.tools + [vs_tool],
    prompt=IMPROVED_PROMPT,
)

def improved_predict_fn(inputs):
    response = improved_agent.invoke(inputs)
    return response["messages"][-1].content

# COMMAND ----------

# Evaluate the improved agent
with mlflow.start_run(run_name="agent_eval_v2_improved"):
    improved_results = mlflow.genai.evaluate(
        predict_fn=improved_predict_fn,
        data=eval_data_wrapped,
        scorers=[
            RelevanceToQuery(),
            Safety(),
            retention_guidelines,
        ],
    )

print("✓ Improved agent evaluation complete")

# COMMAND ----------

# Compare results
display(improved_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare Versions in MLflow UI
# MAGIC
# MAGIC Navigate to the MLflow experiment to compare the two evaluation runs side-by-side:
# MAGIC 1. Select both runs (`agent_eval_v1` and `agent_eval_v2_improved`)
# MAGIC 2. Click **Compare**
# MAGIC 3. Review score distributions and per-example results
# MAGIC
# MAGIC This is the core workflow for iterative agent development: **change → evaluate → compare → repeat**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We evaluated our agent with:
# MAGIC - **5 test scenarios** covering different retention situations
# MAGIC - **Built-in scorers**: RelevanceToQuery, Safety
# MAGIC - **Custom scorer**: retention_quality with domain-specific guidelines
# MAGIC - **Two versions compared** to measure improvement from prompt changes
# MAGIC
# MAGIC **Congratulations!** You've completed Module 2. You now have an evaluated, deployed GenAI agent built entirely on Databricks.
