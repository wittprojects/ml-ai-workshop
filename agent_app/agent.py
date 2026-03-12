"""
Telecom Customer Retention Agent

LangGraph agent with UC function tools and vector search.
Extracted from module_2_genai/03_build_agent.py for deployment as a Databricks App.
"""

import os
from databricks_langchain import ChatDatabricks, UCFunctionToolkit, VectorSearchRetrieverTool
from langgraph.prebuilt import create_react_agent

# Configuration — reads from environment or uses defaults
CATALOG = os.environ.get("CATALOG", "ml_ai_workshop")
SCHEMA = os.environ.get("SCHEMA", "workshop")
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-claude-sonnet-4-6")
VS_INDEX = os.environ.get("VS_INDEX", f"{CATALOG}.{SCHEMA}.product_knowledge_index")

SYSTEM_PROMPT = """You are a telecom customer retention specialist assistant. Your job is to help service representatives retain customers who are at risk of churning.

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


def create_agent():
    """Create and return the retention agent."""
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)

    uc_toolkit = UCFunctionToolkit(
        function_names=[
            f"{CATALOG}.{SCHEMA}.get_latest_ticket",
            f"{CATALOG}.{SCHEMA}.get_customer_profile",
            f"{CATALOG}.{SCHEMA}.get_ticket_history",
            f"{CATALOG}.{SCHEMA}.get_retention_policy",
            f"{CATALOG}.{SCHEMA}.get_churn_risk",
        ]
    )

    vs_tool = VectorSearchRetrieverTool(
        index_name=VS_INDEX,
        tool_name="search_knowledge_base",
        tool_description="Search the product knowledge base for troubleshooting guides, plan comparisons, feature info, and FAQs.",
        columns=["article_id", "title", "content", "category"],
        num_results=3,
    )

    agent = create_react_agent(
        model=llm,
        tools=uc_toolkit.tools + [vs_tool],
        prompt=SYSTEM_PROMPT,
    )

    return agent
