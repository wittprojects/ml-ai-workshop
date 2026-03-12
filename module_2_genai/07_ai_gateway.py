# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: GenAI Development & Deployment
# MAGIC ## Notebook 07 — AI Gateway
# MAGIC
# MAGIC **Time**: ~5 min
# MAGIC
# MAGIC **AI Gateway** (Mosaic AI Gateway) provides governance, guardrails, and monitoring for LLM traffic.
# MAGIC
# MAGIC Key concepts:
# MAGIC - AI Gateway routes for FMAPI endpoints
# MAGIC - Rate limiting and usage tracking
# MAGIC - Content guardrails (safety filters, PII detection)
# MAGIC - Gateway logs and metrics

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is AI Gateway?
# MAGIC
# MAGIC AI Gateway sits between your application and LLM endpoints, providing:
# MAGIC
# MAGIC | Feature | Description |
# MAGIC |---------|-------------|
# MAGIC | **Rate Limiting** | Control requests per minute/hour per user or app |
# MAGIC | **Usage Tracking** | Token consumption by user, app, and endpoint |
# MAGIC | **Guardrails** | Content safety filters and PII detection |
# MAGIC | **Routing** | A/B testing, fallback endpoints |
# MAGIC | **Logging** | Full request/response logging for audit |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure AI Gateway on an Endpoint
# MAGIC
# MAGIC We can add AI Gateway configuration to any serving endpoint, including FMAPI endpoints.

# COMMAND ----------

from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayRateLimit,
    AiGatewayRateLimitRenewalPeriod,
    AiGatewayRateLimitKey,
    AiGatewayGuardrails,
    AiGatewayGuardrailParameters,
    AiGatewayGuardrailPiiParameters,
    AiGatewayUsageTrackingConfig,
    AiGatewayInferenceTableConfig,
)

# COMMAND ----------

# Configure AI Gateway on a serving endpoint
gateway_config = AiGatewayConfig(
    # Rate limiting
    rate_limits=[
        AiGatewayRateLimit(
            calls=100,
            renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
            key=AiGatewayRateLimitKey.USER,
        ),
    ],
    # Usage tracking
    usage_tracking_config=AiGatewayUsageTrackingConfig(
        enabled=True,
    ),
    # Guardrails
    guardrails=AiGatewayGuardrails(
        input=AiGatewayGuardrailParameters(
            safety=True,
            pii=AiGatewayGuardrailPiiParameters(behavior="BLOCK"),
        ),
        output=AiGatewayGuardrailParameters(
            safety=True,
            pii=AiGatewayGuardrailPiiParameters(behavior="BLOCK"),
        ),
    ),
    # Inference logging
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name=catalog,
        schema_name=schema,
        table_name_prefix="gateway_logs",
        enabled=True,
    ),
)

print("AI Gateway configuration defined:")
print(f"  Rate limit: 100 calls/minute per user")
print(f"  Usage tracking: enabled")
print(f"  Input guardrails: safety=True, PII=BLOCK")
print(f"  Output guardrails: safety=True, PII=BLOCK")
print(f"  Inference logging: {catalog}.{schema}.gateway_logs_*")

# COMMAND ----------

# Apply gateway config to the churn model serving endpoint
try:
    w.serving_endpoints.put_ai_gateway(
        name=churn_model_serving_endpoint,
        ai_gateway=gateway_config,
    )
    print(f"✓ AI Gateway configured on '{churn_model_serving_endpoint}'")
except Exception as e:
    print(f"Note: {e}")
    print("\nAI Gateway may not be available on all endpoint types.")
    print("It is most commonly used with FMAPI and external model endpoints.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Guardrails
# MAGIC
# MAGIC Let's test the safety guardrails by sending requests through a gateway-enabled endpoint.

# COMMAND ----------

# Normal request (should pass)
try:
    response = w.serving_endpoints.query(
        name=llm_endpoint,
        messages=[{"role": "user", "content": "What are good strategies for customer retention in telecom?"}],
        max_tokens=200,
    )
    print("✓ Normal request passed:")
    print(f"  {response.choices[0].message.content[:200]}...")
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review Gateway Usage
# MAGIC
# MAGIC AI Gateway usage data is available via system tables and the serving endpoint API.

# COMMAND ----------

# Check endpoint usage via API
try:
    endpoint_info = w.serving_endpoints.get(name=churn_model_serving_endpoint)
    if endpoint_info.ai_gateway:
        print("AI Gateway status:")
        print(f"  Rate limits: {endpoint_info.ai_gateway.rate_limits}")
        print(f"  Usage tracking: {endpoint_info.ai_gateway.usage_tracking_config}")
        print(f"  Guardrails: {endpoint_info.ai_gateway.guardrails}")
    else:
        print("AI Gateway not yet configured on this endpoint")
except Exception as e:
    print(f"Note: {e}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check inference/gateway logs (will populate after requests flow through)
# MAGIC -- SHOW TABLES LIKE 'gateway_logs*'

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Gateway Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────┐     ┌───────────────┐     ┌──────────────┐
# MAGIC │   Your App  │────▶│  AI Gateway   │────▶│ FMAPI / LLM  │
# MAGIC │  (Agent)    │     │               │     │  Endpoint     │
# MAGIC └─────────────┘     │ • Rate limits │     └──────────────┘
# MAGIC                     │ • Guardrails  │
# MAGIC                     │ • PII filter  │     ┌──────────────┐
# MAGIC                     │ • Logging     │────▶│ Inference    │
# MAGIC                     │ • Routing     │     │ Table (logs) │
# MAGIC                     └───────────────┘     └──────────────┘
# MAGIC ```
# MAGIC
# MAGIC In production, AI Gateway is essential for:
# MAGIC - **Cost control**: Rate limits prevent runaway token usage
# MAGIC - **Compliance**: PII detection ensures sensitive data doesn't leak to LLMs
# MAGIC - **Safety**: Content filters block harmful inputs and outputs
# MAGIC - **Audit**: Full request/response logging for governance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Module 2 Complete! 🎉
# MAGIC
# MAGIC We built an end-to-end GenAI application:
# MAGIC
# MAGIC | Step | What We Did |
# MAGIC |------|-------------|
# MAGIC | 01 AI Functions | Used ai_query(), ai_extract(), ai_sentiment() in SQL |
# MAGIC | 02 Create Tools | Built UC functions and vector search as agent tools |
# MAGIC | 03 Build Agent | LangGraph retention agent with 6 tools |
# MAGIC | 04 MLflow Tracing | Full observability for every agent action |
# MAGIC | 05 Agent Eval | Systematic evaluation with custom scorers |
# MAGIC | 06 Deploy to Apps | Live web application via Databricks Apps |
# MAGIC | 07 AI Gateway | Guardrails, rate limits, and governance |
# MAGIC
# MAGIC ### Workshop Complete!
# MAGIC
# MAGIC The ML churn model from Module 1 became a callable tool in the Module 2 agent.
# MAGIC This demonstrates how **classical ML** and **GenAI** work together on Databricks:
# MAGIC - ML provides **predictions** (churn risk score)
# MAGIC - GenAI provides **reasoning** (retention strategy)
# MAGIC - Together they power an **intelligent application** (retention agent)
