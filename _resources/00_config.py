# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Configuration
# MAGIC
# MAGIC Set these values before running the workshop — edit `catalog` / `schema`
# MAGIC to point at a Unity Catalog location you can write to. All downstream
# MAGIC notebooks reference these variables via `%run ../_resources/00_config`.

# COMMAND ----------

# ---- Catalog & Schema ----
# Edit these to your own Unity Catalog location. `main` exists in most
# workspaces; if you don't have rights there, set `catalog` to one you can use.
catalog = "main"
schema = "churn_workshop"

# ---- Foundation Model API Endpoints ----
llm_endpoint = "databricks-claude-sonnet-4-6"
embedding_endpoint = "databricks-gte-large-en"

# ---- Vector Search ----
vector_search_endpoint = "workshop_vs_endpoint"
product_knowledge_index = f"{catalog}.{schema}.product_knowledge_index"

# ---- Model Serving ----
churn_model_serving_endpoint = "workshop-churn-model"

# COMMAND ----------

# Derived references (do not edit)
feature_table_name = f"{catalog}.{schema}.churn_feature_table"
model_name = f"{catalog}.{schema}.churn_model"
predictions_table_name = f"{catalog}.{schema}.churn_predictions"
documents_volume = f"{catalog}.{schema}.documents"
documents_volume_path = f"/Volumes/{catalog}/{schema}/documents"
documents_source_path = f"{documents_volume_path}/source"
parsed_images_path = f"{documents_volume_path}/parsed_images"

# COMMAND ----------

# Set default catalog and schema for SQL (skip if not yet created by setup)
try:
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"USE SCHEMA {schema}")
except Exception:
    print(f"Note: catalog '{catalog}' or schema '{schema}' not yet created — run 01_setup first")

print(f"Catalog:            {catalog}")
print(f"Schema:             {schema}")
print(f"LLM Endpoint:       {llm_endpoint}")
print(f"Embedding Endpoint: {embedding_endpoint}")
print(f"Feature Table:      {feature_table_name}")
print(f"Model:              {model_name}")

# COMMAND ----------

# ---- Grant helpers ---------------------------------------------------------
# Every asset the workshop creates needs to be readable by all workshop users.
# UC GRANT is naturally idempotent (re-running is a no-op), so callers can
# invoke these every run without guarding. The non-UC helpers use the
# Permissions REST API with PATCH semantics (additive, also idempotent).

PARTICIPANTS_PRINCIPAL = "`account users`"
PARTICIPANTS_GROUP_SDK = "users"  # workspace-level group that maps to account users

def _grant_sql(stmt: str, label: str):
    try:
        spark.sql(stmt)
        print(f"  ✓ granted {label}")
    except Exception as e:
        print(f"  ⚠ {label}: {e}")

def grant_table(fqn):
    _grant_sql(f"GRANT SELECT ON TABLE {fqn} TO {PARTICIPANTS_PRINCIPAL}", f"SELECT on {fqn}")

def grant_view(fqn):
    _grant_sql(f"GRANT SELECT ON VIEW {fqn} TO {PARTICIPANTS_PRINCIPAL}", f"SELECT on {fqn}")

def grant_function(fqn):
    _grant_sql(f"GRANT EXECUTE ON FUNCTION {fqn} TO {PARTICIPANTS_PRINCIPAL}", f"EXECUTE on {fqn}")

def grant_model(fqn):
    _grant_sql(f"GRANT EXECUTE ON MODEL {fqn} TO {PARTICIPANTS_PRINCIPAL}", f"EXECUTE on {fqn}")

def grant_volume_read(fqn):
    _grant_sql(f"GRANT READ VOLUME ON VOLUME {fqn} TO {PARTICIPANTS_PRINCIPAL}", f"READ VOLUME on {fqn}")

def grant_volume_write(fqn):
    _grant_sql(f"GRANT WRITE VOLUME ON VOLUME {fqn} TO {PARTICIPANTS_PRINCIPAL}", f"WRITE VOLUME on {fqn}")

def _grant_permissions_api(object_type: str, object_id: str, permission_level: str, label: str):
    """PATCH /api/2.0/permissions/{object_type}/{object_id} — additive, idempotent."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        w.api_client.do(
            "PATCH",
            f"/api/2.0/permissions/{object_type}/{object_id}",
            body={
                "access_control_list": [
                    {"group_name": PARTICIPANTS_GROUP_SDK, "permission_level": permission_level}
                ]
            },
        )
        print(f"  ✓ granted {permission_level} on {label}")
    except Exception as e:
        print(f"  ⚠ {label}: {e}")

def grant_serving_endpoint(name):
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        endpoint = w.serving_endpoints.get(name)
        _grant_permissions_api("serving-endpoints", endpoint.id, "CAN_QUERY", f"serving endpoint {name}")
    except Exception as e:
        print(f"  ⚠ serving endpoint {name}: {e}")

def grant_vector_search_endpoint(name):
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        endpoint = w.vector_search_endpoints.get_endpoint(name)
        endpoint_id = getattr(endpoint, "id", None) or getattr(endpoint, "endpoint_id", None)
        if not endpoint_id:
            print(f"  ⚠ vector search endpoint {name}: could not resolve endpoint id")
            return
        _grant_permissions_api("vector-search-endpoints", endpoint_id, "CAN_USE", f"vector search endpoint {name}")
    except Exception as e:
        print(f"  ⚠ vector search endpoint {name}: {e}")

def grant_genie_space(space_id):
    """Open the Genie space to All Account Users (CAN_RUN). Verifies the ACL via GET."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        w.api_client.do(
            "PATCH",
            f"/api/2.0/permissions/genie/{space_id}",
            body={
                "access_control_list": [
                    {"group_name": PARTICIPANTS_GROUP_SDK, "permission_level": "CAN_RUN"}
                ]
            },
        )
        # Verify the ACL actually contains `users` — surfaces silent endpoint mismatches.
        result = w.api_client.do("GET", f"/api/2.0/permissions/genie/{space_id}") or {}
        acl = result.get("access_control_list") or []
        users_acl = next((a for a in acl if a.get("group_name") == PARTICIPANTS_GROUP_SDK), None)
        perms = [p.get("permission_level") for p in (users_acl.get("all_permissions") or [])] if users_acl else []
        if "CAN_RUN" in perms or "CAN_MANAGE" in perms:
            print(f"  ✓ granted CAN_RUN on genie space {space_id} (verified)")
        else:
            print(f"  ⚠ genie space {space_id}: PATCH returned 200 but ACL does not include `{PARTICIPANTS_GROUP_SDK}` (got: {acl})")
    except Exception as e:
        print(f"  ⚠ genie space {space_id}: {e}")
