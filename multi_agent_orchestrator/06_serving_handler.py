# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 06_serving_handler
# MAGIC Minimal handler to deploy via Databricks Model Serving or Apps.
# MAGIC
# MAGIC **Source**: Serving pattern from your spec 【7†files_uploaded_in_conversation】

# COMMAND ----------

from typing import Dict, Any
from 05_orchestrator_core import Orchestrator

_orch = Orchestrator()

def predict(request: Dict[str, Any]) -> Dict[str, Any]:
    # Expected: { "query": "...", "user_context": {...}, "conversation_id": "..." }
    query = request.get("query", "")
    user_context = request.get("user_context", {})
    conversation_id = request.get("conversation_id", "serving")
    result = _orch.invoke(conversation_id, query, user_context)
    return {"final_response": result["final"], "plan": result["plan"]}
