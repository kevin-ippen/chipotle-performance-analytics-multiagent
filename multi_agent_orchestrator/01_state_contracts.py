# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 01_state_contracts
# MAGIC Typed state and validation helpers.
# MAGIC
# MAGIC **Source**: Orchestration state structure specified in your doc 【7†files_uploaded_in_conversation】

# COMMAND ----------

from typing import Any, Dict, List, Optional, TypedDict

class AgentState(TypedDict, total=False):
    conversation_id: str
    user_query: str
    user_context: Dict[str, Any]
    orchestration_plan: Dict[str, Any]
    current_step: str
    strategic_context: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    synthesized_response: str
    validation_results: str
    final_response: str
    execution_time_ms: float
    agents_used: List[str]
    confidence_score: float
    token_usage: Dict[str, int]
    cost_estimate_usd: float
    trace_ids: Dict[str, str]

def ensure_keys(d: Dict[str, Any], keys: List[str]) -> None:
    for k in keys:
        if k not in d:
            d[k] = None

print("AgentState schema loaded.")
