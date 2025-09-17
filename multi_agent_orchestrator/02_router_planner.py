# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 02_router_planner
# MAGIC Router → JSON plan (strict) with safe fallbacks.
# MAGIC
# MAGIC **Source**: Router contract from your spec 【7†files_uploaded_in_conversation】

# COMMAND ----------

import json
from typing import Dict, Any

# Placeholder LLM invoker — swap to Databricks model serving client
class ChatDatabricks:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    def invoke(self, prompt: str, temperature: float = 0.1) -> str:
        # TODO: implement REST call to model serving endpoint
        # For now, return a deterministic minimal plan
        return json.dumps({
            "primary_intent": "analytics",
            "complexity": "multi_agent_analysis",
            "required_agents": ["sales_analytics"],
            "context_needs": [],
            "data_analysis": {"genie_queries": [], "ml_tasks": []},
            "synthesis_requirements": ["actionable"],
            "validation_requirements": ["feasibility"]
        })

def route_query(llm: ChatDatabricks, user_query: str, user_role: str = "analyst") -> Dict[str, Any]:
    routing_prompt = f"""
You are an orchestration planner. Return ONLY JSON with keys:
primary_intent, complexity, required_agents, context_needs, data_analysis, synthesis_requirements, validation_requirements.
Query: {user_query}
User Role: {user_role}
"""
    raw = llm.invoke(routing_prompt, temperature=0.0)
    try:
        plan = json.loads(raw)
    except Exception:
        plan = {"primary_intent": "analytics",
                "required_agents": ["sales_analytics"],
                "data_analysis": {}}
    return plan

# Demo (safe to run)
if __name__ == "__main__":
    llm = ChatDatabricks(endpoint="databricks-llama-3-8b-instruct")
    print(route_query(llm, "Store #42 is underperforming — what should we do?"))
