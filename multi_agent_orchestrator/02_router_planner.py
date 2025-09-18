# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 02_router_planner
# MAGIC Router → JSON plan (strict) with safe fallbacks.
# MAGIC
# MAGIC **Source**: Router contract from your spec 【7†files_uploaded_in_conversation】

# COMMAND ----------

pip install databricks-sdk[openai]

# COMMAND ----------

# Import necessary libraries
import json
import requests
from typing import Dict, Any

from databricks.sdk import WorkspaceClient
from typing import Dict, Any
import json

class ChatDatabricksOpenAI:
    def __init__(self, model: str):
        self.model = model
        self.client = WorkspaceClient().serving_endpoints.get_open_ai_client()

    def invoke(self, prompt: str, temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an orchestration planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content

def route_query_openai(
    llm: ChatDatabricksOpenAI,
    user_query: str,
    user_role: str = "analyst"
) -> Dict[str, Any]:
    routing_prompt = (
        "Return ONLY JSON with keys: "
        "primary_intent, complexity, required_agents, context_needs, "
        "data_analysis, synthesis_requirements, validation_requirements.\n"
        f"Query: {user_query}\nUser Role: {user_role}"
    )
    raw = llm.invoke(routing_prompt, temperature=0.0)
    try:
        return json.loads(raw)
    except Exception:
        return {
            "primary_intent": "analytics",
            "required_agents": ["sales_analytics"],
            "data_analysis": {}
        }

# --- Usage ---
llm = ChatDatabricksOpenAI(model="databricks-claude-3-7-sonnet")
print(route_query_openai(llm, "Store #42 is underperforming — what should we do?"))

def route_query(
    llm: ChatDatabricks,
    user_query: str,
    user_role: str = "analyst"
) -> Dict[str, Any]:
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
        plan = {
            "primary_intent": "analytics",
            "required_agents": ["sales_analytics"],
            "data_analysis": {}
        }
    return plan

if __name__ == "__main__":
    db_host = "https://adb-984752964297111.11.azuredatabricks.net"
    db_token = dbutils.secrets.get(scope="chipotle-analytics", key="claude-key")
    llm = ChatDatabricks(
        endpoint="databricks-gpt-oss-120b",
        databricks_host=db_host,
        databricks_token=db_token
    )
    print(route_query(llm, "Store #42 is underperforming — what should we do?"))

# COMMAND ----------

# Replace 'my-scope' and 'my-secret' with your actual scope and secret names
secret_scope = "chipotle-analytics"
secret_name = "claude-key"

try:
    # Attempt to retrieve the secret
    secret_value = dbutils.secrets.get(scope=secret_scope, key=secret_name)
    
    # Verify retrieval by checking its length or type
    if secret_value:
        print("Secret successfully retrieved! ✅")
        # Print a portion of the secret for verification, e.g., the first 5 characters
        print(f"Secret value snippet: {secret_value[:5]}...") 
    else:
        print("Secret retrieved, but value is empty or None. ❌")
        
except Exception as e:
    print(f"Failed to retrieve secret. Error: {e} ❌")
