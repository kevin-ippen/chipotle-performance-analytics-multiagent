# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-Agent Orchestrator Setup & Configuration
# MAGIC
# MAGIC This notebook sets up the foundational components for your Strategic Business Intelligence multi-agent system.
# MAGIC
# MAGIC ## What we'll accomplish:
# MAGIC 1. Install dependencies and setup project structure
# MAGIC 2. Configure Databricks integrations (Models, Vector Search, Genie)
# MAGIC 3. Create initial knowledge base indices
# MAGIC 4. Test basic connectivity to all services
# MAGIC 5. Load sample strategic documents

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup

# COMMAND ----------

# Install required packages
%pip install databricks-sdk>=0.18.0 langchain>=0.1.0 langgraph>=0.0.40 pydantic>=2.0.0 mlflow>=2.8.0 PyYAML>=6.0

# COMMAND ----------

# Restart Python environment
dbutils.library.restartPython()

# COMMAND ----------

import os
import yaml
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.vector_search.client import VectorSearchClient
import mlflow

# Initialize Databricks clients
w = WorkspaceClient()
vs_client = VectorSearchClient()

print("✅ Databricks SDK initialized")
print(f"✅ Workspace URL: {w.config.host}")
print(f"✅ Current user: {w.current_user.me().display_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Project Configuration

# COMMAND ----------

# Create configuration structure
config = {
    "workspace": {
        "host": w.config.host,
        "workspace_id": "your_workspace_id"  # Update this
    },
    "models": {
        "supervisor_endpoint": "databricks-meta-llama-3-1-405b-instruct",
        "small_router_endpoint": "databricks-meta-llama-3-1-70b-instruct", 
        "embedding_endpoint": "databricks-bge-large-en"
    },
    "genie": {
        "spaces": {
            "ExecutiveDashboard": {
                "id": "exec_kpis_space",
                "description": "Executive KPIs and high-level metrics"
            },
            "StorePerformance": {
                "id": "store_perf_space", 
                "description": "Individual store performance metrics"
            },
            "CustomerAnalytics": {
                "id": "customer_analytics_space",
                "description": "Customer behavior and segmentation"
            },
            "MarketIntel": {
                "id": "market_intel_space",
                "description": "Market trends and competitive intelligence"
            }
        }
    },
    "vector_search": {
        "endpoint_name": "pizza_intelligence_endpoint",
        "indices": {
            "strategy_knowledge": {
                "index_name": "main.pizza_intelligence.strategy_docs",
                "description": "Strategic plans, priorities, market expansion playbooks"
            },
            "operations_knowledge": {
                "index_name": "main.pizza_intelligence.operations_docs", 
                "description": "Operations manuals, best practices, procedures"
            },
            "research_knowledge": {
                "index_name": "main.pizza_intelligence.research_docs",
                "description": "A/B tests, customer research, learnings, benchmarks"
            }
        }
    },
    "ml_tools": {
        "demand_forecast_v1": {
            "serving_endpoint": "demand-forecast-v1",
            "description": "28-day demand forecasting for stores"
        },
        "cannibalization_v2": {
            "serving_endpoint": "cannibalization-v2", 
            "description": "Market cannibalization analysis for new locations"
        },
        "customer_lifetime_value": {
            "serving_endpoint": "clv-model-v1",
            "description": "Customer lifetime value prediction"
        }
    },
    "limits": {
        "mode_defaults": {
            "fast": {"max_agents": 2, "max_docs": 6, "timeout_s": 20},
            "balanced": {"max_agents": 3, "max_docs": 10, "timeout_s": 45}, 
            "thorough": {"max_agents": 5, "max_docs": 16, "timeout_s": 90}
        },
        "retries": {"max_attempts": 2, "backoff": "exponential"},
        "circuit_breaker": {"error_rate_pct": 35, "rolling_window_s": 60}
    }
}

# Save configuration to DBFS
config_path = "/dbfs/pizza_intelligence/config/orchestrator.yaml"
os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ Configuration saved to:", config_path)
print("\n📋 Current configuration:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Foundation Model Access

# COMMAND ----------

def test_foundation_model(endpoint_name: str) -> bool:
    """Test if we can access a foundation model endpoint"""
    try:
        # Test with a simple query
        response = w.serving_endpoints.query(
            name=endpoint_name,
            messages=[
                ChatMessage(role=ChatMessageRole.USER, content="Hello, can you respond with just 'success'?")
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        success = "success" in result
        
        print(f"✅ {endpoint_name}: {response.choices[0].message.content}")
        return success
        
    except Exception as e:
        print(f"❌ {endpoint_name}: {str(e)}")
        return False

# Test
