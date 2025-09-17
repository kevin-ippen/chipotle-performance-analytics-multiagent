# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 00_config_setup
# MAGIC - Installs libs (if needed)
# MAGIC - Loads `orchestrator.yaml`
# MAGIC - Creates UC tables for traces/observability
# MAGIC - Configures Databricks Secrets references
# MAGIC #
# MAGIC **Source**: Derived from your orchestration spec (uploaded file) 【7†files_uploaded_in_conversation】

# COMMAND ----------

# Optional: lightweight installs (no-ops if already on cluster)
# %pip install pydantic langgraph aiohttp

import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# ---- CONFIG ----
def widget_names():
    try:
        return [w.name for w in dbutils.widgets.get()]
    except Exception:
        return []

CATALOG = "main"
SCHEMA  = "orchestrator"
try:
    if "CATALOG" in widget_names():
        CATALOG = dbutils.widgets.get("CATALOG")
    if "SCHEMA" in widget_names():
        SCHEMA = dbutils.widgets.get("SCHEMA")
except Exception as e:
    print("Widget access skipped:", e)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# ---- YAML CONFIG PATH ----
yaml_path = "/dbfs/FileStore/multi_agent/orchestrator.yaml"
if not os.path.exists(yaml_path):
    print("Config not found at", yaml_path)
    print("➡️ Please upload orchestrator.yaml to /FileStore/multi_agent (DBFS).")
else:
    print("Using config at:", yaml_path)
    display(dbutils.fs.ls("dbfs:/FileStore/multi_agent"))

# ---- Observability Tables ----
spark.sql(f'''
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.orchestrator_traces (
  ts TIMESTAMP,
  conversation_id STRING,
  node STRING,
  model STRING,
  latency_ms DOUBLE,
  prompt_hash STRING,
  token_input INT,
  token_output INT,
  cost_usd DOUBLE,
  trace_id STRING,
  meta MAP<STRING, STRING>
) USING DELTA
''')

spark.sql(f'''
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.orchestrator_cache (
  key STRING,
  value STRING,
  created_ts TIMESTAMP
) USING DELTA
''')

display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}"))
