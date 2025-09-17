# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 07_eval_harness
# MAGIC Tiny evaluation harness and Delta logging.
# MAGIC
# MAGIC **Source**: Eval harness sketch from your spec 【7†files_uploaded_in_conversation】

# COMMAND ----------

try:
    from 05_orchestrator_core import Orchestrator
    from pyspark.sql import Row
    spark
except NameError:
    # Allow running outside interactive cluster for import checks
    class Row(dict):
        pass

from 05_orchestrator_core import Orchestrator

cases = [
  {"q": "Store #42 is underperforming. What should we do?"},
  {"q": "Should we open in downtown Denver?"},
  {"q": "How will a 5% price increase affect CSAT and revenue?"}
]

orch = Orchestrator()
rows = []
for i, c in enumerate(cases, 1):
    out = orch.invoke(f"eval-{i}", c["q"], {"role":"analyst"})
    rows.append(Row(case_id=i, query=c["q"], response=out["final"]))

try:
    df = spark.createDataFrame(rows)
    display(df)
except Exception as e:
    print("Spark not available in this environment; showing raw rows instead.")
    print(rows[:1])
