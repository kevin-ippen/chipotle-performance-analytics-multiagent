# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # 05_orchestrator_core
# MAGIC LangGraph-like state machine wiring (route → context → analyze → synthesize → validate).
# MAGIC
# MAGIC **Source**: Graph & node flow from your spec 【7†files_uploaded_in_conversation】

# COMMAND ----------

import json, asyncio
from typing import Dict, Any

from 02_router_planner import ChatDatabricks, route_query
from 03_knowledge_agents import StrategyKnowledgeAgent, OperationsKnowledgeAgent, ResearchKnowledgeAgent
from 04_sales_analytics_agent import SalesAnalyticsAgent

class Orchestrator:
    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.llm_supervisor = ChatDatabricks(endpoint=self.cfg.get("models.supervisor_endpoint", "databricks-claude-sonnet-4"))
        self.agents = {
            "sales_analytics": SalesAnalyticsAgent(),
            "knowledge_strategy": StrategyKnowledgeAgent(),
            "knowledge_operations": OperationsKnowledgeAgent(),
            "knowledge_research": ResearchKnowledgeAgent(),
        }

    def _determine_knowledge_agent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["strategy","priority","expansion","positioning"]): return "knowledge_strategy"
        if any(k in q for k in ["operations","constraint","best practice","playbook"]): return "knowledge_operations"
        return "knowledge_research"

    async def gather_context(self, user_query: str) -> Dict[str, Any]:
        context_queries = []
        ql = user_query.lower()
        if "store" in ql:
            context_queries += [
                "current store performance improvement strategies",
                "operational constraints for underperforming stores"
            ]
        if "market" in ql:
            context_queries += ["market expansion priorities"]
        tasks, labels = [], []
        for cq in context_queries:
            agent_key = self._determine_knowledge_agent(cq)
            tasks.append(self.agents[agent_key].ainvoke(cq))
            labels.append(cq)
        results = await asyncio.gather(*tasks) if tasks else []
        return {labels[i]: results[i] for i in range(len(results))}

    def analyze(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        res = {}
        for gq in plan.get("data_analysis",{}).get("genie_queries", []):
            res[f"genie_{gq.get('space')}"] = self.agents["sales_analytics"].query_genie_space(gq["space"], gq["query"])
        for task in plan.get("data_analysis",{}).get("ml_tasks", []):
            res[f"ml_{task.get('tool')}"] = self.agents["sales_analytics"].call_ml_tool(task["tool"], task.get("parameters", {}))
        return res

    def synthesize(self, user_query: str, user_context: Dict[str, Any], strategic_context: Dict[str, Any], analysis_results: Dict[str, Any]) -> str:
        # TODO: Replace with LLM call to supervisor; for now return formatted markdown
        return f"""## Answer
- Direct response to: **{user_query}**

## Action Plan (Now/Next/Later)
- **Now**: Review KPIs and drivers from Genie payloads.
- **Next**: Run forecast and waste anomaly tools.
- **Later**: Validate strategy alignment and ops constraints.

## Risks
- Data freshness; model drift.

## Success Metrics
- Rev lift %, waste reduction %, CSAT.
"""

    def validate(self, synthesized: str, strategic_context: Dict[str, Any]) -> str:
        # TODO: Replace with validator LLM
        return "Validation: Recommendations appear feasible given current strategic context (stub)."

    def invoke(self, conversation_id: str, user_query: str, user_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        user_context = user_context or {"role":"analyst"}
        plan = route_query(ChatDatabricks("databricks-llama-3-8b-instruct"), user_query, user_context.get("role","analyst"))
        ctx = asyncio.get_event_loop().run_until_complete(self.gather_context(user_query))
        analysis = self.analyze(plan)
        synthesized = self.synthesize(user_query, user_context, ctx, analysis)
        validation = self.validate(synthesized, ctx)
        final_md = "# Recommendations\n\n" + synthesized + "\n\n---\n\n**Validation**\n\n" + validation
        return {"final": final_md, "plan": plan, "context": ctx, "analysis": analysis}

if __name__ == "__main__":
    orchestrator = Orchestrator()
    out = orchestrator.invoke("demo-1", "Store #42 is underperforming — what should we do?", {"role":"ops_manager"})
    print(out["final"][:240], "...")
