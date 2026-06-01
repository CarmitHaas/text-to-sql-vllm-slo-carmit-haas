"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

TODO markers indicate what you implement. The execute node and the
graph wiring around it are provided.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    """Chat client pointed at the local vLLM endpoint."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="not-needed",  # vLLM ignores the key, but the SDK requires a string
        temperature=0.0,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start."""
    return {"schema": render_schema(state.db_id)}


def generate_sql_node(state: AgentState) -> dict:
    """TODO (Phase 3, step 1).

    - Build messages using GENERATE_SQL_SYSTEM / GENERATE_SQL_USER from
      agent.prompts, filling in state.schema and state.question.
    - Call llm().invoke(messages).
    - Strip any ```sql ... ``` fencing the model emits.
    - Return {"sql": <sql>, "iteration": state.iteration + 1, "history": ...}.

    The history field is yours to use - one entry per iteration with
    whatever info you'd want to inspect later (sql, exec result, verify
    verdict). The Final deliverables make use of per-iteration data, so
    populate it now.
    """
    raise NotImplementedError("Implement in Phase 3")


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    """TODO (Phase 3, step 1).

    - Render state.execution via ExecutionResult.render().
    - Build messages using VERIFY_SYSTEM / VERIFY_USER, filling in
      question, sql, execution_result.
    - Call llm() with response_format={"type": "json_object"} so the
      response is parseable JSON.
    - Parse {"ok": bool, "issue": str}.
    - Return {"verify_ok": ok, "verify_issue": issue}.
    """
    raise NotImplementedError("Implement in Phase 3")


def revise_node(state: AgentState) -> dict:
    """TODO (Phase 3, step 1).

    - Build messages using REVISE_SYSTEM / REVISE_USER, including
      state.verify_issue so the revise call has specific feedback.
    - Call llm().invoke(messages).
    - Strip fences.
    - Return {"sql": <new sql>, "iteration": state.iteration + 1, "history": ...}.
    """
    raise NotImplementedError("Implement in Phase 3")


def route_after_verify(state: AgentState) -> str:
    """TODO (Phase 3, step 3).

    Return:
      - "revise" if verify said not ok AND state.iteration < MAX_ITERATIONS
      - "end"    otherwise
    """
    raise NotImplementedError("Implement in Phase 3")


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
