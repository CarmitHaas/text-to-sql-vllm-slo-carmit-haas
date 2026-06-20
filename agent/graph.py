"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 3

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


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
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _parse_verdict(text: str) -> tuple[bool, str]:
    """Parse the verifier reply into (ok, issue), defensively.

    The verifier is asked for a single-line JSON object {"ok": bool, "issue": str},
    but a model may wrap it in prose or a ```json fence. We strip a fence if present,
    pull the first {...} block, and json.loads it. If parsing fails we default to
    NOT ok (with a note) so the loop gets a chance to revise rather than passing an
    answer we could not actually validate.
    """
    blob = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", blob, re.DOTALL | re.IGNORECASE)
    if fenced:
        blob = fenced.group(1).strip()
    match = re.search(r"\{.*\}", blob, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return bool(obj.get("ok", False)), str(obj.get("issue", "") or "")
        except json.JSONDecodeError:
            pass
    return False, "verifier reply was not parseable JSON"


def _result_text(state: AgentState) -> str:
    """Compact view of the last execution for verify/revise prompts."""
    return state.execution.render() if state.execution else "ERROR: no execution result"


def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    response = llm().invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Builds the VERIFY_* prompt (str.replace so brace characters in result rows do
    not break templating), calls the shared llm(), and parses a {"ok", "issue"}
    JSON verdict defensively. Returns the two verify_* fields the router reads.
    """
    user = (
        prompts.VERIFY_USER
        .replace("{question}", state.question)
        .replace("{sql}", state.sql)
        .replace("{result}", _result_text(state))
    )
    response = llm().invoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", user),
    ])
    ok, issue = _parse_verdict(response.content)
    return {"verify_ok": ok, "verify_issue": issue}


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given the verifier's complaint and prior attempt.

    Same shape as generate_sql_node: it bumps `iteration` and appends to `history`
    with the same {"node", "sql"} record so Phase 5 can reconstruct per-iteration
    accuracy from the returned history.
    """
    user = (
        prompts.REVISE_USER
        .replace("{schema}", state.schema)
        .replace("{question}", state.question)
        .replace("{sql}", state.sql)
        .replace("{result}", _result_text(state))
        .replace("{issue}", state.verify_issue or "the previous result did not answer the question")
    )
    response = llm().invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", user),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: "revise" to loop, "end" to terminate.

    End when the verifier was happy, or when we have hit the iteration cap.
    Otherwise revise. `iteration` was bumped in generate/revise, so after the
    first attempt it is 1, after the first revise it is 2, and so on.
    """
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


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
