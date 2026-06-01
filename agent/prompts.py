"""Prompt templates for the agent nodes.

These start as deliberate stubs. Filling them in (and calibrating them
against the eval set) is part of the assignment.

Notes:
- The verify prompt is the hardest. Too strict and the loop never
  terminates, too lenient and verify is decoration. Aim for it to fire
  on the obvious failure modes (SQL errored, zero rows when the question
  implies rows exist, columns that clearly don't answer the question).
- For verify, force structured JSON output via response_format so you
  can parse it deterministically.
- For revise, pass the specific issue from verify - generic "try again"
  prompts produce the same SQL.
"""

GENERATE_SQL_SYSTEM = """\
TODO: write the system prompt for SQL generation.
The model should produce SQL ONLY, no explanation, no markdown fences.
"""

GENERATE_SQL_USER = """\
TODO: format the user message. Variables available:
  {schema}   - rendered CREATE TABLE statements
  {question} - the analyst's question
"""


VERIFY_SYSTEM = """\
TODO: write the verify system prompt. Demand strict JSON output:
  {"ok": bool, "issue": "<short description, empty when ok>"}
"""

VERIFY_USER = """\
TODO: format the user message. Variables available:
  {question}
  {sql}
  {execution_result}  - ExecutionResult.render() output
"""


REVISE_SYSTEM = """\
TODO: write the revise system prompt.
The model should rewrite the SQL, addressing the specific issue from verify.
SQL only, no explanation, no fences.
"""

REVISE_USER = """\
TODO: format the user message. Variables available:
  {schema}
  {question}
  {prev_sql}
  {issue}  - issue string from verify
"""
