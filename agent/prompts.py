"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = ""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = ""


VERIFY_SYSTEM = ""

VERIFY_USER = ""


REVISE_SYSTEM = ""

REVISE_USER = ""
