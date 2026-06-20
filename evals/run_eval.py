"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question by execution accuracy, per iteration.

    Calls the agent over HTTP, then re-runs each SQL the agent emitted (the
    generate_sql attempt plus any revise attempts, taken from the agent's
    `history`) against the target DB and compares each to the gold result set.
    That gives a per-iteration correctness list, which `summarize` turns into a
    per-iteration pass rate.
    """
    db_id = question["db_id"]
    gold_ok, gold_rows, gold_err = run_sql(db_id, question["gold_sql"])

    record: dict = {
        "db_id": db_id,
        "question": question["question"],
        "gold_ok": gold_ok,
        "gold_error": gold_err,
        "agent_ok": False,
        "agent_error": None,
        "n_iterations": 0,
        "per_iter": [],          # correctness of each generate/revise attempt, in order
        "final_correct": False,
        "latency_seconds": None,
    }

    t0 = time.monotonic()
    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id, "tags": {"run": "eval"}},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        record["agent_error"] = f"{type(e).__name__}: {e}"
        record["latency_seconds"] = time.monotonic() - t0
        return record
    record["latency_seconds"] = time.monotonic() - t0

    record["agent_ok"] = bool(data.get("ok", False))
    record["agent_error"] = data.get("error")

    # Each generate/revise node logged {"node", "sql"}; score those SQLs in order.
    attempts = [h["sql"] for h in data.get("history", []) if "sql" in h]
    if not attempts and data.get("sql"):
        attempts = [data["sql"]]
    record["n_iterations"] = len(attempts)

    if gold_ok:
        per_iter = []
        for sql in attempts:
            ok, rows, _ = run_sql(db_id, sql)
            per_iter.append(bool(ok) and matches(gold_rows, rows))
        record["per_iter"] = per_iter
        record["final_correct"] = per_iter[-1] if per_iter else False
    # If gold itself did not run, the question is not gradable; summarize counts
    # those separately rather than scoring them as wrong.
    return record


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    gradable = [r for r in results if r.get("gold_ok")]
    n = len(gradable)

    overall = (sum(1 for r in gradable if r["final_correct"]) / n) if n else 0.0

    # Per-iteration pass rate, carrying each question's last attempt forward to
    # every later iteration index.
    max_iters = max((len(r["per_iter"]) for r in gradable), default=0)
    pass_rate_by_iteration: list[float] = []
    for k in range(max_iters):
        hits = 0
        for r in gradable:
            pi = r["per_iter"]
            if pi and pi[min(k, len(pi) - 1)]:
                hits += 1
        pass_rate_by_iteration.append(round(hits / n, 4) if n else 0.0)

    iters_dist: dict[str, int] = {}
    for r in gradable:
        key = str(len(r["per_iter"]))
        iters_dist[key] = iters_dist.get(key, 0) + 1

    return {
        "n_questions": len(results),
        "n_gradable": n,
        "n_gold_errors": len(results) - n,
        "n_agent_errors": sum(1 for r in gradable if not r["agent_ok"]),
        "overall_pass_rate": round(overall, 4),
        "pass_rate_by_iteration": pass_rate_by_iteration,  # index 0 = after first generate
        "iterations_distribution": iters_dist,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
