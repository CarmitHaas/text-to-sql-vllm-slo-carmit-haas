# Instructor notes

Course-staff-only notes for running this assignment. Not for student-facing material.

## Before you ship it to students

Do a full dry-run on the same VM image students will use. The pieces most likely to break in unfamiliar hands are:

1. **BIRD download URL (`scripts/load_data.py`).** The script defaults to `BIRD_DEV_URL=https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip`. BIRD's hosting has moved a couple of times. Confirm the URL still resolves and the archive layout still has `dev.json` + `dev_databases/<db_id>/<db_id>.sqlite`. If it has moved, override with `BIRD_DEV_URL=<new_url>` in env, or rewrite the extraction logic. If you can't find a reliable mirror, fall back to Spider (`xlangai/spider` on HF) - swap dataset everywhere it appears in code; the assignment structure is unchanged.

2. **Langfuse self-hosted (`docker-compose.yml`).** The compose pins `langfuse/langfuse:3` and `langfuse/langfuse-worker:3`. Langfuse's v3 self-hosting has had churn — check the current pinned `:3` tag actually boots, and that the env-var contract hasn't drifted. The minio-init service is what creates the `langfuse` bucket; if it races, students will see trace ingestion failures.

3. **vLLM metric names (`infra/grafana/provisioning/dashboards/serving.json`).** The starter dashboard uses `vllm:num_requests_running` and `vllm:generation_tokens_total`. Confirm these names still exist in the vLLM version resolved by `uv sync`. Newer vLLM has renamed some metrics; verify against the actual `/metrics` output of the running server.

4. **Qwen quantization variants.** The README implies BF16, FP8, AWQ-Int4, GPTQ-Int4 are all available. The official FP8 (`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`) should exist; community AWQ/GPTQ mirrors come and go. Before the assignment goes out, sanity-check at least 2 quant variants actually load and serve on the course VM.

5. **GPU memory.** On H100 80GB, BF16 weights are ~61 GB. After vLLM overhead this leaves ~12-18 GB for KV cache. Students who pick BF16 + an aggressive `max-model-len` will hit OOM — that's by design (the assignment expects them to learn this) but make sure the VM image doesn't have something else stealing memory before they start.

## VM sizing

- 1× H100 80GB
- ~150 GB disk free (model weights + HF cache + BIRD ~500 MB + Docker volumes)
- 32 GB RAM minimum
- Docker + docker-compose pre-installed
- `uv` not assumed to be pre-installed; setup tells students to curl-install it

## Walk-through timing (full dry-run)

| Phase | Realistic time |
|---|---|
| 0 (setup) | 30-45 min, mostly waiting on docker image pulls + HF download |
| 1 (vLLM + config) | 60-90 min, depending on how many quant/config combinations they explore |
| 2 (dashboard) | 45-60 min |
| 3 (agent) | 90-120 min |
| 4 (Langfuse) | 20-30 min |
| 5 (eval) | 45-60 min |
| 6 (SLO hit cycle) | 90-120 min — this is where they actually learn |
| 7 (writeup) | 30 min |

Total realistic range: 7-9 hours for an engaged student.

## Common student pitfalls (watch for these in TA Slack)

- **vLLM won't start, "KV cache too small".** Their `max-num-seqs × max-model-len` is bigger than the remaining VRAM after weights. They need to either pick a smaller quant, lower `max-model-len`, or lower `max-num-seqs`. Resist the urge to give them the formula — making the connection is the lesson.
- **Grafana panel shows flat line / NaN on a histogram metric.** Almost always `histogram_quantile()` applied directly to a `_bucket` counter instead of to `rate(_bucket[range])`. Point at the Prometheus docs, not the answer.
- **Langfuse traces don't appear.** Worker container is racing the bucket creation, or the LangChain callback isn't actually attached. Check `langfuse-worker` logs and the agent server's startup log.
- **Verify node loops forever / never fires.** Prompt calibration. Have them run 5 questions and look at the verify outputs; the diagnosis is usually obvious.
- **Eval pass rate ~0.** Almost always a canonicalization bug (column-name case, NULL handling, row order). The provided `canonicalize()` helper handles the common case but they may have bypassed it.
- **Load test driver shows P95 << observed in browser.** They're hitting an endpoint that returns before the model finishes (e.g., a stub) - check the agent server is actually wired to the graph.

## Grading approach

The assignment is structured so the "MLOps part" (Phase 6) is the biggest single bucket. A student who does Phases 1-5 cleanly but never genuinely engages with Phase 6 should land in the 50-60 range; one who does Phase 6 honestly even with rough Phase 1-5 work should land in the 70-80 range. What to look for in their `REPORT.md`:

- **Was the config choice defended with numbers?** "I picked FP8 because it has more KV cache budget" without showing throughput/quality numbers is a soft answer.
- **Did Grafana actually lead the tuning iterations?** Each iteration should reference a specific observation. "I bumped max-num-seqs and it got faster" without a prior observation that scheduling was the bottleneck is the most common failure mode.
- **Did they re-run evals after tuning?** Quality regressions sneak in via aggressive batching choices. Students who tuned without re-checking lose points even if their dashboard looks great.
- **Honest writeup beats polished writeup.** A student who says "I couldn't break the SLO, here's what I tried and where I think the ceiling is" is showing more diagnosis than one who claims clean wins everywhere.

The artifact checklist in the README's Final deliverables table is the structural minimum. Use it to triage, then dig into the report.

## Files students see vs. you maintain

| Path | Role |
|---|---|
| `README.md` | Student-facing assignment |
| `pyproject.toml`, `uv.lock` | uv manages |
| `docker-compose.yml` | Stays as-is unless you change Langfuse version |
| `infra/prometheus.yml`, `infra/grafana/provisioning/**` | Adjust if vLLM metric names change |
| `agent/execution.py`, `agent/schema.py`, `agent/server.py` | Provided complete |
| `agent/graph.py`, `agent/prompts.py` | Students implement |
| `evals/run_eval.py` | Students implement (helpers provided) |
| `load_test/driver.py` | Provided complete |
| `scripts/load_data.py` | Provided complete; URL may need updating |
| `scripts/start_vllm.sh` | Stub - students fill in flags |
| `data/`, `results/`, `screenshots/` | Generated/populated during the assignment |

## Don't tell the students

A few things that would defeat the lesson if surfaced too early. If you're TA'ing, ask back rather than answer:

- Which specific vLLM flags help the SLO. The answer differs per quant choice and is the point.
- Which Prometheus query shape to use for histogram percentiles. The struggle is informative.
- Whether the verify node should fire on a given question. They'll calibrate via the eval set.
- What BIRD's known gold-SQL errors are. Catching one is a bonus point in the rubric.
