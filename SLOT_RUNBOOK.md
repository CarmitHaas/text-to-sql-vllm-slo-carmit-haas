# H100 slot runbook (11 PM, one hour)

Everything is built and tested locally. This hour is pure capture. Do not debug code here, only run it.
Delete this file before zipping the final submission if you want a clean repo.

## Pre-decided config (already in scripts/start_vllm.sh)
Baseline: FP8 checkpoint, max-model-len 8192, gpu-mem-util 0.90, max-num-seqs 32,
max-num-batched-tokens 8192, prefix caching on. The one Phase 6 change is picked by symptom (see step 7).

## 0. The first 10 minutes (do these in parallel)

Open 3 terminals on the VM. Forward 5 ports from your laptop first:
`ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 3001:localhost:3001 -L 8000:localhost:8000 -L 8001:localhost:8001 <user>@<vm>`

**Terminal A — start vLLM FIRST so the model downloads while you set up the rest:**
```bash
git clone https://github.com/CarmitHaas/text-to-sql-vllm-slo-carmit-haas.git mlops-assignment && cd mlops-assignment
# create .env (see step 1 below) BEFORE this if HF_TOKEN gating matters; FP8 Qwen is public
export HF_TOKEN=hf_...            # your token
bash scripts/start_vllm.sh        # downloads ~30GB FP8 weights, then loads. Leave it.
```

**Terminal B — while vLLM loads, stand up data + o11y:**
```bash
cd mlops-assignment
uv sync
uv run python scripts/load_data.py     # regenerates eval_set.jsonl + perf_pool.jsonl (seed 0, identical)
docker compose up -d                    # prometheus, grafana, langfuse stack
```

**Terminal C / browser — while the above run:**
- Open Langfuse http://localhost:3001, sign up (fresh VM = new account), create org/project,
  Settings -> API Keys -> create. You will paste these into .env next.

## 1. Write .env on the VM (points the agent at LOCAL vLLM, not Nebius)
```
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
OPENAI_API_KEY=not-needed
HF_TOKEN=hf_...
LANGFUSE_PUBLIC_KEY=pk-lf-...     # from the VM's Langfuse
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3001
```

## 2. vLLM ready -> manual queries  [screenshots/vllm_manual_query.png]
```bash
curl -s http://localhost:8000/v1/models | jq .
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen/Qwen3-30B-A3B-Instruct-2507",
  "messages":[{"role":"user","content":"Write one SQLite SELECT that returns the number 1."}]}' | jq -r '.choices[0].message.content'
```
Screenshot the terminal showing vLLM serving + a query returning SQL.

## 3. Start agent, fire a burst -> dashboard  [screenshots/grafana_serving.png]
```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001 &   # picks up .env
# small burst so panels move:
for i in $(seq 1 20); do curl -s -X POST localhost:8001/answer -H 'Content-Type: application/json' \
  -d '{"question":"List the coordinates of the Australian Grand Prix circuit.","db":"formula_1"}' >/dev/null & done; wait
```
Open Grafana http://localhost:3000 (admin/admin), dashboard "vLLM serving". Screenshot the full board reacting.

## 4. Langfuse traces  [screenshots/langfuse_trace.png + langfuse_tags.png]
```bash
# 10 tagged questions, including one that triggers a revise:
uv run python - <<'PY'
import json, httpx
qs=[json.loads(l) for l in open("evals/eval_set.jsonl")][:9]
qs.append({"question":"Mention the reputation of users who had obtained the badge on 7/19/2010 7:39:08 PM.","db_id":"codebase_community"})
for i,q in enumerate(qs,1):
    httpx.post("http://localhost:8001/answer", json={"question":q["question"],"db":q["db_id"],
        "tags":{"run":"h100","qid":str(i)}}, timeout=120)
print("fired 10")
PY
```
In Langfuse: open the codebase_community trace -> screenshot the generate_sql / verify / revise waterfall.
Then the trace LIST showing the `run=h100` / `qid` tags -> second screenshot.

## 5. Baseline eval  [results/eval_baseline.json + screenshots/grafana_eval_run.png]
```bash
uv run python evals/run_eval.py --out results/eval_baseline.json   # ~60 calls; screenshot Grafana mid-run
```

## 6. Load test at baseline  [screenshots/grafana_before.png]
First restart the agent with tracing OFF, so the reported SLO latency is not taxed by Langfuse span
export and the Langfuse stack does not contend with vLLM on the same VM. Your Phase 4 traces are already
captured in step 4, so you lose nothing.
```bash
# stop the step-3 uvicorn (the one started with keys), then relaunch untraced:
LANGFUSE_PUBLIC_KEY= LANGFUSE_SECRET_KEY= uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001 &
```
Then run the load test and screenshot Grafana during the 5 minutes:
```bash
uv run python load_test/driver.py --rps 10 --duration 300
```
Note the P50/P95/P99 it prints (these are your reported SLO numbers, untainted by tracing).
The agent stays untraced through step 7 too.

## 7. ONE change, restart vLLM, load test again  [screenshots/grafana_after.png]
Pick the change from what the dashboard showed in step 6:
- KV usage near 1.0 / preemptions  -> `KV_CACHE_DTYPE=fp8 bash scripts/start_vllm.sh`
- waiting queue grows, KV has room   -> `MAX_NUM_SEQS=64 bash scripts/start_vllm.sh`
- TTFT high                          -> `MAX_NUM_BATCHED_TOKENS=16384 bash scripts/start_vllm.sh`
```bash
# Ctrl-C the old vLLM, relaunch with the chosen env var, then:
uv run python load_test/driver.py --rps 10 --duration 300         # screenshot Grafana "after"
```

## 8. Re-eval at final config  [results/eval_after_tuning.json]
```bash
uv run python evals/run_eval.py --out results/eval_after_tuning.json
```

## 9. Capture numbers (fill REPORT.md after the slot)
- From the two load-test JSONs: achieved RPS, P50/P95/P99.
- From the two eval JSONs: overall_pass_rate, pass_rate_by_iteration.
- Write the Phase 6 line: "saw X -> hypothesized Y -> changed Z -> result W".

## Slot savers
- If model load + CUDA-graph capture eats the hour, add `--enforce-eager` (edit start_vllm.sh exec line).
- Keep load-test --duration 300 only for the two reported runs.
- Skip nothing in steps 2-8; that order captures all 8 screenshots + both JSONs.
- After firing the 10 questions in step 4, wait ~6s before screenshotting Langfuse so the last traces flush.

## Submit (after the slot)
- Fill REPORT.md numbers, then commit and push the deliverables (the .gitignore now tracks these files):
  `git add REPORT.md results/eval_baseline.json results/eval_after_tuning.json screenshots/*.png`
  `git commit -m "H100 run: results, screenshots, report" && git push`
- Your public repo IS the submission: https://github.com/CarmitHaas/text-to-sql-vllm-slo-carmit-haas
- If a zip is required, build a clean one from tracked files only (avoids shipping the 3.5G data/ dir):
  `git archive --format=zip -o submission.zip HEAD`
- Optional: delete SLOT_RUNBOOK.md before the final commit if you want a clean deliverable set.
