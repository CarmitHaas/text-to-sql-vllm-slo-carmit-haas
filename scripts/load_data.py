#!/usr/bin/env python3
"""Download and prepare the BIRD subset for this assignment.

Produces:
- data/bird/<db_id>.sqlite          - sqlite DBs to query
- data/bird/dev_databases/...       - raw extracted contents
- evals/eval_set.jsonl              - 30 curated questions
- load_test/perf_pool.jsonl         - ~1500 questions for the load test

Instructor note: BIRD's hosted ZIP URL has moved a few times. If the download
in BIRD_DEV_URL stops working, point it at a working mirror, or have students
download dev.zip manually from https://bird-bench.github.io and unzip into
data/bird/ before running this script.
"""

import json
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "bird"
EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
PERF_FILE = ROOT / "load_test" / "perf_pool.jsonl"

BIRD_DEV_URL = os.environ.get(
    "BIRD_DEV_URL",
    "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
)

# DBs we scope the assignment to. Keeps the corpus tight enough that BM25 /
# schema-rendering aren't a bottleneck for the student.
SCOPED_DBS = ["california_schools", "european_football_2", "financial"]

N_EVAL = 30
N_PERF = 1500


def download_and_extract() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "dev.zip"
    if not zip_path.exists():
        print(f"Downloading {BIRD_DEV_URL} ...")
        urllib.request.urlretrieve(BIRD_DEV_URL, zip_path)
    if not any(DATA_DIR.rglob("dev.json")):
        print(f"Extracting to {DATA_DIR} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
    else:
        print("Already extracted.")


def build_eval_files() -> None:
    dev_json_path = next(DATA_DIR.rglob("dev.json"), None)
    if dev_json_path is None:
        sys.exit("Could not find dev.json after extraction - check the archive layout.")

    rows = json.loads(dev_json_path.read_text())
    scoped = [r for r in rows if r.get("db_id") in SCOPED_DBS]
    print(f"Scoped to {len(scoped)} questions across {len(SCOPED_DBS)} DBs.")

    rnd = random.Random(0)  # stable shuffle for reproducibility
    rnd.shuffle(scoped)

    eval_rows = scoped[:N_EVAL]
    perf_source = scoped[N_EVAL:]

    EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_FILE.open("w") as f:
        for r in eval_rows:
            f.write(json.dumps({
                "question": r["question"],
                "db_id": r["db_id"],
                "gold_sql": r["SQL"],
            }) + "\n")
    print(f"Wrote {len(eval_rows)} eval questions to {EVAL_FILE}")

    PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
    if len(perf_source) >= N_PERF:
        perf_rows = perf_source[:N_PERF]
    else:
        # not enough unique rows; cycle through reshuffles until we have N_PERF
        perf_rows: list[dict] = []
        while len(perf_rows) < N_PERF:
            rnd.shuffle(perf_source)
            perf_rows.extend(perf_source)
        perf_rows = perf_rows[:N_PERF]
    with PERF_FILE.open("w") as f:
        for r in perf_rows:
            f.write(json.dumps({
                "question": r["question"],
                "db_id": r["db_id"],
            }) + "\n")
    print(f"Wrote {len(perf_rows)} perf questions to {PERF_FILE}")


def consolidate_sqlite() -> None:
    """Surface each db at data/bird/<db_id>.sqlite for easy loading."""
    for db in SCOPED_DBS:
        found = next(DATA_DIR.rglob(f"{db}.sqlite"), None)
        if found is None:
            print(f"WARNING: sqlite for {db} not found under {DATA_DIR}", file=sys.stderr)
            continue
        dest = DATA_DIR / f"{db}.sqlite"
        if not dest.exists() or dest.resolve() != found.resolve():
            shutil.copy(found, dest)
    available = sorted(p.name for p in DATA_DIR.glob("*.sqlite"))
    print(f"Sqlite DBs available: {available}")


if __name__ == "__main__":
    download_and_extract()
    build_eval_files()
    consolidate_sqlite()
    print("Done.")
