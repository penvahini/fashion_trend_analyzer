# src/run_pipeline.py
"""
Orchestrates one full pipeline run end-to-end: scrape -> segment -> cluster
-> label -> trend_tracker -> (optional) sync to S3 for the API.

This is what actually answers "how do we get a second data point for trends
over time" -- before this, every stage had to be run manually by hand with a
matching --run-id. Each invocation defaults to a fresh timestamp run_id, so
running it twice (e.g. from the dashboard's "Run New Scrape" button) always
produces a new lineage point for trend_tracker.py to diff against.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --run-id 2025-08-18 --k 7 --max-images 40

Or import `run_pipeline(...)` as a generator that yields log lines, so a
caller (like the Streamlit dashboard) can stream progress live.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import os
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

def run_step(label, args, cwd=PROJECT_ROOT, env=None, required=True):
    """Yields log lines, then a final '[run_pipeline:STEP_OK]'/'[run_pipeline:STEP_FAILED]'
    sentinel line so callers (e.g. the dashboard) can tell a non-required step's
    real outcome without scraping log text for "exited with code"."""
    yield f"\n=== {label} ===  ({' '.join(args)})"
    proc = subprocess.Popen(
        args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        yield line.rstrip()
    proc.wait()
    if proc.returncode != 0:
        msg = f"[run_pipeline] '{label}' exited with code {proc.returncode}"
        yield msg
        yield f"[run_pipeline:STEP_FAILED] {label}"
        if required:
            raise RuntimeError(msg)
    else:
        yield f"[run_pipeline] '{label}' done."
        yield f"[run_pipeline:STEP_OK] {label}"

def run_pipeline(run_id: str, k: int = 7, max_images: int = 40,
                  use_original: bool = True, model: str = "gpt-4o-mini",
                  sync_to_api: bool = True):
    """Generator yielding log lines as each stage runs."""
    env = os.environ.copy()

    yield f"[run_pipeline] starting run_id={run_id}"

    yield from run_step(
        "1/5 scrape", [PYTHON, "src/webscraper.py", "--run-id", run_id, "--max-images", str(max_images)],
        env=env,
    )
    yield from run_step(
        "2/5 segment", [PYTHON, "src/segment.py", "--run-id", run_id], env=env,
    )
    yield from run_step(
        "3/5 cluster", [PYTHON, "src/cluster_pipeline.py", "--run-id", run_id, "--k", str(k)], env=env,
    )

    label_args = [PYTHON, "src/label_clusters.py", "--run-id", run_id, "--model", model]
    if use_original:
        label_args.append("--use-original")
    yield from run_step("4/5 label", label_args, env=env)

    yield from run_step("5/5 trend_tracker", [PYTHON, "src/trend_tracker.py"], env=env)

    if sync_to_api:
        yield from run_step(
            "sync to S3 (for the API)", [PYTHON, "src/sync_artifacts_to_s3.py"], env=env, required=False,
        )

    yield f"[run_pipeline] run {run_id} complete."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--no-original", action="store_true")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--no-sync", action="store_true")
    args = ap.parse_args()

    for line in run_pipeline(
        args.run_id, k=args.k, max_images=args.max_images,
        use_original=not args.no_original, model=args.model, sync_to_api=not args.no_sync,
    ):
        print(line)

if __name__ == "__main__":
    main()
