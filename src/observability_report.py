# src/observability_report.py
"""Prints a summary table of recent pipeline runs from logs/pipeline_runs.jsonl."""
from pathlib import Path
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline_runs.jsonl"

def load_records():
    if not LOG_PATH.exists():
        return []
    records = []
    for line in LOG_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=20, help="show only the N most recent runs")
    args = ap.parse_args()

    records = load_records()
    if not records:
        print(f"No pipeline runs logged yet at {LOG_PATH}")
        return

    records = records[-args.last:]
    header = f"{'stage':<18} {'status':<8} {'duration_s':<11} {'started_at':<26} context"
    print(header)
    print("-" * len(header))
    for r in records:
        ctx = ", ".join(f"{k}={v}" for k, v in r.get("context", {}).items())
        print(f"{r['stage']:<18} {r['status']:<8} {r['duration_seconds']:<11} {r['started_at']:<26} {ctx}")

    n_success = sum(1 for r in records if r["status"] == "success")
    n_error = sum(1 for r in records if r["status"] == "error")
    print(f"\n{len(records)} runs shown: {n_success} success, {n_error} error")

if __name__ == "__main__":
    main()
