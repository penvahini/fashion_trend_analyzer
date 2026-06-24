# src/observability.py
"""
Lightweight run-tracking for pipeline stages. Wrap a stage's main() body in
`with track_run("stage_name", **context):` to append a timing/status record
to logs/pipeline_runs.jsonl -- gives you a real audit trail of every
scrape/segment/cluster/label/trend run without standing up a metrics stack.
"""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import json
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline_runs.jsonl"

@contextmanager
def track_run(stage: str, **context):
    start = time.time()
    record = {
        "stage": stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "context": context,
    }
    try:
        yield record
        record["status"] = "success"
    except Exception as e:
        record["status"] = "error"
        record["error"] = str(e)
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        record["duration_seconds"] = round(time.time() - start, 3)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
