# src/sync_artifacts_to_s3.py
"""
Uploads the artifacts the API Lambda needs to S3, so the Lambda doesn't need
filesystem access to data/runs/. Embeddings are exported to plain JSON
(rather than .npy) so the Lambda can read them with boto3 alone, no numpy.
"""
from pathlib import Path
import json
import argparse

from aws_clients import get_s3_client, S3_BUCKET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
TREND_REPORT_PATH = PROJECT_ROOT / "data" / "trend_report.json"

def export_embeddings_json(run_dir: Path) -> Path:
    import numpy as np  # only needed locally, never inside Lambda
    Z = np.load(run_dir / "clip_embeddings.npy")
    filenames = json.loads((run_dir / "clip_filenames.json").read_text())
    out_path = run_dir / "clip_embeddings_export.json"
    out_path.write_text(json.dumps({
        "filenames": filenames,
        "embeddings": Z.tolist(),
    }))
    return out_path

def sync_run(run_id: str, s3_client):
    run_dir = RUNS_DIR / run_id
    embeddings_export = export_embeddings_json(run_dir)

    files_to_upload = ["clusters.csv", "cluster_labels.json", embeddings_export.name]
    for fname in files_to_upload:
        local_path = run_dir / fname
        if not local_path.exists():
            print(f"[sync] skip missing {local_path}")
            continue
        key = f"artifacts/{run_id}/{fname}"
        s3_client.upload_file(str(local_path), S3_BUCKET, key)
        print(f"[sync] uploaded {key}")

def sync_trend_report(s3_client):
    if not TREND_REPORT_PATH.exists():
        print(f"[sync] skip missing {TREND_REPORT_PATH}")
        return
    key = "artifacts/trend_report.json"
    s3_client.upload_file(str(TREND_REPORT_PATH), S3_BUCKET, key)
    print(f"[sync] uploaded {key}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, help="Specific run to sync (default: all runs under data/runs/)")
    args = ap.parse_args()

    s3_client = get_s3_client()
    run_ids = [args.run_id] if args.run_id else sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())

    for run_id in run_ids:
        sync_run(run_id, s3_client)
    sync_trend_report(s3_client)

if __name__ == "__main__":
    main()
