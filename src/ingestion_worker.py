# src/ingestion_worker.py
"""
Queue-based ingestion worker: polls the SQS queue for scrape jobs, downloads
each job's image from S3 into images/original_images/<run_id>/, appends it to
that run's manifest.json, and deletes the message on success.

This stands in for an EC2 worker fleet processing images at scale -- the same
code runs locally against LocalStack or as a container against real AWS by
just changing AWS_ENDPOINT_URL / credentials (see src/aws_clients.py).
"""
from pathlib import Path
import json
import time
import argparse

from aws_clients import get_s3_client, get_sqs_client, get_queue_url, S3_BUCKET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_IMAGES_DIR = PROJECT_ROOT / "images" / "original_images"

def load_manifest(run_dir: Path):
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return []

def append_manifest(run_dir: Path, entry: dict):
    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(run_dir)
    manifest.append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2))

def process_message(body: dict, s3_client):
    run_id = body["run_id"]
    filename = body["filename"]
    s3_key = body["s3_key"]

    run_dir = ORIGINAL_IMAGES_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dest_path = run_dir / filename

    s3_client.download_file(S3_BUCKET, s3_key, str(dest_path))
    append_manifest(run_dir, {
        "filename": filename,
        "source_url": body.get("source_url", ""),
        "run_id": run_id,
        "scraped_at": body.get("scraped_at", ""),
    })
    print(f"[worker] processed {s3_key} -> {dest_path}")

def poll_loop(max_messages=10, wait_seconds=5, idle_exit_after=None):
    """
    Polls SQS until no messages are received for idle_exit_after seconds
    (or runs forever if idle_exit_after is None).
    """
    sqs_client = get_sqs_client()
    s3_client = get_s3_client()
    queue_url = get_queue_url(sqs_client)

    last_message_at = time.time()
    while True:
        resp = sqs_client.receive_message(
            QueueUrl=queue_url, MaxNumberOfMessages=max_messages, WaitTimeSeconds=wait_seconds,
        )
        messages = resp.get("Messages", [])
        if not messages:
            if idle_exit_after is not None and (time.time() - last_message_at) >= idle_exit_after:
                print("[worker] idle timeout reached, exiting")
                return
            continue

        last_message_at = time.time()
        for msg in messages:
            try:
                body = json.loads(msg["Body"])
                process_message(body, s3_client)
                sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])
            except Exception as e:
                print(f"[worker] failed to process message: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-messages", type=int, default=10)
    ap.add_argument("--wait-seconds", type=int, default=5, help="SQS long-poll wait time")
    ap.add_argument("--idle-exit-after", type=int, default=None,
                     help="Exit after this many seconds with no messages (default: run forever)")
    args = ap.parse_args()
    poll_loop(max_messages=args.max_messages, wait_seconds=args.wait_seconds, idle_exit_after=args.idle_exit_after)

if __name__ == "__main__":
    main()
