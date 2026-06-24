# src/lambdas/scrape_trigger_handler.py
"""
Target Lambda for the scheduled "new arrivals" scrape trigger (EventBridge
Scheduler invokes this on a cron, e.g. daily at 6am UTC).

The actual scrape/segment/cluster/label pipeline (Playwright + YOLO + SAM +
CLIP) is far too heavy to run inside Lambda (large binaries, multi-minute
runtime, needs a real CPU). So in a real AWS deployment this handler's job
is just to kick off the *real* compute -- calling ecs:RunTask against a
Fargate task definition built from deployment/Dockerfile.pipeline (the same
container image used by run_pipeline.py). That call is included below,
gated behind an env var so it can be turned on once a task definition is
actually registered; until then it writes a heartbeat object to S3 so the
schedule's wiring (EventBridge -> Lambda) can be verified end-to-end on
LocalStack without needing a real ECS cluster.
"""
import json
import os
from datetime import datetime, timezone
import boto3

S3_BUCKET = os.environ.get("S3_BUCKET", "fashion-trend-images")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

ECS_CLUSTER = os.environ.get("ECS_CLUSTER")
ECS_TASK_DEFINITION = os.environ.get("ECS_TASK_DEFINITION")
ECS_SUBNETS = os.environ.get("ECS_SUBNETS", "")
ECS_SECURITY_GROUPS = os.environ.get("ECS_SECURITY_GROUPS", "")

def lambda_handler(event, context):
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    s3 = boto3.client("s3", endpoint_url=AWS_ENDPOINT_URL, region_name=AWS_REGION)

    if ECS_CLUSTER and ECS_TASK_DEFINITION:
        ecs = boto3.client("ecs", endpoint_url=AWS_ENDPOINT_URL, region_name=AWS_REGION)
        resp = ecs.run_task(
            cluster=ECS_CLUSTER,
            taskDefinition=ECS_TASK_DEFINITION,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": ECS_SUBNETS.split(","),
                    "securityGroups": ECS_SECURITY_GROUPS.split(","),
                    "assignPublicIp": "ENABLED",
                }
            },
            overrides={
                "containerOverrides": [{
                    "name": "pipeline",
                    "command": ["python3", "src/run_pipeline.py", "--run-id", run_id],
                }]
            },
        )
        result = {"mode": "ecs_run_task", "run_id": run_id, "task_arns": [t["taskArn"] for t in resp.get("tasks", [])]}
    else:
        result = {
            "mode": "heartbeat_only",
            "run_id": run_id,
            "note": "ECS_CLUSTER/ECS_TASK_DEFINITION not set -- wrote a heartbeat instead of "
                    "launching the real pipeline. Set those env vars once a Fargate task "
                    "definition for deployment/Dockerfile.pipeline is registered.",
        }

    s3.put_object(
        Bucket=S3_BUCKET,
        Key="scheduler/last_trigger.json",
        Body=json.dumps({**result, "triggered_at": datetime.now(timezone.utc).isoformat()}).encode("utf-8"),
    )
    return {"statusCode": 200, "body": json.dumps(result)}
