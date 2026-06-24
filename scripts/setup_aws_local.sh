#!/usr/bin/env bash
# Creates the S3 bucket and SQS queue against LocalStack.
# Run after `docker compose up -d localstack`.
set -euo pipefail

ENDPOINT="${AWS_ENDPOINT_URL:-http://localhost:4566}"
BUCKET="${S3_BUCKET:-fashion-trend-images}"
QUEUE="${SQS_QUEUE_NAME:-fashion-trend-ingestion}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"

echo "[setup] waiting for LocalStack at ${ENDPOINT}..."
until curl -s "${ENDPOINT}/_localstack/health" > /dev/null; do
  sleep 1
done

echo "[setup] creating S3 bucket: ${BUCKET}"
aws --endpoint-url="${ENDPOINT}" --region "${REGION}" s3 mb "s3://${BUCKET}" 2>/dev/null \
  || echo "[setup] bucket ${BUCKET} already exists"

echo "[setup] creating SQS queue: ${QUEUE}"
aws --endpoint-url="${ENDPOINT}" --region "${REGION}" sqs create-queue --queue-name "${QUEUE}" \
  || echo "[setup] queue ${QUEUE} already exists"

echo "[setup] done."
