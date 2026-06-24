#!/usr/bin/env bash
# Creates (or updates) an EventBridge Scheduler schedule that invokes the
# scrape-trigger Lambda on a recurring cadence -- the automated alternative
# to clicking "Run New Scrape" in the dashboard every day.
#
# Usage:
#   ./scripts/schedule_new_arrivals_scrape.sh                          # daily at 6am UTC
#   ./scripts/schedule_new_arrivals_scrape.sh --expression "rate(1 hour)"
#   ./scripts/schedule_new_arrivals_scrape.sh --test                   # rate(1 minute), for verifying the wiring fires
set -euo pipefail

ENDPOINT="${AWS_ENDPOINT_URL:-http://localhost:4566}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
SCHEDULE_NAME="new-arrivals-daily-scrape"
FUNCTION_NAME="scrape-trigger"
SCHEDULE_EXPRESSION="cron(0 6 * * ? *)"   # default: daily at 06:00 UTC

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --expression) SCHEDULE_EXPRESSION="$2"; shift 2 ;;
    --test) SCHEDULE_EXPRESSION="rate(1 minute)"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

awscmd() { aws --endpoint-url="${ENDPOINT}" --region "${REGION}" "$@"; }

LAMBDA_ARN=$(awscmd lambda get-function --function-name "${FUNCTION_NAME}" --query 'Configuration.FunctionArn' --output text 2>/dev/null) \
  || { echo "[schedule] ${FUNCTION_NAME} Lambda not found -- run ./scripts/deploy_scrape_trigger_lambda.sh first"; exit 1; }

echo "[schedule] ensuring scheduler assume-role exists"
awscmd iam create-role \
  --role-name scheduler-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
  > /dev/null 2>&1 || echo "[schedule] role already exists"
ROLE_ARN="arn:aws:iam::000000000000:role/scheduler-role"

echo "[schedule] creating/updating schedule: ${SCHEDULE_NAME} (${SCHEDULE_EXPRESSION})"
if awscmd scheduler get-schedule --name "${SCHEDULE_NAME}" > /dev/null 2>&1; then
  awscmd scheduler update-schedule \
    --name "${SCHEDULE_NAME}" \
    --schedule-expression "${SCHEDULE_EXPRESSION}" \
    --flexible-time-window "Mode=OFF" \
    --target "Arn=${LAMBDA_ARN},RoleArn=${ROLE_ARN}" \
    > /dev/null
else
  awscmd scheduler create-schedule \
    --name "${SCHEDULE_NAME}" \
    --schedule-expression "${SCHEDULE_EXPRESSION}" \
    --flexible-time-window "Mode=OFF" \
    --target "Arn=${LAMBDA_ARN},RoleArn=${ROLE_ARN}" \
    > /dev/null
fi

echo "[schedule] done. Schedule '${SCHEDULE_NAME}' set to: ${SCHEDULE_EXPRESSION}"
echo "[schedule] check it fired: aws --endpoint-url=${ENDPOINT} s3api get-object --bucket fashion-trend-images --key scheduler/last_trigger.json /dev/stdout"
