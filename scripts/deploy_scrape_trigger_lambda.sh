#!/usr/bin/env bash
# Deploys src/lambdas/scrape_trigger_handler.py -- the target invoked by the
# scheduled "new arrivals" scrape (see schedule_new_arrivals_scrape.sh).
set -euo pipefail

ENDPOINT="${AWS_ENDPOINT_URL:-http://localhost:4566}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
FUNCTION_NAME="scrape-trigger"
ROLE_ARN="arn:aws:iam::000000000000:role/lambda-role"

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d)"
ZIP_PATH="${BUILD_DIR}/scrape_trigger_handler.zip"

awscmd() { aws --endpoint-url="${ENDPOINT}" --region "${REGION}" "$@"; }

echo "[deploy] packaging Lambda"
cp "${ROOT_DIR}/src/lambdas/scrape_trigger_handler.py" "${BUILD_DIR}/"
(cd "${BUILD_DIR}" && zip -q scrape_trigger_handler.zip scrape_trigger_handler.py)

echo "[deploy] ensuring IAM role exists"
awscmd iam create-role \
  --role-name lambda-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
  > /dev/null 2>&1 || echo "[deploy] role already exists"

echo "[deploy] creating/updating Lambda function: ${FUNCTION_NAME}"
if awscmd lambda get-function --function-name "${FUNCTION_NAME}" > /dev/null 2>&1; then
  awscmd lambda update-function-code --function-name "${FUNCTION_NAME}" --zip-file "fileb://${ZIP_PATH}" > /dev/null
else
  awscmd lambda create-function \
    --function-name "${FUNCTION_NAME}" \
    --runtime python3.11 \
    --handler scrape_trigger_handler.lambda_handler \
    --role "${ROLE_ARN}" \
    --zip-file "fileb://${ZIP_PATH}" \
    --environment "Variables={AWS_ENDPOINT_URL=http://localstack:4566,S3_BUCKET=fashion-trend-images,AWS_DEFAULT_REGION=${REGION}}" \
    --timeout 30 \
    > /dev/null
fi

awscmd lambda wait function-active --function-name "${FUNCTION_NAME}"
LAMBDA_ARN=$(awscmd lambda get-function --function-name "${FUNCTION_NAME}" --query 'Configuration.FunctionArn' --output text)
echo "[deploy] done. Lambda ARN: ${LAMBDA_ARN}"
