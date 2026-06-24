#!/usr/bin/env bash
# Deploys src/lambdas/api_handler.py as a Lambda function and wires it up
# behind an API Gateway REST API, all against LocalStack.
#
# Uses REST API (apigateway) rather than HTTP API (apigatewayv2) since
# apigatewayv2 is a LocalStack Pro-only feature; REST API + Lambda proxy
# integration is available in the community edition and is the more
# "classic" API Gateway + Lambda pattern anyway.
set -euo pipefail

ENDPOINT="${AWS_ENDPOINT_URL:-http://localhost:4566}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
FUNCTION_NAME="fashion-trend-api"
ROLE_ARN="arn:aws:iam::000000000000:role/lambda-role"
API_NAME="fashion-trend-api"
STAGE_NAME="local"

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d)"
ZIP_PATH="${BUILD_DIR}/api_handler.zip"

awscmd() { aws --endpoint-url="${ENDPOINT}" --region "${REGION}" "$@"; }

echo "[deploy] packaging Lambda"
cp "${ROOT_DIR}/src/lambdas/api_handler.py" "${BUILD_DIR}/"
(cd "${BUILD_DIR}" && zip -q api_handler.zip api_handler.py)

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
    --handler api_handler.lambda_handler \
    --role "${ROLE_ARN}" \
    --zip-file "fileb://${ZIP_PATH}" \
    --environment "Variables={AWS_ENDPOINT_URL=http://localstack:4566,S3_BUCKET=fashion-trend-images,AWS_DEFAULT_REGION=${REGION}}" \
    --timeout 30 \
    > /dev/null
fi

echo "[deploy] waiting for function to become active"
awscmd lambda wait function-active --function-name "${FUNCTION_NAME}"

LAMBDA_ARN=$(awscmd lambda get-function --function-name "${FUNCTION_NAME}" --query 'Configuration.FunctionArn' --output text)
echo "[deploy] Lambda ARN: ${LAMBDA_ARN}"
LAMBDA_URI="arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations"

echo "[deploy] creating REST API"
API_ID=$(awscmd apigateway get-rest-apis --query "items[?name=='${API_NAME}'].id" --output text)
if [[ -z "${API_ID}" || "${API_ID}" == "None" ]]; then
  API_ID=$(awscmd apigateway create-rest-api --name "${API_NAME}" --query 'id' --output text)
fi
echo "[deploy] API ID: ${API_ID}"

ROOT_RESOURCE_ID=$(awscmd apigateway get-resources --rest-api-id "${API_ID}" --query "items[?path=='/'].id" --output text)

create_resource_if_missing() {
  local parent_id="$1" path_part="$2"
  local existing
  existing=$(awscmd apigateway get-resources --rest-api-id "${API_ID}" \
    --query "items[?parentId=='${parent_id}' && pathPart=='${path_part}'].id" --output text)
  if [[ -n "${existing}" && "${existing}" != "None" ]]; then
    echo "${existing}"
  else
    awscmd apigateway create-resource --rest-api-id "${API_ID}" --parent-id "${parent_id}" \
      --path-part "${path_part}" --query 'id' --output text
  fi
}

wire_route() {
  local resource_id="$1"
  awscmd apigateway put-method --rest-api-id "${API_ID}" --resource-id "${resource_id}" \
    --http-method GET --authorization-type NONE > /dev/null 2>&1 || true
  awscmd apigateway put-integration --rest-api-id "${API_ID}" --resource-id "${resource_id}" \
    --http-method GET --type AWS_PROXY --integration-http-method POST --uri "${LAMBDA_URI}" > /dev/null
}

echo "[deploy] wiring /runs"
RUNS_ID=$(create_resource_if_missing "${ROOT_RESOURCE_ID}" "runs")
wire_route "${RUNS_ID}"

echo "[deploy] wiring /runs/{run_id}/clusters"
RUN_ID_RESOURCE=$(create_resource_if_missing "${RUNS_ID}" "{run_id}")
RUN_CLUSTERS_RESOURCE=$(create_resource_if_missing "${RUN_ID_RESOURCE}" "clusters")
wire_route "${RUN_CLUSTERS_RESOURCE}"

echo "[deploy] wiring /trends"
TRENDS_ID=$(create_resource_if_missing "${ROOT_RESOURCE_ID}" "trends")
wire_route "${TRENDS_ID}"

echo "[deploy] wiring /clusters/{cluster_id}"
CLUSTERS_ID=$(create_resource_if_missing "${ROOT_RESOURCE_ID}" "clusters")
CLUSTER_ID_RESOURCE=$(create_resource_if_missing "${CLUSTERS_ID}" "{cluster_id}")
wire_route "${CLUSTER_ID_RESOURCE}"

echo "[deploy] wiring /similarity-search"
SIMILARITY_ID=$(create_resource_if_missing "${ROOT_RESOURCE_ID}" "similarity-search")
wire_route "${SIMILARITY_ID}"

echo "[deploy] deploying stage: ${STAGE_NAME}"
awscmd apigateway create-deployment --rest-api-id "${API_ID}" --stage-name "${STAGE_NAME}" > /dev/null

INVOKE_URL="${ENDPOINT}/restapis/${API_ID}/${STAGE_NAME}/_user_request_"
echo -n "${INVOKE_URL}" > "${ROOT_DIR}/.api_endpoint"
echo "[deploy] wrote invoke URL to .api_endpoint (dashboard.py reads this by default)"
echo "[deploy] done."
echo "[deploy] Invoke base URL: ${INVOKE_URL}"
echo "  curl '${INVOKE_URL}/trends'"
echo "  curl '${INVOKE_URL}/clusters/0'"
echo "  curl '${INVOKE_URL}/similarity-search?filename=<name>'"
