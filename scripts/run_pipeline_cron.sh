#!/usr/bin/env bash
# Wrapper for cron: activates the venv, sets AWS env vars (only needed if
# LocalStack/real AWS sync is being used), runs one full pipeline cycle, and
# logs output with a timestamped filename. This is the no-AWS-EventBridge
# alternative to scripts/schedule_new_arrivals_scrape.sh -- runs entirely on
# whatever machine cron fires on, no Lambda/Scheduler involved.
#
# Install with (daily at 6am):
#   crontab -e
#   0 6 * * * /path/to/fashion_trend_analyzer/scripts/run_pipeline_cron.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/cron"
mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"
source venv/bin/activate

# Only needed if you want the run synced to a deployed API (LocalStack or real AWS):
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://localhost:4566}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%dT%H-%M-%S).log"
python3 src/run_pipeline.py >> "${LOG_FILE}" 2>&1
echo "Pipeline run logged to ${LOG_FILE}"
