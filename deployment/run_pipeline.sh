#!/usr/bin/env bash
# Fashion Trend Analyzer – end-to-end deployment runner
# Runs: venv + deps -> src/run_pipeline.py (scrape -> segment -> cluster -> label -> trend_tracker -> sync) -> (optionally) dashboard
#
# This used to manually invoke each stage by module path (src.scrape_reformation,
# src.clip_embed_cluster, ...) which no longer exist after the pipeline was
# restructured around run_id-scoped runs and incremental clustering. It now
# delegates the actual pipeline to src/run_pipeline.py, which is the single
# source of truth for stage order (see also: the dashboard's "Run New Scrape"
# button and scripts/run_pipeline_cron.sh, which call the same script).

set -Eeuo pipefail

# --- Config (defaults) --------------------------------------------------------
RUN_ID=""                   # empty = let run_pipeline.py default to a timestamp
MAX_IMAGES=40                # images to scrape
K=7                          # max new clusters to discover among unmatched images
MODEL="gpt-4o-mini"          # LLM for labeling
USE_ORIGINAL=true            # prefer originals when labeling
LAUNCH_DASHBOARD=true        # run Streamlit at the end
CLEAN=false                  # remove previous intermediates
PYTHON_BIN="${PYTHON_BIN:-python3}"  # allow override, e.g., PYTHON_BIN=python

# --- CLI flags ---------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --run-id <id>     Run identifier (default: run_pipeline.py's timestamp default)
  --max-images <n>  Images to scrape (default: ${MAX_IMAGES})
  -k <int>          Max new clusters to discover among unmatched images (default: ${K})
  -m <str>          LLM model for labeling (default: ${MODEL})
  --no-original     Do NOT pass --use-original to labeling step
  --no-dashboard     Skip launching Streamlit dashboard
  --clean            Clean previous intermediates (images/segmented, images/clustered)
  -h, --help         Show this help
Env:
  PYTHON_BIN=python3|python   Override python binary
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --max-images) MAX_IMAGES="$2"; shift 2 ;;
    -k) K="$2"; shift 2 ;;
    -m) MODEL="$2"; shift 2 ;;
    --no-original) USE_ORIGINAL=false; shift ;;
    --no-dashboard) LAUNCH_DASHBOARD=false; shift ;;
    --clean) CLEAN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# --- Paths -------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
VENV_DIR="${ROOT_DIR}/venv"
ENV_FILE="${ROOT_DIR}/.env"

mkdir -p "${LOG_DIR}" "${ROOT_DIR}/data" "${ROOT_DIR}/images" "${ROOT_DIR}/models"

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }
log() { echo "[$(timestamp)] $*"; }

# --- Environment --------------------------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
  log "Loading environment from .env"
  set -a; source "${ENV_FILE}"; set +a
else
  log "No .env found at ${ENV_FILE} (continuing with current env)"
fi

# --- Optional clean -----------------------------------------------------------
if [[ "${CLEAN}" == "true" ]]; then
  log "Cleaning previous intermediates… (NOTE: does not reset data/cluster_registry.json -- "
  log "delete that yourself if you want a clean incremental-clustering bootstrap)"
  rm -rf "${ROOT_DIR}/images/segmented_images" \
         "${ROOT_DIR}/images/clustered_images"
  mkdir -p "${ROOT_DIR}/images/segmented_images" "${ROOT_DIR}/images/clustered_images"
fi

# --- Python & deps ------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating venv…"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip >/dev/null

log "Installing requirements…"
pip install -r "${ROOT_DIR}/requirements.txt" | tee "${LOG_DIR}/00_requirements_$(timestamp).log"

log "Installing Playwright browser…"
python -m playwright install chromium | tee "${LOG_DIR}/00_playwright_$(timestamp).log"

# --- Run the pipeline ---------------------------------------------------------
# scrape -> segment (garment-level) -> cluster (incremental, registry-based) ->
# label -> trend_tracker -> sync to S3 (best-effort if AWS env vars aren't set)
RUN_ARGS=(--max-images "${MAX_IMAGES}" --k "${K}" --model "${MODEL}")
if [[ -n "${RUN_ID}" ]]; then
  RUN_ARGS+=(--run-id "${RUN_ID}")
fi
if [[ "${USE_ORIGINAL}" == "false" ]]; then
  RUN_ARGS+=(--no-original)
fi

set -x
python "${ROOT_DIR}/src/run_pipeline.py" "${RUN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/pipeline_$(timestamp).log"
set +x

# 5) Dashboard (optional)
if [[ "${LAUNCH_DASHBOARD}" == "true" ]]; then
  log "Launching dashboard (Ctrl+C to stop)…"
  # Streamlit runs in foreground; write URL to log as well
  streamlit run "${ROOT_DIR}/src/dashboard.py" 2>&1 | tee "${LOG_DIR}/05_dashboard_$(timestamp).log"
else
  log "Dashboard launch skipped (--no-dashboard)."
  log "You can start it manually with:"
  echo "  source ${VENV_DIR}/bin/activate && streamlit run src/dashboard.py"
fi

log "Done."
