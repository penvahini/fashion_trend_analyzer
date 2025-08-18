#!/usr/bin/env bash
# Fashion Trend Analyzer – end-to-end deployment runner
# Runs: venv + deps → scrape → segment → embed+cluster → label → (optionally) dashboard

set -Eeuo pipefail

# --- Config (defaults) --------------------------------------------------------
K=7                         # k-means clusters
PER_CLUSTER=4               # images per cluster for labeling
MODEL="gpt-4o-mini"         # LLM for labeling
USE_ORIGINAL=true           # prefer originals when labeling
LAUNCH_DASHBOARD=true       # run Streamlit at the end
CLEAN=false                 # remove previous intermediates
PYTHON_BIN="${PYTHON_BIN:-python3}"  # allow override, e.g., PYTHON_BIN=python

# --- CLI flags ---------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -k <int>        Number of clusters (default: ${K})
  -p <int>        Images per cluster for labeling (default: ${PER_CLUSTER})
  -m <str>        LLM model for labeling (default: ${MODEL})
  --no-original   Do NOT pass --use-original to labeling step
  --no-dashboard  Skip launching Streamlit dashboard
  --clean         Clean previous intermediates (images/segmented, images/clustered, data/derived)
  -h, --help      Show this help
Env:
  PYTHON_BIN=python3|python   Override python binary
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -k) K="$2"; shift 2 ;;
    -p) PER_CLUSTER="$2"; shift 2 ;;
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
  log "Cleaning previous intermediates…"
  rm -rf "${ROOT_DIR}/images/segmented_images" \
         "${ROOT_DIR}/images/clustered_images" \
         "${ROOT_DIR}/data/derived"
  mkdir -p "${ROOT_DIR}/images/segmented_images" "${ROOT_DIR}/images/clustered_images" "${ROOT_DIR}/data/derived"
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

# --- Steps -------------------------------------------------------------------
set -x

# 1) Scrape
python -m src.scrape_reformation 2>&1 | tee "${LOG_DIR}/01_scrape_$(timestamp).log"

# 2) Segment (SAM + custom YOLO)
python -m src.segment 2>&1 | tee "${LOG_DIR}/02_segment_$(timestamp).log"

# 3) Embed + cluster (CLIP + KMeans)
python -m src.clip_embed_cluster --k "${K}" 2>&1 | tee "${LOG_DIR}/03_embed_cluster_$(timestamp).log"

# 4) Label clusters (LLM summaries + keywords)
LABEL_ARGS=(--per-cluster "${PER_CLUSTER}" --model "${MODEL}")
if [[ "${USE_ORIGINAL}" == "true" ]]; then
  LABEL_ARGS+=(--use-original)
fi
python -m src.label_clusters "${LABEL_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/04_label_$(timestamp).log"

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
