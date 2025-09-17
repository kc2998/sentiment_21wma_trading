#!/usr/bin/env bash
set -euo pipefail

# --- config / defaults ---
ENV_NAME="sentiment-21wma"
PORT="8501"
NO_BROWSER=0
USE_VENV=0
DEFAULT_TICKER="${DEFAULT_TICKER:-AAPL}"
FINNHUB_API_KEY="${FINNHUB_API_KEY:-}"
FINNHUB_KEY_FILE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --key <KEY>           Finnhub API key (actual key string)
  --key-file <PATH>     Path to a file that contains the Finnhub API key
  --ticker <TICKER>     Default ticker for the app UI (default: AAPL)
  --port <PORT>         Streamlit server port (default: 8501)
  --no-browser          Run headless (don’t open a browser)
  --use-venv            Use Python venv + pip instead of Conda
  -h, --help            Show this help

Examples:
  $0 --key "abcdef123..." --ticker MSFT
  $0 --key-file ~/.keys/finnhub.txt --port 8502
  $0 --use-venv --no-browser
EOF
}

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --key)        FINNHUB_API_KEY="$2"; shift 2 ;;
    --key-file)   FINNHUB_KEY_FILE="$2"; shift 2 ;;
    --ticker)     DEFAULT_TICKER="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --no-browser) NO_BROWSER=1; shift ;;
    --use-venv)   USE_VENV=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# --- read key from file if provided and not already set ---
if [[ -z "${FINNHUB_API_KEY}" && -n "${FINNHUB_KEY_FILE}" ]]; then
  if [[ -f "${FINNHUB_KEY_FILE}" ]]; then
    FINNHUB_API_KEY="$(cat "${FINNHUB_KEY_FILE}")"
  else
    echo "Key file not found: ${FINNHUB_KEY_FILE}" >&2
    exit 1
  fi
fi

# --- prompt for key if still empty ---
if [[ -z "${FINNHUB_API_KEY}" ]]; then
  read -r -p "Enter Finnhub API key (leave blank to skip): " FINNHUB_API_KEY || true
fi

# --- write Streamlit secrets if we have a key ---
if [[ -n "${FINNHUB_API_KEY}" ]]; then
  mkdir -p ".streamlit"
  cat > .streamlit/secrets.toml <<EOF
FINNHUB_API_KEY = "${FINNHUB_API_KEY}"
EOF
  echo "Wrote .streamlit/secrets.toml"
fi

# --- helper: command exists ---
command_exists() { command -v "$1" >/dev/null 2>&1; }

# --- activate Conda or set up venv ---
if [[ "${USE_VENV}" -eq 0 ]] && command_exists conda; then
  echo "Using Conda environment: ${ENV_NAME}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  # create env if missing
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ -f "environment-macos.yml" ]]; then
      echo "Creating Conda env from environment-macos.yml ..."
      conda env create -f environment-macos.yml
    else
      echo "environment-macos.yml not found; creating minimal env ..."
      conda create -y -n "${ENV_NAME}" python=3.11
      conda activate "${ENV_NAME}"
      pip install -r requirements.txt
      conda deactivate
    fi
  fi

  conda activate "${ENV_NAME}"
  # Apple Silicon nicety
  export PYTORCH_ENABLE_MPS_FALLBACK=1

else
  echo "Conda not available or --use-venv set; using Python venv"
  if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

# --- pass default ticker via env (optional) ---
export DEFAULT_TICKER="${DEFAULT_TICKER}"

# --- launch Streamlit ---
ARGS=( "--server.port" "${PORT}" )
if [[ "${NO_BROWSER}" -eq 1 ]]; then
  ARGS+=( "--server.headless" "true" )
fi

echo "Starting app on http://localhost:${PORT} (DEFAULT_TICKER=${DEFAULT_TICKER})"
streamlit run app.py "${ARGS[@]}"
