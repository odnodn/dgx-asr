#!/bin/bash
# Start Neuro-Parakeet MLX Server

PORT=${PORT:-8002}
MODEL=${PARAKEET_MODEL:-NeurologyAI/neuro-parakeet-mlx}
ENV=${ENV:-development}

# Production: require API_KEY in environment, no prompt; development: prompt if no TTY
if [ "$ENV" = "production" ]; then
    if [ -z "${API_KEY}" ]; then
        echo "Error: ENV=production requires API_KEY to be set in the environment."
        exit 1
    fi
    ALLOW_LAN=${ALLOW_LAN:-0}
    if [ "$ALLOW_LAN" = "1" ]; then
        echo "Warning: Production with ALLOW_LAN=1 is not recommended. Set CORS_ORIGINS to specific origins."
    fi
else
    if [ "${USE_API_KEY:-1}" = "1" ] && [ -z "${API_KEY}" ]; then
        if [ -t 0 ]; then
            echo -n "Enter API key (or leave empty to disable): "
            read -s API_KEY
            echo ""
        else
            echo "  No TTY: API key not set. Set API_KEY in the environment or run interactively to enter one."
            API_KEY=""
        fi
    fi
    ALLOW_LAN=${ALLOW_LAN:-1}
fi
CONDA_ENV_NAME="parakeet"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

# Allow all incoming traffic from local network: bind to all interfaces and allow CORS from any origin
if [ "$ALLOW_LAN" = "1" ] || [ "$ALLOW_LAN" = "true" ] || [ "$ALLOW_LAN" = "yes" ]; then
    export BIND="${BIND:-0.0.0.0}"
    export CORS_ORIGINS="${CORS_ORIGINS:-*}"
fi

export ENV="$ENV"
echo "Starting Neuro-Parakeet MLX Server..."
echo "  ENV: $ENV"
echo "  Port: $PORT"
echo "  Model: $MODEL"
echo "  Bind: ${BIND:-127.0.0.1}"
if [ -n "$API_KEY" ]; then
    echo "  API Key: ENABLED"
else
    echo "  API Key: DISABLED"
fi
echo ""

# Use conda: create parakeet env if missing, activate it, install deps, then run
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true

    if ! conda run -n "$CONDA_ENV_NAME" true 2>/dev/null; then
        echo "  Creating conda environment: $CONDA_ENV_NAME (python=3.10)..."
        if ! conda create -n "$CONDA_ENV_NAME" python=3.10 -y 2>/dev/null; then
            echo "  Conda may require accepting Terms of Service. Run once:"
            echo "    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main"
            echo "    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r"
            echo "  Then run ./start_server.sh again."
            exit 1
        fi
    fi

    conda activate "$CONDA_ENV_NAME"
    echo "  Using conda environment: $CONDA_ENV_NAME"
    echo "  Python: $(which python)"

    if ! python -c "import fastapi" 2>/dev/null; then
        echo "  Installing dependencies into $CONDA_ENV_NAME..."
        pip install -r "$REQUIREMENTS"
    fi
else
    echo "  Warning: conda not found. Using system Python."
    echo ""
fi

# Use python if available, otherwise python3
PYTHON=$(command -v python 2>/dev/null || command -v python3)
if [ -z "$PYTHON" ]; then
    echo "  Error: neither python nor python3 found in PATH."
    exit 1
fi

export PARAKEET_MODEL="$MODEL"
[ -n "$API_KEY" ] && export API_KEY="$API_KEY"
[ -n "$LOG_LEVEL" ] && export LOG_LEVEL="$LOG_LEVEL"
[ -n "$CORS_ORIGINS" ] && export CORS_ORIGINS="$CORS_ORIGINS"

# Prevent Mac from sleeping while the server runs (display can still sleep)
echo "  Caffeinate: preventing system sleep while server runs."
exec caffeinate -i -s -- "$PYTHON" parakeet_server.py --port "$PORT" --model "$MODEL"

