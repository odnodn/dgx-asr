#!/bin/bash
# Install reranker server as a LaunchAgent so it restarts when it crashes.
# Expects: $HOME/reranker_server.py and $HOME/reranker-env (venv with uvicorn).

set -e
RERANKER_DIR="${RERANKER_DIR:-$HOME}"
UVICORN="${RERANKER_DIR}/reranker-env/bin/uvicorn"
SERVER_MODULE="reranker_server:app"
PORT="${RERANKER_PORT:-8000}"
LOGS_DIR="${RERANKER_DIR}/reranker-logs"
LAUNCH_AGENTS="${HOME}/Library/LaunchAgents"
PLIST_NAME="com.reranker.server.plist"

if [ ! -f "${RERANKER_DIR}/reranker_server.py" ]; then
    echo "Error: reranker_server.py not found at ${RERANKER_DIR}/reranker_server.py"
    echo "Set RERANKER_DIR if the reranker lives elsewhere."
    exit 1
fi
if [ ! -x "$UVICORN" ]; then
    echo "Error: uvicorn not found at ${UVICORN}. Create the venv and install deps:"
    echo "  python3 -m venv ${RERANKER_DIR}/reranker-env"
    echo "  ${RERANKER_DIR}/reranker-env/bin/pip install sentence-transformers fastapi uvicorn"
    exit 1
fi

mkdir -p "$LOGS_DIR"

cat > "${LAUNCH_AGENTS}/${PLIST_NAME}" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.reranker.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>${UVICORN}</string>
        <string>${SERVER_MODULE}</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>${PORT}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${RERANKER_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>${LOGS_DIR}/reranker-server.log</string>
    <key>StandardErrorPath</key>
    <string>${LOGS_DIR}/reranker-server.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
EOF

echo "Installed LaunchAgent: ${LAUNCH_AGENTS}/${PLIST_NAME}"
echo "  Reranker: ${RERANKER_DIR}/reranker_server.py"
echo "  Port: ${PORT}"
echo "  Logs: ${LOGS_DIR}/reranker-server.log and reranker-server.err"
echo ""
echo "Starting (and restart on crash)..."
launchctl unload "${LAUNCH_AGENTS}/${PLIST_NAME}" 2>/dev/null || true
launchctl load "${LAUNCH_AGENTS}/${PLIST_NAME}"
echo "Done."
echo ""
echo "To stop: launchctl unload ${LAUNCH_AGENTS}/${PLIST_NAME}"
echo "To restart: run this script again."
