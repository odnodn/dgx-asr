#!/bin/bash
# Install Parakeet MLX Server as LaunchAgents:
# 1. Caffeinate: prevents sleep/idle so the Mac never sleeps.
# 2. Server: runs at login, restarts on crash (never stops).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
LAUNCH_AGENTS="${HOME}/Library/LaunchAgents"
SERVER_PLIST="com.parakeet-mlx.server.plist"
CAFFEINATE_PLIST="com.parakeet-mlx.caffeinate.plist"

mkdir -p "$LOGS_DIR"
mkdir -p "$LAUNCH_AGENTS"

# 1. Caffeinate plist: run forever, no sleep, no idle
cat > "${LAUNCH_AGENTS}/${CAFFEINATE_PLIST}" << 'CAFFEOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.parakeet-mlx.caffeinate</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/caffeinate</string>
        <string>-i</string>
        <string>-s</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
CAFFEOF

# 2. Server plist: run at login, restart on crash (throttle 10s between restarts)
# Escape value for plist (ampersand and angle brackets)
plist_escape() { echo "$1" | sed 's/&/\&amp;/g;s/</\&lt;/g;s/>/\&gt;/g'; }
EXTRA_ENV=""
[ -n "$API_KEY" ] && EXTRA_ENV="${EXTRA_ENV}
        <key>API_KEY</key>
        <string>$(plist_escape "$API_KEY")</string>"
[ -n "$ENV" ] && EXTRA_ENV="${EXTRA_ENV}
        <key>ENV</key>
        <string>$(plist_escape "$ENV")</string>"
[ -n "$CORS_ORIGINS" ] && EXTRA_ENV="${EXTRA_ENV}
        <key>CORS_ORIGINS</key>
        <string>$(plist_escape "$CORS_ORIGINS")</string>"
[ -n "$LOG_LEVEL" ] && EXTRA_ENV="${EXTRA_ENV}
        <key>LOG_LEVEL</key>
        <string>$(plist_escape "$LOG_LEVEL")</string>"

cat > "${SCRIPT_DIR}/${SERVER_PLIST}.generated" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.parakeet-mlx.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-lc</string>
        <string>cd ${SCRIPT_DIR} &amp;&amp; ./start_server.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>${LOGS_DIR}/parakeet-server.log</string>
    <key>StandardErrorPath</key>
    <string>${LOGS_DIR}/parakeet-server.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>${EXTRA_ENV}
    </dict>
</dict>
</plist>
EOF

cp "${SCRIPT_DIR}/${SERVER_PLIST}.generated" "${LAUNCH_AGENTS}/${SERVER_PLIST}"
rm -f "${SCRIPT_DIR}/${SERVER_PLIST}.generated"

echo "Installed LaunchAgents:"
echo "  1. ${CAFFEINATE_PLIST}  (no sleep, no idle)"
echo "  2. ${SERVER_PLIST}     (server, restarts on crash)"
echo ""
echo "Starting both now (and at every login)..."
launchctl unload "${LAUNCH_AGENTS}/${CAFFEINATE_PLIST}" 2>/dev/null || true
launchctl unload "${LAUNCH_AGENTS}/${SERVER_PLIST}"     2>/dev/null || true
launchctl load "${LAUNCH_AGENTS}/${CAFFEINATE_PLIST}"
launchctl load "${LAUNCH_AGENTS}/${SERVER_PLIST}"
echo "Done."
echo ""
echo "To stop everything:"
echo "  launchctl unload ${LAUNCH_AGENTS}/${CAFFEINATE_PLIST}"
echo "  launchctl unload ${LAUNCH_AGENTS}/${SERVER_PLIST}"
echo ""
echo "Logs: ${LOGS_DIR}/parakeet-server.log and parakeet-server.err"
echo ""
echo "Note: The server LaunchAgent uses a login shell (-lc) so that conda is available."
echo "      Ensure conda is initialized in your login profile (e.g. ~/.zshrc or ~/.bash_profile) on the Mac mini."
if [ -n "$API_KEY" ] || [ "$ENV" = "production" ]; then
    echo "      Production env vars (API_KEY, ENV, CORS_ORIGINS) were set in the plist from this run."
    echo "      To change them: unload the agent, edit ${LAUNCH_AGENTS}/${SERVER_PLIST}, then load again."
fi
