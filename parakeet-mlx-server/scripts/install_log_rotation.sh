#!/bin/bash
# Install log rotation for Parakeet server logs (macOS newsyslog).
# Run with: sudo ./install_log_rotation.sh
# Requires: run from the parakeet-mlx-server repo directory, or set SCRIPT_DIR.

set -e
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOGS_DIR="${SCRIPT_DIR}/logs"
NEWSYSLOG_D="/etc/newsyslog.d"

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo: sudo $0"
    exit 1
fi

mkdir -p "$NEWSYSLOG_D"
# Format: logfile [owner:group] mode count size when flags
# Keep 7 rotated files, rotate at midnight (@T00), compress (Z), no signal (no pid file)
CONF="${NEWSYSLOG_D}/com.parakeet-mlx.server.conf"
cat > "$CONF" << EOF
# Parakeet MLX Server logs (created by install_log_rotation.sh)
${LOGS_DIR}/parakeet-server.log    $(logname):staff  640  7  *  @T00  Z
${LOGS_DIR}/parakeet-server.err    $(logname):staff  640  7  *  @T00  Z
EOF
echo "Installed: $CONF"
echo "Logs in $LOGS_DIR will rotate daily at midnight, keep 7 copies, compressed."
echo "Test: sudo newsyslog -v (dry run) or wait for next @T00."
