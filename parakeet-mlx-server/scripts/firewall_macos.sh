#!/bin/bash
# macOS firewall notes for Parakeet. Run relevant commands as needed.
# Prefer BIND=127.0.0.1 so only nginx hits the app; then you don't need to open 8002.

echo "=== Option 1: App only on localhost (recommended) ==="
echo "Start server with BIND=127.0.0.1 (default in production)."
echo "Only nginx listens on a public port (80/443). No firewall rule needed for 8002."
echo ""
echo "=== Option 2: App on 0.0.0.0:8002 – restrict by firewall ==="
echo "If you must bind to 0.0.0.0, restrict port 8002 to trusted IPs:"
echo "  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/python"
echo "  Or in System Settings > Network > Firewall: allow only your LAN or nginx host."
echo ""
echo "=== Check current firewall ==="
echo "  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate"
echo "  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --listapps | head -20"
