#!/bin/bash
# Run this on the Mac mini to avoid password prompts after restart/sleep.
# - Disables "require password when waking from sleep or screen saver"
# - Instructions for automatic login (so server comes up after reboot without typing password)

set -e

echo "=== Mac mini: reduce password prompts ==="
echo ""

# 1. No password when waking from sleep or after screen saver
echo "Disabling 'Require password when waking from sleep or screen saver'..."
defaults write com.apple.screensaver askForPassword -bool false
defaults write com.apple.screensaver askForPasswordDelay -int 0
echo "  Done. (Lock Screen will not ask for password on wake.)"
echo ""

# 2. Automatic login after restart (so LaunchAgents start without typing password)
echo "Automatic login (log in as you after reboot):"
echo "  1. Open System Settings → Users & Groups → Login Options"
echo "  2. Set 'Automatic login' to your user (e.g. the one that runs the server)"
echo "  3. Enter your password when prompted (stored securely for login only)."
echo ""
echo "If your Mac has FileVault enabled, automatic login may still require one password after a full power loss (then it stays logged in until next reboot)."
echo ""
echo "Done."
