#!/bin/bash
# Check dependencies for security (pip-audit) and updates (pip list --outdated).
# Run from repo root with the same Python/conda env you use for the server.
# Usage: ./scripts/check_dependencies.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

echo "=== pip-audit (known vulnerabilities) ==="
if python -m pip show pip-audit &>/dev/null; then
    python -m pip audit 2>/dev/null || true
else
    echo "pip-audit not installed. Install with: pip install pip-audit"
    echo "Then run: pip audit"
fi

echo ""
echo "=== pip list --outdated ==="
python -m pip list --outdated 2>/dev/null || true

echo ""
echo "To fix vulnerabilities: pip audit --fix (review first)"
echo "To update packages: pip install -U <package> then update requirements.txt"
