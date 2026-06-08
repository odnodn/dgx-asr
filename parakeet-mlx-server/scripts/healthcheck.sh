#!/bin/bash
# Health check for Parakeet server. Exit 0 if healthy, 1 otherwise.
# Usage: ./healthcheck.sh [BASE_URL]
# Example: ./healthcheck.sh https://your-domain.example.com/stt
#          ./healthcheck.sh http://127.0.0.1:8002  (direct)
# Env: HEALTHCHECK_URL overrides the first argument.

BASE_URL="${HEALTHCHECK_URL:-${1:-http://127.0.0.1:8002}}"
URL="${BASE_URL%/}/health"

HTTP=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 --max-time 10 "$URL" 2>/dev/null || echo "000")
if [ "$HTTP" != "200" ]; then
    echo "Health check failed: HTTP $HTTP ($URL)" >&2
    exit 1
fi

# Optional: require "status":"healthy" and "model_loaded":true in JSON
BODY=$(curl -s --connect-timeout 5 --max-time 10 "$URL" 2>/dev/null)
if echo "$BODY" | grep -q '"status":"healthy"' && echo "$BODY" | grep -q '"model_loaded":true'; then
    echo "OK: $URL"
    exit 0
fi
echo "Health check failed: unhealthy or model not loaded ($URL)" >&2
exit 1
