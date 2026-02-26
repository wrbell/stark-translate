#!/usr/bin/env bash
# Launch the live bilingual pipeline for church service
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source stt_env/bin/activate

# Print display URLs
IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I | awk '{print $1}')
echo "========================================="
echo "  Stark Road Live Translation"
echo "========================================="
echo ""
echo "  Operator:  http://localhost:8080/displays/ab_display.html"
echo "  Projector: http://localhost:8080/displays/audience_display.html"
echo "  Mobile:    http://${IP}:8080/displays/mobile_display.html"
echo ""

python dry_run_ab.py "$@"
