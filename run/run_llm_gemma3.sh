#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

osascript <<EOF
tell application "Terminal"
    activate
    do script "bash $SCRIPT_DIR/run_llm_gemma3_mixed.sh"
end tell
EOF

osascript <<EOF
tell application "Terminal"
    activate
    do script "bash $SCRIPT_DIR/run_llm_gemma3_onestage.sh"
end tell
EOF

exit 0