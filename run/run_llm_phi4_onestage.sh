#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

osascript <<EOF
tell application "Terminal"
    activate
    do script "bash $SCRIPT_DIR/run_llm_phi4_onestage_clinc.sh"
end tell
EOF

osascript <<EOF
tell application "Terminal"
    activate
    do script "bash $SCRIPT_DIR/run_llm_phi4_onestage_hwu.sh"
end tell
EOF

osascript <<EOF
tell application "Terminal"
    activate
    do script "bash $SCRIPT_DIR/run_llm_phi4_onestage_banking.sh"
end tell
EOF

exit 0