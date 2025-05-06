#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

osascript <<EOF
tell application "Terminal"
    activate
    do script "bash $SCRIPT_DIR/run_llm_phi4.sh"
end tell
EOF

osascript <<EOF
tell application "Terminal"
    activate
    do script "bash $SCRIPT_DIR/run_llm_gemma3.sh"
end tell
EOF

osascript <<EOF
tell application "Terminal"
    activate
    do script "bash $SCRIPT_DIR/run_llm_llama8.sh"
end tell
EOF

osascript <<EOF
tell application "Terminal"
    activate
    do script "bash $SCRIPT_DIR/run_llm_llama70.sh"
end tell
EOF