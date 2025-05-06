#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
CLI_FILE="$SCRIPT_DIR/src/experiments/cli.py"
PYTHON_PATH="/Users/alex/Code/llm_osr/venv/bin/python"

echo "Running LLMs..."
echo $CLI_FILE
echo $PYTHON_PATH

$PYTHON_PATH "$CLI_FILE" llm --filter-name ".*phi4.*" --skip-confirmation