#!/bin/bash

# Source the utility functions
source run/util.sh

# Define an array of memory-intensive commands
MEMORY_INTENSIVE_COMMANDS=(
    "python run/long_process.py"
    "python run/long_process.py"
)

# Run memory-intensive commands sequentially, each in a new terminal window
echo "Starting memory-intensive processes..."
for cmd in "${MEMORY_INTENSIVE_COMMANDS[@]}"; do
    open_terminal "$cmd"
    echo "Started: $cmd"
done
