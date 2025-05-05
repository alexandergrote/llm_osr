#!/bin/bash

# Source the utility functions
source run/util.sh

# Define an array of less intensive commands
LESS_INTENSIVE_COMMANDS=(
    "$PYTHON_PATH run/short_process.py"
    "$PYTHON_PATH run/short_process.py"
    "$PYTHON_PATH run/short_process.py"
    "$PYTHON_PATH run/short_process.py"
    "$PYTHON_PATH run/short_process.py"
)

# Maximum number of parallel less intensive processes
MAX_PARALLEL=2

# Run less intensive commands with parallelism limit, each in a new terminal window
echo "Starting less intensive processes with parallelism limit of $MAX_PARALLEL..."
running=0
for cmd in "${LESS_INTENSIVE_COMMANDS[@]}"; do
    # Start the command in a new terminal window
    open_terminal "$cmd"
    echo "Started: $cmd"
    
    # Increment the counter
    ((running++))
    
    # If we've reached the maximum number of parallel processes, wait a bit before continuing
    if [ $running -ge $MAX_PARALLEL ]; then
        echo "Maximum parallel processes reached. Waiting before starting more..."
        sleep 3  # Wait 3 seconds before starting more processes
        running=0
    fi
done
