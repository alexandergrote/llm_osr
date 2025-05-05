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
pids=()

for cmd in "${LESS_INTENSIVE_COMMANDS[@]}"; do
    # If we've reached the maximum number of parallel processes, wait for one to finish
    if [ $running -ge $MAX_PARALLEL ]; then
        echo "Maximum parallel processes reached. Waiting for a process to finish..."
        # Wait for any process to finish
        wait -n "${pids[@]}" 2>/dev/null || true
        # Decrement the counter
        ((running--))
    fi
    
    # Start the command in a new terminal window and capture its PID
    open_terminal "$cmd" &
    current_pid=$!
    pids+=($current_pid)
    echo "Started: $cmd (PID: $current_pid)"
    
    # Increment the counter
    ((running++))
done

# Wait for all remaining processes to finish
echo "Waiting for all remaining processes to finish..."
wait "${pids[@]}" 2>/dev/null || true
echo "All processes completed."
