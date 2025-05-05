#!/bin/bash

# Define an array of memory-intensive commands
MEMORY_INTENSIVE_COMMANDS=(
    "python run/long_process.py"
    "python run/long_process.py"
)

# Define an array of less intensive commands
LESS_INTENSIVE_COMMANDS=(
    "python run/short_process.py"
    "python run/short_process.py"
    "python run/short_process.py"
)

# Maximum number of parallel less intensive processes
MAX_PARALLEL=2

# Run memory-intensive processes sequentially first
echo "Running ${#MEMORY_INTENSIVE_COMMANDS[@]} memory-intensive processes sequentially..."
for cmd in "${MEMORY_INTENSIVE_COMMANDS[@]}"; do
    echo "Starting memory-intensive process: $cmd"
    eval "$cmd"
    echo "Memory-intensive process completed: $cmd"
done

# Start less intensive processes with parallelism limit
echo "Starting ${#LESS_INTENSIVE_COMMANDS[@]} less intensive processes with max $MAX_PARALLEL in parallel..."
SHORT_PIDS=()
RUNNING=0
for cmd in "${LESS_INTENSIVE_COMMANDS[@]}"; do
    # If we've reached the maximum number of parallel processes, wait for one to finish
    if [ $RUNNING -ge $MAX_PARALLEL ]; then
        # Wait for any process to finish
        if [ ${#SHORT_PIDS[@]} -gt 0 ]; then
            echo "Reached max parallel processes, waiting for one to complete..."
            wait -n ${SHORT_PIDS[@]}
            RUNNING=$((RUNNING - 1))
        fi
    fi
    
    # Start a new process
    echo "Starting process: $cmd"
    eval "$cmd" &
    SHORT_PIDS+=($!)
    RUNNING=$((RUNNING + 1))
done

# Wait for all less intensive processes to complete
echo "Waiting for less intensive processes to complete..."
for pid in "${SHORT_PIDS[@]}"; do
    wait $pid
done

echo "All processes completed."
