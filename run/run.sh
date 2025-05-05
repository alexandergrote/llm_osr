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
    "python run/short_process.py"
    "python run/short_process.py"
    "python run/short_process.py"
)

# Maximum number of parallel less intensive processes
MAX_PARALLEL=2

# Start short processes in a separate subshell
(
    echo "Starting ${#LESS_INTENSIVE_COMMANDS[@]} less intensive processes with max $MAX_PARALLEL in parallel..."
    SHORT_PIDS=()
    RUNNING=0
    
    for cmd in "${LESS_INTENSIVE_COMMANDS[@]}"; do
        # If we've reached the maximum number of parallel processes, wait for one to finish
        if [ $RUNNING -ge $MAX_PARALLEL ]; then
            # Wait for the first process to finish
            if [ ${#SHORT_PIDS[@]} -gt 0 ]; then
                echo "Reached max parallel processes, waiting for one to complete..."
                wait ${SHORT_PIDS[0]}
                # Remove the first PID from the array
                SHORT_PIDS=("${SHORT_PIDS[@]:1}")
                RUNNING=$((RUNNING - 1))
            fi
        fi
        
        # Start a new process
        echo "Starting short process: $cmd"
        eval "$cmd" &
        SHORT_PIDS+=($!)
        RUNNING=$((RUNNING + 1))
    done
    
    # Wait for all short processes to complete
    for pid in "${SHORT_PIDS[@]}"; do
        wait $pid
    done
    echo "All short processes completed."
) &
SHORT_SHELL_PID=$!

# Run memory-intensive processes sequentially in the main shell
echo "Running ${#MEMORY_INTENSIVE_COMMANDS[@]} memory-intensive processes sequentially..."
for cmd in "${MEMORY_INTENSIVE_COMMANDS[@]}"; do
    echo "Starting memory-intensive process: $cmd"
    eval "$cmd"
    echo "Memory-intensive process completed: $cmd"
done

# Wait for the short processes subshell to complete
echo "Waiting for any remaining short processes to complete..."
wait $SHORT_SHELL_PID

echo "All processes completed."
