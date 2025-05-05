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

# Start all less intensive processes in parallel
echo "Starting ${#LESS_INTENSIVE_COMMANDS[@]} less intensive processes in parallel..."
SHORT_PIDS=()
for cmd in "${LESS_INTENSIVE_COMMANDS[@]}"; do
    echo "Starting process: $cmd"
    eval "$cmd" &
    SHORT_PIDS+=($!)
done

# Run memory-intensive processes sequentially
echo "Running ${#MEMORY_INTENSIVE_COMMANDS[@]} memory-intensive processes sequentially..."
for cmd in "${MEMORY_INTENSIVE_COMMANDS[@]}"; do
    echo "Starting memory-intensive process: $cmd"
    eval "$cmd"
    echo "Memory-intensive process completed: $cmd"
done

# Wait for all less intensive processes to complete
echo "Waiting for less intensive processes to complete..."
for pid in "${SHORT_PIDS[@]}"; do
    wait $pid
done

echo "All processes completed."
