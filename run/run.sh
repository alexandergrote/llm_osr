#!/bin/bash

# Define the memory-intensive command
MEMORY_INTENSIVE_COMMAND="python run/long_process.py"

# Define an array of less intensive commands
LESS_INTENSIVE_COMMANDS=(
    "python src/experiments/cli.py bench --filter-name \".*simple.*\""
    "python run/short_process.py"
    "python run/short_process.py --param1 value1"
    "python run/short_process.py --param2 value2"
)

# Run the memory-intensive process first
echo "Starting memory-intensive process..."
eval "$MEMORY_INTENSIVE_COMMAND" &
LONG_PID=$!

# Run all less intensive processes
echo "Starting ${#LESS_INTENSIVE_COMMANDS[@]} less intensive processes..."
SHORT_PIDS=()
for cmd in "${LESS_INTENSIVE_COMMANDS[@]}"; do
    echo "Starting process: $cmd"
    eval "$cmd" &
    SHORT_PIDS+=($!)
done

# Wait for the memory-intensive process to complete
echo "Waiting for memory-intensive process to complete..."
wait $LONG_PID
echo "Memory-intensive process completed."

# Wait for all less intensive processes to complete
echo "Waiting for less intensive processes to complete..."
for pid in "${SHORT_PIDS[@]}"; do
    wait $pid
done

echo "All processes completed."
