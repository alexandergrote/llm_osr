#!/bin/bash

# Number of short processes to run
NUM_SHORT_PROCESSES=5

# Run the memory-intensive process first
echo "Starting memory-intensive process..."
python run/long_process.py &
LONG_PID=$!

# Run multiple short processes
echo "Starting $NUM_SHORT_PROCESSES short processes..."
for i in $(seq 1 $NUM_SHORT_PROCESSES); do
    echo "Starting short process $i..."
    python run/short_process.py &
    SHORT_PIDS[$i]=$!
done

# Wait for the long process to complete
echo "Waiting for memory-intensive process to complete..."
wait $LONG_PID
echo "Memory-intensive process completed."

# Wait for all short processes to complete
echo "Waiting for short processes to complete..."
for pid in ${SHORT_PIDS[@]}; do
    wait $pid
done

echo "All processes completed."
