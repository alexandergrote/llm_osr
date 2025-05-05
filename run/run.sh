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
)

# Maximum number of parallel less intensive processes
MAX_PARALLEL=2

# Run memory-intensive commands sequentially, each in a new shell
for cmd in "${MEMORY_INTENSIVE_COMMANDS[@]}"; do
    bash -c "$cmd"
done

# Run less intensive commands with parallelism limit, each in a new shell
running=0
for cmd in "${LESS_INTENSIVE_COMMANDS[@]}"; do
    # Start the command in a new shell
    bash -c "$cmd" &
    
    # Increment the counter
    ((running++))
    
    # If we've reached the maximum number of parallel processes, wait for one to finish
    if [ $running -ge $MAX_PARALLEL ]; then
        wait -n  # Wait for any child process to exit
        ((running--))
    fi
done

# Wait for all remaining processes to finish
wait
