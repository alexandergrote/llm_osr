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

# Detect the terminal emulator to use
if command -v gnome-terminal &> /dev/null; then
    TERMINAL="gnome-terminal --"
elif command -v xterm &> /dev/null; then
    TERMINAL="xterm -e"
elif command -v konsole &> /dev/null; then
    TERMINAL="konsole -e"
elif command -v terminal &> /dev/null; then
    TERMINAL="terminal -e"
elif command -v iTerm &> /dev/null; then
    TERMINAL="iTerm -e"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS Terminal
    TERMINAL="open -a Terminal"
else
    echo "No supported terminal emulator found. Falling back to background execution."
    TERMINAL=""
fi

# Run memory-intensive commands sequentially, each in a new terminal window
for cmd in "${MEMORY_INTENSIVE_COMMANDS[@]}"; do
    if [ -n "$TERMINAL" ]; then
        $TERMINAL bash -c "$cmd; echo 'Press Enter to close this window...'; read" &
        sleep 1  # Give the terminal time to open
    else
        bash -c "$cmd" 
    fi
done

# Run less intensive commands with parallelism limit, each in a new terminal window
running=0
for cmd in "${LESS_INTENSIVE_COMMANDS[@]}"; do
    # Start the command in a new terminal window
    if [ -n "$TERMINAL" ]; then
        $TERMINAL bash -c "$cmd; echo 'Press Enter to close this window...'; read" &
        sleep 1  # Give the terminal time to open
    else
        bash -c "$cmd" &
    fi
    
    # Increment the counter
    ((running++))
    
    # If we've reached the maximum number of parallel processes, wait for one to finish
    if [ $running -ge $MAX_PARALLEL ]; then
        running=0
    fi
done
