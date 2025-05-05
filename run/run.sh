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
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS specific handling
    function open_terminal() {
        # Create a temporary script file
        TMPFILE=$(mktemp /tmp/terminal_cmd.XXXXXX)
        echo "#!/bin/bash" > $TMPFILE
        echo "cd $(pwd)" >> $TMPFILE  # Ensure we're in the same directory
        echo "$1" >> $TMPFILE
        chmod +x $TMPFILE
        
        # Open Terminal with this script
        open -a Terminal $TMPFILE
        sleep 1  # Give the terminal time to open
    }
elif command -v gnome-terminal &> /dev/null; then
    function open_terminal() {
        gnome-terminal -- bash -c "cd $(pwd); $1"
        sleep 1
    }
elif command -v xterm &> /dev/null; then
    function open_terminal() {
        xterm -e bash -c "cd $(pwd); $1" &
        sleep 1
    }
elif command -v konsole &> /dev/null; then
    function open_terminal() {
        konsole -e bash -c "cd $(pwd); $1" &
        sleep 1
    }
else
    echo "No supported terminal emulator found. Falling back to background execution."
    function open_terminal() {
        bash -c "cd $(pwd); $1" &
    }
fi

# Run memory-intensive commands sequentially, each in a new terminal window
echo "Starting memory-intensive processes..."
for cmd in "${MEMORY_INTENSIVE_COMMANDS[@]}"; do
    open_terminal "$cmd"
    echo "Started: $cmd"
done

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

echo "All processes have been started."
