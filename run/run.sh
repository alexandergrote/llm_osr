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
        echo "$1" >> $TMPFILE
        echo "echo 'Press Enter to close this window...'" >> $TMPFILE
        echo "read" >> $TMPFILE
        chmod +x $TMPFILE
        
        # Open Terminal with this script
        open -a Terminal $TMPFILE
        sleep 1  # Give the terminal time to open
    }
elif command -v gnome-terminal &> /dev/null; then
    function open_terminal() {
        gnome-terminal -- bash -c "$1; echo 'Press Enter to close this window...'; read"
        sleep 1
    }
elif command -v xterm &> /dev/null; then
    function open_terminal() {
        xterm -e bash -c "$1; echo 'Press Enter to close this window...'; read" &
        sleep 1
    }
elif command -v konsole &> /dev/null; then
    function open_terminal() {
        konsole -e bash -c "$1; echo 'Press Enter to close this window...'; read" &
        sleep 1
    }
else
    echo "No supported terminal emulator found. Falling back to background execution."
    function open_terminal() {
        bash -c "$1" &
    }
fi

# Run memory-intensive commands sequentially, each in a new terminal window
for cmd in "${MEMORY_INTENSIVE_COMMANDS[@]}"; do
    open_terminal "$cmd"
done

# Run less intensive commands with parallelism limit, each in a new terminal window
running=0
for cmd in "${LESS_INTENSIVE_COMMANDS[@]}"; do
    # Start the command in a new terminal window
    open_terminal "$cmd"
    
    # Increment the counter
    ((running++))
    
    # If we've reached the maximum number of parallel processes, wait for one to finish
    if [ $running -ge $MAX_PARALLEL ]; then
        echo "Maximum parallel processes reached. Press Enter to continue with the next batch..."
        read
        running=0
    fi
done

echo "All processes have been started."
