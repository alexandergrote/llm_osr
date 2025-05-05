#!/bin/bash

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