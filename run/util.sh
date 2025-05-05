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
        echo "set -e" >> $TMPFILE  # Exit on error
        echo "$1" >> $TMPFILE
        echo "EXIT_CODE=\$?" >> $TMPFILE
        echo "if [ \$EXIT_CODE -ne 0 ]; then" >> $TMPFILE
        echo "  echo \"Command failed with exit code \$EXIT_CODE\"" >> $TMPFILE
        echo "  echo \"Press Enter to close this window...\"" >> $TMPFILE
        echo "  read" >> $TMPFILE
        echo "fi" >> $TMPFILE
        chmod +x $TMPFILE
        
        # Open Terminal with this script
        open -a Terminal $TMPFILE
        sleep 1  # Give the terminal time to open
    }
elif command -v gnome-terminal &> /dev/null; then
    function open_terminal() {
        gnome-terminal -- bash -c "cd $(pwd); set -e; $1; EXIT_CODE=\$?; if [ \$EXIT_CODE -ne 0 ]; then echo \"Command failed with exit code \$EXIT_CODE\"; echo \"Press Enter to close this window...\"; read; fi"
        sleep 1
    }
elif command -v xterm &> /dev/null; then
    function open_terminal() {
        xterm -e bash -c "cd $(pwd); set -e; $1; EXIT_CODE=\$?; if [ \$EXIT_CODE -ne 0 ]; then echo \"Command failed with exit code \$EXIT_CODE\"; echo \"Press Enter to close this window...\"; read; fi" &
        sleep 1
    }
elif command -v konsole &> /dev/null; then
    function open_terminal() {
        konsole -e bash -c "cd $(pwd); set -e; $1; EXIT_CODE=\$?; if [ \$EXIT_CODE -ne 0 ]; then echo \"Command failed with exit code \$EXIT_CODE\"; echo \"Press Enter to close this window...\"; read; fi" &
        sleep 1
    }
else
    echo "No supported terminal emulator found. Falling back to background execution."
    function open_terminal() {
        bash -c "cd $(pwd); set -e; $1; EXIT_CODE=\$?; if [ \$EXIT_CODE -ne 0 ]; then echo \"Command failed with exit code \$EXIT_CODE\"; fi" &
    }
fi
