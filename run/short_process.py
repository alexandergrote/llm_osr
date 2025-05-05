import time
import sys
import os

if __name__ == "__main__":
    # Get any command line arguments
    args = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "no args"
    
    # Get process ID
    pid = os.getpid()
    
    start_time = time.time()
    # Simulate a short process
    time.sleep(2)  # Reduced to 2 seconds to make it clearly shorter than long_process
    end_time = time.time()
    
    start_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print(f"SHORT PROCESS (PID: {pid}, Args: {args}) - Start: {start_time_formatted}, Duration: {end_time - start_time:.2f}s")
