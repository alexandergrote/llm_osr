import time
import sys
import os

if __name__ == "__main__":
    # Get any command line arguments
    args = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "no args"
    
    # Get process ID
    pid = os.getpid()
    
    start_time = time.time()
    print(f"LONG PROCESS (PID: {pid}, Args: {args}) - Starting...")
    
    # Simulate a memory-intensive process
    time.sleep(5)
    
    end_time = time.time()
    print(f"LONG PROCESS (PID: {pid}) - Completed in {end_time - start_time:.2f}s")
