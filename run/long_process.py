import time

if __name__ == "__main__":
    start_time = time.time()
    # Simulate a long process
    time.sleep(5)
    end_time = time.time()
    print(f"Process took {end_time - start_time} seconds")