import time

if __name__ == "__main__":
    start_time = time.time()
    # Simulate a long process
    time.sleep(5)
    end_time = time.time()
    start_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    end_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    print(f"Short start: {start_time_formatted}, Process took {end_time - start_time} seconds")