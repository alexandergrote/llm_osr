"""
This is a utility script to see in which json files a supplied text appears
"""

import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple


from src.ml.util.job_queue import Job
from src.util.constants import Directory


def get_json_files() -> List[Path]:
    """
    Get all JSON files in the specified directory.
    """
    return list(Path(Directory.JOB_DIR).rglob("*.json"))


def get_json_content(file: Path) -> Job:
    """
    Load the content of a JSON file.
    """
    job = Job.from_json_file(file)
    return job


def search_text_in_json_file(file: Path, text: str) -> Tuple[Path, bool]:
    """
    Search for the provided text in a JSON file.
    Returns a tuple of (file_path, found_status)
    """
    try:
        job = get_json_content(file)
        found = text in str(job.request_dict)
        return (file, found)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return (file, False)


def process_files_in_parallel(files: List[Path], text: str, max_workers: int = None) -> List[Path]:
    """
    Process files in parallel using threading.
    Returns a list of files where the text was found.
    """
    matching_files = []
    
    # If max_workers is None, ThreadPoolExecutor will use a default value
    # based on the number of processors on the machine
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary of futures to their file paths
        future_to_file = {
            executor.submit(search_text_in_json_file, file, text): file 
            for file in files
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                          total=len(files), 
                          desc="Searching files"):
            file_path, found = future.result()
            if found:
                matching_files.append(file_path)
                print(f"Text found in {file_path}")
    
    return matching_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect jobs for a given text")
    parser.add_argument("text", type=str, help="The text to search for")
    parser.add_argument("--workers", type=int, default=None, 
                        help="Number of worker threads (default: auto)")
    args = parser.parse_args()

    directory_path = Path(Directory.JOB_DIR)
    files = get_json_files()
    
    # For testing - remove this line in production
    args.text = "What are the currency exchange fees?"
    
    print(f"Searching for text: '{args.text}' in {len(files)} files")
    matching_files = process_files_in_parallel(files, args.text, args.workers)
    
    print(f"\nFound text in {len(matching_files)} files")
