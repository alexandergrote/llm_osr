"""
This is a utility script to see in which json files a supplied text appears
"""

import concurrent.futures
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict

from src.ml.util.job_queue import Job
from src.util.constants import Directory


def get_json_files() -> List[Path]:
    """
    Get all JSON files in the specified directory.
    """

    result = []

    for el in Path(Directory.JOB_DIR).rglob("*.json"): 

        if "HFEmbeddingPreprocessor" in str(el):
            continue

        result.append(el)

    return result


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
    
    Args:
        files: List of file paths to search
        text: Text to search for
        max_workers: Number of worker threads (default: CPU count * 2)
    """
    matching_files = []
    
    # If max_workers is None, use CPU count * 2 for optimal I/O-bound performance
    if max_workers is None:
        max_workers = os.cpu_count() * 2
        
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
    
    return matching_files


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Inspect jobs for a given text")
    parser.add_argument("text", type=str, help="The text to search for")
    parser.add_argument("--workers", type=int, default=None, 
                        help=f"Number of worker threads (default: CPU count * 2 = {os.cpu_count() * 2})")
    args = parser.parse_args()

    directory_path = Path(Directory.JOB_DIR)
    files = get_json_files()
    
    print(f"Searching for text: '{args.text}' in {len(files)} files")
    matching_files = process_files_in_parallel(files, args.text, args.workers)
    
    print(f"\nFound text in {len(matching_files)} files")

    # group matching files by parent directory
    matching_files_by_parent_dir = defaultdict(list)
    for file_path in matching_files:
        parent_dir = file_path.parent
        matching_files_by_parent_dir[parent_dir].append(file_path)

    for parent_dir, files in matching_files_by_parent_dir.items():
        print(parent_dir, len(files))

    # print the results
    for parent_dir, files in matching_files_by_parent_dir.items():
        print('---'*20)
        print(f"\nFiles in directory '{parent_dir}':")
        for file_path in files:
            job = Job.from_json_file(file_path)
            data = job.request_dict['data']["messages"][0]["content"]
            end_index = data.find("Let's")
            print("***"*5)
            print(data[:end_index] if end_index != -1 else data)