"""
This is a utility script to see in which json files a supplied text appears
"""

from pathlib import Path
from tqdm import tqdm
from typing import List


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


def search_text_in_json_file(file: Path, text: str) -> bool:

    """
    Search for the provided text in a JSON file.
    """
    job = get_json_content(file)

    if text in str(job.request_dict):
        return True
    return False
    



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Inspect jobs for a given text")
    parser.add_argument("text", type=str, help="The text to search for")
    args = parser.parse_args()

    directory_path = Path(Directory.JOB_DIR)
    
    files = get_json_files()

    args.text = "What are the currency exchange fees?"

    for file in tqdm(files):
        if search_text_in_json_file(file, args.text):
            print(f"Text found in {file}")
