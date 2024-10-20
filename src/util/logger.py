import json

from rich.console import Console
from src.util.constants import Directory

console = Console()

def log_error(filename: str, json_dict: dict):

    full_path = Directory.ERROR_LOG_DIR / filename

    with open(full_path, 'w') as f:
        json.dump(json_dict, f, indent=4)