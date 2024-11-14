import json

from pydantic import BaseModel
from rich.console import Console
from src.util.constants import Directory


console = Console()

def get_log_dir():
    return Directory.ERROR_LOG_DIR


def log_error(filename: str, json_dict: dict):

    full_path = get_log_dir() / filename

    console.log(f"Logging error to {full_path}")

    with open(full_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
        
def log_pydantic_error(filename: str, error: BaseModel):
    log_error(filename, error.model_dump_json())

def delete_error_log(filename: str):
    
    full_path = get_log_dir() / filename

    if full_path.exists():
        full_path.unlink()