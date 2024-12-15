import json

from pydantic import BaseModel
from rich.console import Console
from typing import Callable

from src.util.constants import Directory


console = Console()

def get_log_dir():
    return Directory.ERROR_LOG_DIR

def get_logging_fun(*args, **kwargs) -> Callable:

    logging_fun = console.log
    if 'pbar' in kwargs:
        pbar = kwargs['pbar']

        if hasattr(pbar, 'write'):
            logging_fun = pbar.write

    # check if callable
    assert callable(logging_fun), "Invalid logging_function argument"

    return logging_fun


def log_error(filename: str, json_dict: dict, **kwargs):

    full_path = get_log_dir() / filename

    logging_fun = get_logging_fun(**kwargs)

    logging_fun(f"Logging error to {full_path}")

    if isinstance(json_dict, str):
        json_dict = json.loads(json_dict)

    with open(full_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
        
def log_pydantic_error(filename: str, error: BaseModel, **kwargs):
    log_error(filename, error.model_dump_json(), **kwargs)

def delete_error_log(filename: str):
    
    full_path = get_log_dir() / filename

    if full_path.exists():
        full_path.unlink()