import pickle
import os
import inspect
from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Any, Optional

from src.util.constants import Directory


def get_executing_script_filepath():
    frame = inspect.stack()[3]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    filename_abs_path = os.path.abspath(filename)

    return filename_abs_path


class PickleCacheHandler(BaseModel):

    filepath: Path

    @field_validator("filepath")
    def _set_directory(cls, v):

        filepath = Path(
            get_executing_script_filepath()
        )

        # get the shared path with the caching dir
        shared_path = filepath.relative_to(Directory.ROOT) / filepath.stem

        return Directory.CACHING_DIR / shared_path /  v

    def read(self) -> Optional[Any]: 

        if not self.filepath.exists():
            return None
        
        with open(self.filepath, 'rb') as cachehandle:
            return pickle.load(cachehandle)
 
    def write(self, obj: Any):

        if not self.filepath.exists():
            self.filepath.parent.mkdir(exist_ok=True, parents=True)

        # write to cache file
        with open(self.filepath, 'wb') as cachehandle:
            pickle.dump(obj, cachehandle)