import pickle
import os
import inspect
import json
from abc import ABC, abstractmethod
from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Any, Optional

from src.util.constants import Directory
from src.util.hashing import Hash


def get_executing_script_filepath():
    frame = inspect.stack()[3]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    filename_abs_path = os.path.abspath(filename)

    return filename_abs_path


class CacheHandler(ABC):

    @abstractmethod
    def read(self) -> Optional[Any]:
        pass

    @abstractmethod
    def write(self, obj: Any):
        pass


class PickleCacheHandler(BaseModel, CacheHandler):

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


class JsonCache(BaseModel, CacheHandler):
    
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

        with open(self.filepath, 'r') as file:
            json_data = json.load(file)

        
        return json_data
    
    def write(self, obj: Any):

        if not self.filepath.exists():
            self.filepath.parent.mkdir(exist_ok=True, parents=True)

        with open(self.filepath, 'w') as file:
            json.dump(obj, file)


def environ_pickle_cache():

    def decorator(fn):

        def wrapped(*args, **kwargs):

            # create filename from function call
            full_filename = os.path.join(os.getcwd(), fn.__code__.co_filename)

            repo_path = Path(full_filename).relative_to(Directory.ROOT)

            # transform slashes to dots
            # makes them independent of OS
            repo_path = repo_path.as_posix().replace('/', '.')
            
            filename_components = [repo_path, fn.__qualname__]
            current_filename = '.'.join(filename_components)
                      
            # create hash for shorter filenames
            hash_object = Hash.hash_recursive(
                *args, **kwargs
            )

            cachedir = Directory.CACHING_DIR / current_filename
            cachedir.mkdir(exist_ok=True, parents=True)

            filename_pickle = Path(cachedir) / f"{hash_object}.pickle"

            cache_handler = PickleCacheHandler(
                filepath=filename_pickle
            )

            # if not cache specified, execute function
            if current_filename not in os.environ:
                res = fn(*args, **kwargs)
                cache_handler.write(obj=res)
                return res

            res = cache_handler.read()
            
            if res is not None:
                return res

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            cache_handler.write(obj=res)

            return res

        return wrapped

    return decorator
