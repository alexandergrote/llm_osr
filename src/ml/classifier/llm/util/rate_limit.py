import sys
import json
import time
import os

from enum import Enum
from collections import defaultdict
from datetime import datetime
from abc import abstractmethod
from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from typing import Dict, Any
from pathlib import Path

from src.util.logger import get_logging_fun, console
from src.util.constants import Directory
from src.util.dynamic_import import DynamicImport
from src.util.error import RateLimitException as RateLimitError


def get_rate_limit_dir() -> Path:
    return Directory.RATE_LIMITS_DIR


class BaseRateLimit(BaseModel):

    @abstractmethod
    def check(self) -> bool:
        raise NotImplementedError("This function needs to be implemented")


class IncrementLevel(str, Enum):

    TOKEN = "token"
    FREQUENCY = "frequency"


class Action(str, Enum):

    EXIT = "exit"
    WAIT = "wait"
    IGNORE = "ignore"
    RAISE = "raise"


class RateLimit(BaseRateLimit):

    limit: int
    increment_level: IncrementLevel 
    agg_level: str
    action: Action = Action.IGNORE
    waiting_time: int = 0
    records: Dict[str, int] = defaultdict(int)

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True
    )

    @field_validator("records")
    @classmethod
    def _cast_records(cls, v):
        if not isinstance(v, defaultdict):
            return defaultdict(int, v)
        return v

    def _get_timestamp(self) -> str:
        return datetime.now().strftime(self.agg_level)

    def check(self) -> bool:

        timestamp = self._get_timestamp()

        return self.records[timestamp] <= self.limit

    def update(self, usage_request: int):

        # get timestamp in accordance with aggregation level
        timestamp = self._get_timestamp()

        # reset if old timestamp is present
        if (len(self.records) == 1) and (timestamp not in self.records):
            self.records = defaultdict(int)

        if self.increment_level == IncrementLevel.TOKEN:
            self.records[timestamp] += usage_request
        elif self.increment_level == IncrementLevel.FREQUENCY:
            self.records[timestamp] += 1
        else:
            raise ValueError(f"Invalid increment level {self.increment_level}")
        
    def save(self, path: str):

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def load(cls, path: str) -> "RateLimit":

        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)


class RateLimitManager(BaseModel):

    name: str

    rate_limits: Dict[str, RateLimit] = {}
    _filetype: str = "json"
    
    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True
    )

    @model_validator(mode="before")
    @classmethod
    def _init_rate_limits(cls, values: Any):

        if not isinstance(values, dict):
            raise ValueError("Values must be a dictionary")
        
        key = "rate_limits"
        rate_limits = values.get(key, {})

        if len(rate_limits) == 0:
            return values
        
        for name, rate_limit in rate_limits.items():

            is_rate_limit = isinstance(rate_limit, RateLimit)
            is_dict = isinstance(rate_limit, dict)
            if (not is_rate_limit) and (is_dict):
                rate_limits[name] = RateLimit(**rate_limit)

        values[key] = rate_limits

        return values

    @classmethod
    def create_from_name(cls, name: str) -> "RateLimitManager":

        """This function creates a rate limit manager from rate limit files in json format"""

        full_filename = get_rate_limit_dir() / name

        if not full_filename.exists():
            raise ValueError(f"Rate limit file {full_filename} does not exist")
        
        rlm = RateLimitManager(
            name=name
        )

        rlm = rlm.load()

        return rlm

    @classmethod
    def create_from_config_file(cls, filename: str, init_from_disk: bool = True) -> "RateLimitManager":

        """This function creates a rate limit manager from a config file"""

        filepath = Directory.CONFIG / os.path.join("rlm", filename)

        if not filepath.exists():
            raise ValueError(f"Rate limit config file {filepath} does not exist")
        
        rlm = DynamicImport.init_class_from_yaml(
            filename=str(filepath),
        )

        assert isinstance(rlm, RateLimitManager)

        if init_from_disk:
            rlm.load()
            rlm.save()

        return rlm


    @property
    def directory(self) -> Path:

        dir_path = get_rate_limit_dir() / self.name
        dir_path.mkdir(parents=True, exist_ok=True)

        return dir_path

    def update(self, tokens: int, save: bool = True, tokens_only: bool = False):

        for name, rate_limit in self.rate_limits.items():
            
            try:

                is_freq_request = rate_limit.increment_level == IncrementLevel.FREQUENCY

                if tokens_only and is_freq_request:
                    continue

                rate_limit.update(tokens)

                if save:
                    rate_limit.save(
                        path=str(self.directory / f"{name}.{self._filetype}")
                    )

            except Exception as e:
                console.log(f"Error updating rate limit {name}: {e}")
    
    def load(self) -> "RateLimitManager":

        json_files = self.directory.glob(f"*.{self._filetype}")

        for json_file in json_files:

            rate_limit_name = json_file.stem
            rate_limit = RateLimit.load(str(json_file))

            self.rate_limits[rate_limit_name] = rate_limit

        return self
    
    def _save_rate_limit(self, name: str, rate_limit: RateLimit):

        filename = str(self.directory / f"{name}.{self._filetype}")

        rate_limit.save(filename)

    def save(self):

        for name, rate_limit in self.rate_limits.items():
            self._save_rate_limit(name, rate_limit)
    
    def check_execution(self, num_request_tokens: int, save: bool = True, **kwargs):

        combinations = [(name, rate_limit) for name, rate_limit in self.rate_limits.items()]

        logging_fun = get_logging_fun(**kwargs)

        for (name, rate_limit) in combinations:

            rate_limit.update(
                usage_request=num_request_tokens
            )

            if save:
                self._save_rate_limit(name, rate_limit)
            
            if not rate_limit.check():
                
                logging_fun(f"Rate limit exceeded for {name}")

                if rate_limit.action == Action.EXIT:
                    logging_fun("Exiting...")
                    sys.exit(1)
                
                if rate_limit.action == Action.WAIT:
                    logging_fun(f"Waiting {rate_limit.waiting_time} seconds...")
                    time.sleep(rate_limit.waiting_time)

                if rate_limit.action == Action.IGNORE:
                    logging_fun("No action taken")

                if rate_limit.action == Action.RAISE:
                    logging_fun("Raising RateLimitError")
                    raise RateLimitError(f"Rate limit exceeded for {name}")
