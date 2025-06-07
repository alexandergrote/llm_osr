import sys
import time
import os
import sqlite3

from enum import Enum
from collections import defaultdict
from datetime import datetime
from abc import abstractmethod
from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
from retry import retry  # type: ignore

from src.util.logger import get_logging_fun, console
from src.util.constants import Directory
from src.util.dynamic_import import DynamicImport
from src.util.error import RateLimitException as RateLimitError


def get_rate_limit_dir() -> Path:
    return Directory.RATE_LIMITS_DIR


def get_rate_limit_db_path() -> Path:
    """Returns the path to the rate limit database."""
    db_dir = get_rate_limit_dir()
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "rate_limits.db"


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
        
    def get_record_data(self) -> List[Dict[str, Any]]:
        """Convert records to a list of dicts for database storage."""
        return [{"timestamp": ts, "usage": usage} for ts, usage in self.records.items()]



class DatabaseManager:
    """
    Class to handle database operations for rate limits.
    """
    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.db_path = path or get_rate_limit_db_path()
        self._init_db()

    @retry(sqlite3.OperationalError, delay=1, tries=3)
    def _init_db(self):
        """Initialize database tables if they don't exist."""

        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            
            # Create rate limit configurations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS rate_limit_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manager_name TEXT NOT NULL,
                limit_name TEXT NOT NULL,
                limit_value INTEGER NOT NULL,
                increment_level TEXT NOT NULL,
                agg_level TEXT NOT NULL,
                action TEXT NOT NULL,
                waiting_time INTEGER DEFAULT 0,
                UNIQUE(manager_name, limit_name)
            )
            ''')
            
            # Create usage records table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS rate_limit_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manager_name TEXT NOT NULL,
                limit_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                usage INTEGER DEFAULT 0,
                UNIQUE(manager_name, limit_name, timestamp)
            )
            ''')
            
            conn.commit()

    @retry(sqlite3.OperationalError, delay=1, tries=3)
    def save_rate_limit_config(self, manager_name: str, limit_name: str, rate_limit: RateLimit):
        """Save rate limit configuration to database."""
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO rate_limit_configs 
            (manager_name, limit_name, limit_value, increment_level, agg_level, action, waiting_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                manager_name,
                limit_name,
                rate_limit.limit,
                rate_limit.increment_level,
                rate_limit.agg_level,
                rate_limit.action,
                rate_limit.waiting_time
            ))
            
            conn.commit()


    @retry(sqlite3.OperationalError, delay=1, tries=3)
    def save_rate_limit_records(self, manager_name: str, limit_name: str, records: List[Dict[str, Any]]):
        """Save rate limit usage records to database."""
        if not records:
            return
            
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            
            for record in records:
                cursor.execute('''
                INSERT OR REPLACE INTO rate_limit_records
                (manager_name, limit_name, timestamp, usage)
                VALUES (?, ?, ?, ?)
                ''', (
                    manager_name,
                    limit_name,
                    record["timestamp"],
                    record["usage"]
                ))
            
            conn.commit()

    @retry(sqlite3.OperationalError, delay=1, tries=3)
    def load_rate_limit_config(self, manager_name: str, limit_name: str) -> Optional[Dict[str, Any]]:
        """Load rate limit configuration from database."""
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM rate_limit_configs
            WHERE manager_name = ? AND limit_name = ?
            ''', (manager_name, limit_name))
            
            row = cursor.fetchone()
            
            if row:
                return dict(row)
        return None

    @retry(sqlite3.OperationalError, delay=1, tries=3)
    def load_rate_limit_records(self, manager_name: str, limit_name: str) -> Dict[str, int]:
        """Load rate limit usage records from database."""
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT timestamp, usage FROM rate_limit_records
            WHERE manager_name = ? AND limit_name = ?
            ''', (manager_name, limit_name))
            
            records = defaultdict(int)
            for row in cursor.fetchall():
                records[row[0]] = row[1]
            
        return records

    @retry(sqlite3.OperationalError, delay=1, tries=3)
    def get_all_rate_limit_names(self, manager_name: str) -> List[str]:
        """Get all rate limit names for a manager."""
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT DISTINCT limit_name FROM rate_limit_configs
            WHERE manager_name = ?
            ''', (manager_name,))
            
            names = [row[0] for row in cursor.fetchall()]
        return names

    @retry(sqlite3.OperationalError, delay=1, tries=3)
    def delete_rate_limit(self, manager_name: str, limit_name: str):
        """Delete a rate limit configuration and its records."""
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            DELETE FROM rate_limit_configs
            WHERE manager_name = ? AND limit_name = ?
            ''', (manager_name, limit_name))
            
            cursor.execute('''
            DELETE FROM rate_limit_records
            WHERE manager_name = ? AND limit_name = ?
            ''', (manager_name, limit_name))
            
            conn.commit()



class RateLimitManager(BaseModel):

    name: str

    rate_limits: Dict[str, RateLimit] = {}
    db_manager: DatabaseManager
    
    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True
    )

    @model_validator(mode="before")
    @classmethod
    def _init_rate_limits(cls, values: Any):

        if not isinstance(values, dict):
            raise ValueError("Values must be a dictionary")
        
        key = 'name'
        name = values.get(key, None)

        if name is None:
            raise ValueError(f"Missing required key '{key}'")
        
        key = "db_manager"
        db_dir = get_rate_limit_dir()
        db_dir.mkdir(parents=True, exist_ok=True)

        values[key] = DatabaseManager(
            path=db_dir / f"{name}.db"
        )
        
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

        rlm = RateLimitManager(
            name=name
        )

        rlm = rlm.load()

        return rlm

    @classmethod
    def create_from_config_file(cls, filename: str, init_from_db: bool = True) -> "RateLimitManager":

        """This function creates a rate limit manager from a config file"""

        filepath = Directory.CONFIG / os.path.join("rlm", filename)

        if not filepath.exists():
            raise ValueError(f"Rate limit config file {filepath} does not exist")
        
        rlm = DynamicImport.init_class_from_yaml(
            filename=str(filepath),
        )

        assert isinstance(rlm, RateLimitManager)

        if init_from_db:
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
                    self._save_rate_limit(name, rate_limit)

            except Exception as e:
                console.log(f"Error updating rate limit {name}: {e}")
    
    def load(self) -> "RateLimitManager":

        """Load all rate limits from database."""

        if self.db_manager is None:
            raise ValueError("Database manager not set.")

        limit_names = self.db_manager.get_all_rate_limit_names(self.name)
        
        for limit_name in limit_names:
            config = self.db_manager.load_rate_limit_config(self.name, limit_name)
            records = self.db_manager.load_rate_limit_records(self.name, limit_name)
            if config:
                self.rate_limits[limit_name] = RateLimit(
                    limit=config["limit_value"],
                    increment_level=config["increment_level"],
                    agg_level=config["agg_level"],
                    action=config["action"],
                    waiting_time=config["waiting_time"],
                    records=records
                )

        return self
    
    def _save_rate_limit(self, name: str, rate_limit: RateLimit):

        """Save a single rate limit to database."""
        self.db_manager.save_rate_limit_config(self.name, name, rate_limit)
        records = rate_limit.get_record_data()
        self.db_manager.save_rate_limit_records(self.name, name, records)

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
