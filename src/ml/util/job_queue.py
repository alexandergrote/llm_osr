import requests  # type: ignore
import json 

from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import List, Optional, Union, Iterator, Generator
from tqdm import tqdm
from pathlib import Path

from src.util.constants import Directory


# for easier mocking in unittest put in a function

def get_job_dir() -> Path:
    return Directory.JOB_DIR


class JobStatus(str, Enum):
    pending = "pending"
    success = "success"
    failed = "failed"


class RequestFunction(str, Enum):
    get = "get"
    post = "post"


class Job(BaseModel):

    job_id: str
    request_dict: dict
    request_function: str = RequestFunction.post
    request_output: Optional[Union[dict, List[dict], List]] = None

    error_description: Optional[str] = None

    status: JobStatus = JobStatus.pending

    @property
    def is_success(self) -> bool:
        return self.status == JobStatus.success
    
    @property
    def filepath(self) -> Path:

        job_dir = get_job_dir()

        return job_dir / f"{self.job_id}.json"
    
    @property
    def exists(self) -> bool:
        return Path(self.filepath).exists()

    def save(self):

        with open(self.filepath, "w") as f:
            json.dump(self.model_dump(), f)
    
    @classmethod
    def from_json_file(cls, file_path) -> 'Job':

        with open(file_path, "r") as f:
            return cls(**json.load(f))
        
    def execute(self, save: bool = False, use_cache: bool = True) -> "Job":

        # check if cached
        if self.filepath.exists() and use_cache:

            job = Job.from_json_file(self.filepath)

            if job.is_success:
                return job
  
        fun = getattr(requests, self.request_function)

        response = fun(**self.request_dict)

        status_code = response.status_code

        if status_code == 200:

            assert hasattr(response, "json"), "Response object does not have a json attribute"

            self.status = JobStatus.success
            self.request_output = response.json()
            self.error_description = None
            
        else:

            self.status = JobStatus.failed
            if hasattr(response, "text"):
                self.error_description = response.text

        if save:
            self.save()

        return self


class JobQueue(BaseModel):

    jobs: Union[Iterator[Job], List] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _job_iterator(cls, job_files: Generator) -> Iterator[Job]:
        for file_path in job_files:
            with open(file_path) as f:
                yield Job(**json.load(f))

    @classmethod
    def from_json_files(cls) -> 'JobQueue':

        job_dir = get_job_dir()

        job_files = job_dir.glob("*.json")

        return cls(jobs=cls._job_iterator(job_files))

    def run_failed_jobs(self):

        # keep track of failed jobs
        failed_jobs = []

        progress_bar = tqdm(self.jobs, desc="Job Progress")

        for job in progress_bar:

            progress_bar.set_description(f"Job {job.job_id}")

            if job.is_success:
                continue

            job = job.execute()
            
            if job.status == JobStatus.failed:
                tqdm.write(f"Job {job.job_id} failed")
                failed_jobs.append(job)

        print(f"Number of jobs failed: {len(failed_jobs)}")


if __name__ == "__main__":   

    job_queue = JobQueue.from_json_files()

    job_queue.run_failed_jobs()

    