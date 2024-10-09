import requests  # type: ignore
import json 

from pydantic import BaseModel
from enum import Enum
from typing import List, Optional, Union
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
        
    def execute(self) -> "Job":

        if self.filepath.exists():
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

        self.save()

        return self


class JobQueue(BaseModel):

    jobs: List[Job] = []

    @classmethod
    def from_json_files(cls) -> 'JobQueue':

        job_dir = get_job_dir()

        job_files = job_dir.glob("*.json")

        jobs = [Job(**json.load(open(file_path))) for file_path in job_files]

        return cls(jobs=jobs)


    def add_job(self, job: Job):
        self.jobs.append(job)

    def delete_job(self, job: Job):
        self.jobs.remove(job)

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

    