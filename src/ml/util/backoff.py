import os
from typing import Callable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

from src.ml.util.job_queue import Job, JobStatus


max_random_wait = int(os.environ.get("MAX_RANDOM_WAIT", 60))


class BackoffMixin:
    
    @retry(wait=wait_random_exponential(min=1, max=max_random_wait), stop=stop_after_attempt(5))
    def completion_with_backoff(self, function: Callable, *args, **kwargs):

        try:
            result = function(*args, **kwargs)
        except Exception as e:
            print(e)

        return result
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5), reraise=True)
    def _completion_with_backoff_and_queue(self, function: Callable, job_id: str, *args, save: bool = True, **kwargs) -> Job:

        """
        This function fails with expoential retries
        """

        job = Job(
            job_id=job_id,
            function=function,
            request_dict=kwargs
        )
        
        job = job.execute(save=save)
        
        if job.status == JobStatus.failed:
            raise Exception(job.error_description)
        
        return job

    def completion_with_backoff_and_queue(self, function: Callable, job_id: str, *args, save: bool = True, **kwargs) -> Job:

        """
        This function never fails, it returns a job object with an error message
        """

        job = Job(job_id=job_id, function=function, request_dict=kwargs)

        try:

            job = self._completion_with_backoff_and_queue(function, job_id, *args, save=save, **kwargs)

            return job

        except Exception as e:
            job.status = JobStatus.failed
            job.error_description = str(e)

        return job