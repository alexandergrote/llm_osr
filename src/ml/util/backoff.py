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
    def _completion_with_backoff_and_queue(self, job: Job, save: bool = True, use_cache: bool = True, **kwargs) -> Job:

        """
        This function fails with expoential retries
        """

        job = job.execute(save=save, use_cache=use_cache)
        
        if job.status == JobStatus.failed:
            raise Exception(job.error_description)
        
        return job

    def completion_with_backoff_and_queue(self, function: Callable, job_id: str, rest_model_name: str, save: bool = True, use_cache: bool = True, **kwargs) -> Job:

        """
        This function never fails, it returns a job object with an error message
        """

        job = Job(
            job_id=job_id,
            rest_model_name=rest_model_name, 
            function=function, 
            request_dict=kwargs,
        )

        try:

            job = self._completion_with_backoff_and_queue(job=job, save=save, use_cache=use_cache, **kwargs)

            return job

        except Exception as e:
            job.status = JobStatus.failed
            job.error_description = str(e)

        return job