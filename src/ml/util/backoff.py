from typing import Callable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


from src.ml.util.job_queue import Job

class BackoffMixin:
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def completion_with_backoff(self, function: Callable, *args, **kwargs):
        return function(*args, **kwargs)
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def completion_with_backoff_and_queue(self, function: Callable, job_id: str, *args, **kwargs) -> Job:

        job = Job(
            job_id=job_id,
            function=function,
            request_dict=kwargs
        )

        job = job.execute()

        return job
