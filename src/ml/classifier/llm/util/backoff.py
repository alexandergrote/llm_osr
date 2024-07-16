from typing import Callable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class BackoffMixin:
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1))
    def completion_with_backoff(self, function: Callable, *args, **kwargs):
        return function(*args, **kwargs)