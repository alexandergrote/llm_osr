from pydantic import BaseModel
from typing import List

class Error(BaseModel):
    error_message: str

class LLMError(Error):
    prompt: str
    response: str

class LogProbError(Error):
    logprobs: List[str]

class APIException(Exception):
    pass

class RateLimitException(Exception):
    pass