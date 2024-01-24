from pydantic import BaseModel, validator
from typing import Dict, Any, Union

from interface.base import BaseExecutionBlock
from util.logging import console


class ExecutionBlock(BaseExecutionBlock, BaseModel):

    name: str
    block: Union[Dict[str, dict], Any]

    @validator('block')
    def _init_component(cls, value):
        return {}
    
    def execute(self, *args, **kwargs) -> dict:
        console.log(f"Executing: {self.name}")
        return {}