from pydantic import BaseModel, validator
from typing import Dict, Any, Union

from interface.base import BaseExecutionBlock
from util.logging import console
from util.dynamic_import import DynamicImport


class ExecutionBlock(BaseExecutionBlock, BaseModel):

    name: str
    block: Union[Dict[str, dict], Any]

    @validator('block')
    def _init_component(cls, value):
        obj = DynamicImport.init_class_from_dict(dictionary=value)
        return obj
    
    def execute(self, **kwargs) -> dict:
        console.log(f"Executing: {self.name}")
        self.block.execute(**kwargs)
        return {}