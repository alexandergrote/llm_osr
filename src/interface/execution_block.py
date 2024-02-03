from typing import Any, Union, Dict

from interface.base import BaseExecutionBlock
from pydantic import BaseModel, validator
from util.dynamic_import import DynamicImport
from util.logging import console


class ExecutionBlock(BaseExecutionBlock, BaseModel):

    name: str
    block: Union[Dict[str, dict], Any]

    @validator('block')
    def _init_component(cls, value):
        obj = DynamicImport.init_class_from_dict(dictionary=value)
        return obj
    
    def execute(self, **kwargs) -> dict:
        console.log(f"Executing: {self.name}")

        if not hasattr(self.block, 'execute'):
            raise AttributeError(f"Block {self.name} does not have an execute method.")

        self.block.execute(**kwargs)
        return {}