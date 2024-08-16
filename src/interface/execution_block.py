from typing import Any, Union, Dict

from src.interface.base import BaseExecutionBlock
from pydantic import BaseModel, validator
from src.util.dynamic_import import DynamicImport
from src.util.logger import console


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

        try:
            result = self.block.execute(**kwargs)
        except Exception as e:
            console.log(f"Error in block {self.name}")
            console.log(e)
            raise e
        
        return result