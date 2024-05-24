import os
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

from src.util.constants import EnvMode


load_dotenv()


class PydanticEnvironment(BaseModel):

    mode: EnvMode = EnvMode.DEV
    hf_token: Optional[str] = os.environ.get('HF')
    openai_key: Optional[str] = os.environ.get("OPENAI_API_KEY")

    class Config:
        extra = 'forbid'

    @staticmethod
    def set_environment_variables(data: dict):

        for key, value in data.items():

            # check if supplied argument is an enum and add value to environment
            if isinstance(value, EnvMode):
                value = value.value

            os.environ[key] = str(value)

    @classmethod
    def create_from_environment(cls):
        
        # create dictionary with all attributes of the class as keys and its environment variables as values
        data = {key: os.environ[key] for key in cls.model_fields.keys()}

        # check if attribute is enum and convert it to enum
        for key, value in data.items():
            if cls.__annotations__[key] == EnvMode:
                data[key] = EnvMode(value)

        return cls(**data)
    

if __name__ == '__main__':

    PydanticEnvironment.set_environment_variables(data={
        'model': EnvMode.DEV,
        'hf_token': 'test'
    })
    env = PydanticEnvironment.create_from_environment()

    print(env.model_dump())