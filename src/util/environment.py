import os
from pydantic import BaseModel
from pydantic.config import ConfigDict
from dotenv import load_dotenv
from typing import Optional

from src.util.constants import EnvMode


load_dotenv()


class PydanticEnvironment(BaseModel):

    _mode: EnvMode
    _hf_token: Optional[str]
    _openai_key: Optional[str]

    model_config = ConfigDict(extra = 'forbid')

    @property
    def hf_token(self) -> str:

        if self._hf_token is None:
            raise ValueError("hf token not set")

        return self._hf_token
    
    @hf_token.setter
    def hf_token(self, value):
        self._hf_token = value

    @property
    def openai_key(self) -> str:

        if self._openai_key is None:
            raise ValueError("hf token not set")

        return self._openai_key
    
    @openai_key.setter
    def openai_key(self, value: str):
        self._openai_key = value

    @property
    def mode(self) -> str:
        return EnvMode(self._mode).value
    
    @mode.setter
    def mode(self, value: EnvMode):        
        assert isinstance(value, EnvMode)
        self._mode = value

    def is_dev_mode(self) -> bool:
        return self._mode == EnvMode.DEV
    
    def is_dryrun_mode(self) -> bool:
        return self._mode == EnvMode.DRYRUN
    
    @classmethod
    def from_environment(cls) -> 'PydanticEnvironment':

        # Create an instance and set attributes
        instance = cls()

        instance.mode = EnvMode(os.environ.get('MODE', EnvMode.DEV.value))  # type: ignore
        instance.openai_key = os.environ.get("OPENAI_API_KEY")  # type: ignore
        instance.hf_token = os.environ.get('HF')  # type: ignore

        return instance


if __name__ == '__main__':

    os.environ['MODE'] = EnvMode.DRYRUN.value

    env = PydanticEnvironment.from_environment()
    print(env)
    print(env.mode)