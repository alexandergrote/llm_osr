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
    _groq_key: Optional[str]
    _sambanova_key: Optional[str]
    _openrouter_key: Optional[str]

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
            raise ValueError("openai token not set")

        return self._openai_key
    
    @openai_key.setter
    def openai_key(self, value: str):
        self._openai_key = value

    @property
    def groq_key(self) -> str:

        if self._groq_key is None:
            raise ValueError("groq token not set")
        
        return self._groq_key
    
    @groq_key.setter
    def groq_key(self, value: str):
        self._groq_key = value

    @property
    def sambanova_key(self) -> str:

        if self._sambanova_key is None:
            raise ValueError("groq token not set")
        
        return self._sambanova_key
    
    @sambanova_key.setter
    def sambanova_key(self, value: str):
        self._sambanova_key = value

    @property
    def openrouter_key(self) -> str:

        if self._openrouter_key is None:
            raise ValueError("groq token not set")
        
        return self._openrouter_key
    
    @openrouter_key.setter
    def openrouter_key(self, value: str):
        self._openrouter_key = value

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
        instance.groq_key = os.environ.get('GROQ')  # type: ignore
        instance.sambanova_key = os.environ.get('SAMBANOVA')  # type: ignore
        instance.openrouter_key = os.environ.get('OPENROUTER')  # type: ignore

        return instance


if __name__ == '__main__':

    os.environ['MODE'] = EnvMode.DRYRUN.value

    env = PydanticEnvironment.from_environment()
    print(env)
    print(env.mode)