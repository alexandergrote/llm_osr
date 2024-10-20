import hydra
from enum import Enum
from pathlib import Path


class Directory:

    ROOT = Path(__file__).absolute().parent.parent.parent
    CONFIG = ROOT / "config"
    SRC = ROOT / "src"
    MODEL = ROOT / "model"
    INPUT_DIR = ROOT / 'data'
    OUTPUT_DIR = ROOT / "outputs"
    CACHING_DIR = ROOT / "cache"
    JOB_DIR = ROOT / "jobs"
    ERROR_LOG_DIR = ROOT / "logs"


paths = [
    getattr(Directory, attribute)
    for attribute in dir(Directory)
    if not attribute.startswith("_")
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)


def get_hydra_output_dir() -> Path:

    try:
        config = hydra.core.hydra_config.HydraConfig
        return Path(config.get().runtime.output_dir)
    
    except ValueError:
        return Directory.OUTPUT_DIR


class File:
    CONFIG = Directory.CONFIG / "config.yaml"
    MAIN = Directory.SRC / "main.py"

class DatasetColumn:
    TEXT = "text"
    LABEL = "label"
    FEATURES = "features"


class DictConfigNames:
    RANDOM_SEED = 'random_seed'

class BaseEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class YamlField(BaseEnum):
    CLASS_NAME = "class"
    PARAMS = "params"


class EnvMode(BaseEnum):
    DEV = 'dev'
    PROD = 'prod'
    DRYRUN = 'dryrun'


class LLMModels(BaseEnum):
    OAI_GPT4 = 'gpt4'
    OAI_GPT3 = 'gpt3'
    OAI_GPT2 = 'gpt2'
    LLAMA_3B_Local = 'llama'
    LLAMA_3_8B_Remote = 'llama_remote'
    LLAMA_3_8B_Remote_HF = 'llama_remote_hf'
    LLAMA_3_70B_Remote_HF = 'llama_remote_hf_70b'


class UnknownClassLabel(BaseEnum):
    UNKNOWN_STR = 'unknown'
    UNKNOWN_NUM = -1

class ErrorValues(BaseEnum):
    PARSING_STR = 'parse_error'
    PARSING_NUM = -2
    LOGPROB_STR = 'logprob_error'
    LOGPROB_NUM = -3
    

RESTAPI_URLS = {
    LLMModels.LLAMA_3_8B_Remote_HF: 'https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8b-Instruct',
    LLMModels.LLAMA_3_70B_Remote_HF: 'https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70b-Instruct',
    LLMModels.OAI_GPT3: 'https://api.openai.com/v1/chat/completions',
    LLMModels.OAI_GPT4: 'https://api.openai.com/v1/chat/completions',
}