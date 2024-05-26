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


paths = [
    getattr(Directory, attribute)
    for attribute in dir(Directory)
    if not attribute.startswith("_")
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)


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


class LLMModels(BaseEnum):
    OAI_GPT4 = 'gpt4'
    OAI_GPT3 = 'gpt3'
    OAI_GPT2 = 'gpt2'
    LLAMA_3B = 'llama'
