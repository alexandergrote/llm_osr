from pathlib import Path
from enum import Enum


class Directory:

    ROOT = Path(__file__).absolute().parent.parent.parent
    CONFIG = ROOT / "config"
    SRC = ROOT / "src"
    MODEL = ROOT / "model"
    INPUT_DIR = ROOT / 'data'
    OUTPUT_DIR = ROOT / "outputs"


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