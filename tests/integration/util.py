import warnings
import os

from hydra import compose, initialize
from omegaconf import DictConfig
from typing import List


def get_hydra_config(overrides: List[str]) -> DictConfig:

    # get hydra config
    with warnings.catch_warnings():
        
        warnings.simplefilter(action='ignore', category=UserWarning)

        config_path = os.path.join('..', '..', 'config')

        with initialize(config_path=config_path):
            cfg = compose(config_name="config", overrides=overrides)

    return cfg