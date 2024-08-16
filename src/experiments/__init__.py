
import os
import sys
import yaml

from pathlib import Path
from copy import copy
from pydantic import BaseModel
from typing import Optional, List

from src.util.constants import Directory


class Experiment(BaseModel):

    name: str
    overrides: Optional[List[str]] = None

    def run(self):

        if self.overrides is None:
            self.overrides = []

        overrides_copy = copy(self.overrides)

        overrides_copy.append(f"name={self.name}")
        overrides_copy.append("io__export=mlflow")
        overrides_copy.append(f"io__export.params.experiment_name={self.name}")

        # defaulting to os.system because compose api is limited
        # and does not allow --multirun
        final_command = sys.executable + f" {str(Directory.SRC / 'main.py')} --multirun " + " ".join(overrides_copy)
        return_code = os.system(final_command)

        if return_code != 0:
            raise RuntimeError(f"Experiment {self.name} failed with return code {return_code}")

    @classmethod
    def create_experiments_from_yaml(cls, path: Path) -> List["Experiment"]:

        with open(path, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        return [cls(**el) for el in data]