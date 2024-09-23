
import sys

from copy import copy
from pydantic import BaseModel
from typing import Optional, List

from src.util.constants import Directory


class Experiment(BaseModel):

    name: str
    overrides: Optional[List[str]] = None

    @property
    def command(self):

        if self.overrides is None:
            self.overrides = []

        overrides_copy = copy(self.overrides)

        # defaulting to os.system because compose api is limited
        # and does not allow --multirun
        final_command = sys.executable + f" {str(Directory.SRC / 'main.py')} --multirun " + " ".join(overrides_copy)

        return final_command

