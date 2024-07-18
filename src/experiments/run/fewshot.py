"""
This script is used to run the few-shot OSR experiments. 
You can decide to run the experiments in parallel or sequentially.

Currently, the code works only flawlessly in sequential mode. 
Current caching solutions may have io issues when in parallel mode.
"""

import multiprocessing
import typer
from typing import Optional

from src.experiments import Experiment
from src.util.constants import Directory


experiments = Experiment.create_experiments_from_yaml(
    path=Directory.CONFIG / "experiments" / "fewshot.yaml"
)


def run_experiments(num_processes: int = 1, filter_by_experiment: Optional[str] = None):

    global experiments

    if filter_by_experiment is not None:
        experiments = [experiment for experiment in experiments if experiment.name == filter_by_experiment]

    if num_processes < 1:
        raise ValueError("Number of processes must be greater than 0.")
    
    if num_processes == 1:

        for experiment in experiments:
            experiment.run()

    else:

        # Create a pool of processes
        pool = multiprocessing.Pool(processes=num_processes)

        # Call dummy_function multiple times with different processes
        for experiment in experiments:
            pool.apply_async(experiment.run)

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()


if __name__ == "__main__":
    typer.run(run_experiments)

    