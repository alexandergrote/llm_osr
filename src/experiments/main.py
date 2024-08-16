"""
This script is used to run and analyze the experiments. 
You can decide to run the experiments in parallel or sequentially.

Currently, the code works only flawlessly in sequential mode. 
Current caching solutions may have io issues when in parallel mode.
"""

import multiprocessing
import typer

from typing import Optional, List

from src.util.logger import console
from src.experiments.analysis.fewshot import FewShotAnalyser
from src.io.data_import.mlflow_engine import QueryEngine
from src.experiments import Experiment
from src.util.constants import Directory


def get_experiments() -> List[Experiment]:

    experiments = Experiment.create_experiments_from_yaml(
        path=Directory.CONFIG / "experiments" / "fewshot.yaml"
    )

    return experiments

def run_experiments(experiments: List[Experiment], num_processes: int):

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
    

def execute(filter_by_experiment: Optional[str] = None, num_processes: int = 1):

    experiments = get_experiments()

    experiments_copy = experiments.copy()

    mlflow_engine = QueryEngine()

    analyser = FewShotAnalyser()   

    if filter_by_experiment is not None:
        experiments_copy = [experiment for experiment in experiments_copy if experiment.name == filter_by_experiment]

    run_experiments(experiments=experiments, num_processes=num_processes)

    console.rule("Get aggregated results of experiment runs")

    for experiment in experiments_copy:

        console.log(f"Analyse experiment: {experiment.name}")

        data = mlflow_engine.get_results_of_single_experiment(experiment_name=experiment.name, n=100)
        
        analyser.analyse(data=data)

if __name__ == "__main__":
    typer.run(execute)

    