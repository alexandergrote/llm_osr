

from pydantic import BaseModel
from typing import List, Optional

from src.experiments.util.types import Experiment


class ExperimentFactory(BaseModel):

    @classmethod
    def create_fewshot_experiments(cls, models: Optional[List[str]] = None, datasets: Optional[List[str]] = None, unknown_classes: Optional[List[float]] = None, random_seeds: Optional[List[int]] = None) -> List[Experiment]:
        
        experiments = []

        if models is None:
            models = ["naive", "simpleshot"]

        if datasets is None:
            datasets = ['banking', 'clinc', 'hwu']

        if unknown_classes is None:
            unknown_classes = [0, 0.2, 0.4, 0.6]

        if random_seeds is None:
            random_seeds = [0, 1, 2, 3, 4]

        for dataset in datasets:

            for model in models:

                for unknown_class in unknown_classes:

                    exp_name = f'fewshot__data__{dataset}__model__{model}__unknown_classes__{unknown_class}'

                    overrides = [
                        f'io__import={dataset}',
                        f'ml__classifier={model}',
                        'ml__datasplit=fewshot_osr',
                        f'ml__datasplit.params.percentage_unknown_classes={unknown_class}',
                        f'random_seed={",".join(map(str, random_seeds))}',
                        'ml__evaluation=osr',
                        'io__export=mlflow',
                        f'io__export.params.experiment_name={exp_name}'
                    ]

                    if model == 'simpleshot':
                        overrides.append('ml__preprocessing=random_embedding')

                    experiments.append(Experiment(name=exp_name, overrides=overrides))

        return experiments
