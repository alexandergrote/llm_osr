

from pydantic import BaseModel
from typing import List, Optional

from src.ml.classifier.llm.util.prompt import PromptDataScenario, PromptOODScenario, PromptScenarioName
from src.experiments.util.types import Experiment


BENCHMARK_MODELS = ["hyper_simpleshot", "hyper_fastfit", "hyper_contrastnet"]
BENCHMARK_RANDOM_MODELS = ['naive']

LLM_ONE_STAGE_MODELS = ["one_stage_llama_8", "one_stage_llama_70", "one_stage_gemma3_27", "one_stage_phi4_14"]
LLM_PROMPT_FIRST_RANDOM_SECOND = ["mixed_llama_8", "mixed_llama_70", "mixed_gemma3_27", "mixed_phi4_14"]
LLM_RANDOM_MODELS = ['random_llm']

DATASETS = ['banking', 'clinc', 'hwu']
UNKNOWN_CLASSES = [0, 0.2, 0.4, 0.6]
RANDOM_SEEDS = [0, 1, 2, 3, 4]

N_SUBSET_TEST = 3000


def get_default_overrides(dataset: str, model: str, unknown_class: float, random_seeds: List[int], exp_name: str) -> List[str]:

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

    return overrides


class ExperimentFactory(BaseModel):

    @classmethod
    def create_benchmark_experiments(cls, models: Optional[List[str]] = None, datasets: Optional[List[str]] = None, unknown_classes: Optional[List[float]] = None, random_seeds: Optional[List[int]] = None) -> List[Experiment]:
        
        experiments = []

        if models is None:
            models = BENCHMARK_MODELS

        if datasets is None:
            datasets = DATASETS

        if unknown_classes is None:
            unknown_classes = UNKNOWN_CLASSES

        if random_seeds is None:
            random_seeds = [0, 1, 2, 3] #RANDOM_SEEDS

        for dataset in datasets:

            for model in models:

                for unknown_class in unknown_classes:

                    exp_name = f'fewshot__benchmark__data__{dataset}__model__{model}__unknown_classes__{unknown_class}'

                    overrides = get_default_overrides(
                        dataset=dataset,
                        model=model,
                        unknown_class=unknown_class,
                        random_seeds=random_seeds,
                        exp_name=exp_name
                    )
                    
                    # all the benchmark models need
                    # 1) encoding of text features
                    # 2) hyper parameter tuning
                    if model in ["hyper_simpleshot"]:
                        overrides.append('ml__preprocessing=rest_embedding')

                    experiments.append(Experiment(name=exp_name, overrides=overrides))

        return experiments

    @classmethod
    def create_llm_fewshot_experiments(cls, models: Optional[List[str]] = None, datasets: Optional[List[str]] = None, unknown_classes: Optional[List[float]] = None, random_seeds: Optional[List[int]] = None) -> List[Experiment]:

        experiments = []

        if models is None:
            models = LLM_ONE_STAGE_MODELS

        if datasets is None:
            datasets = DATASETS

        if unknown_classes is None:
            unknown_classes = UNKNOWN_CLASSES

        if random_seeds is None:
            random_seeds = RANDOM_SEEDS

        for dataset in datasets:

            for model in models:

                for unknown_class in unknown_classes:

                    scenario_name = PromptScenarioName.create_from_enums(
                        ood_scenario=PromptOODScenario.IMPLICIT,
                        data_scenario=PromptDataScenario.ZEROSHOT
                    )

                    if model.startswith("mixed"):
                        raise ValueError("Mixed models are not supported for few-shot experiments.")

                    exp_name = f'fewshot__llm__data__{dataset}__model__{model}__scenario__{scenario_name.value}__unknown_classes__{unknown_class}'

                    overrides = get_default_overrides(
                        dataset=dataset,
                        model=model,
                        unknown_class=unknown_class,
                        random_seeds=random_seeds,
                        exp_name=exp_name
                    )

                    if ('one_stage' in model):
                        overrides.append('ml__classifier.params.shuffle_free_llms=true')

                    overrides += [
                        f'ml__classifier.params.unknown_detection_scenario={scenario_name.value}',
                        f'ml__datasplit.params.subset_test={N_SUBSET_TEST}',
                    ]

                    experiments.append(Experiment(name=exp_name, overrides=overrides))

        return experiments

    @classmethod
    def create_all_fewshot_experiments(cls) -> List[Experiment]:

        benchmark_experiments = cls.create_benchmark_experiments()
        llm_experiments = cls.create_llm_fewshot_experiments()

        return benchmark_experiments + llm_experiments
        

    @classmethod
    def create_fewshot_error_experiments(cls, models: Optional[List[str]] = None, datasets: Optional[List[str]] = None, unknown_classes: Optional[List[float]] = None, random_seeds: Optional[List[int]] = None) -> List[Experiment]:

        if unknown_classes is None:
            unknown_classes = [0.2] #UNKNOWN_CLASSES

        if random_seeds is None:
            random_seeds = [0]

        if datasets is None:
            datasets = DATASETS

        experiments = cls.create_benchmark_experiments(datasets=datasets, unknown_classes=unknown_classes, random_seeds=random_seeds)
        experiments.extend(cls.create_llm_fewshot_experiments(datasets=datasets, unknown_classes=unknown_classes, random_seeds=random_seeds))

        return experiments


    @classmethod
    def create_llm_ood_experiments(cls, models: Optional[List[str]] = None, datasets: Optional[List[str]] = None, unknown_classes: Optional[List[float]] = None, random_seeds: Optional[List[int]] = None) -> List[Experiment]:

        if models is None:
            models = LLM_PROMPT_FIRST_RANDOM_SECOND + LLM_ONE_STAGE_MODELS

        if datasets is None:
            datasets = DATASETS

        if unknown_classes is None:
            unknown_classes = [0.2] 

        if random_seeds is None:
            random_seeds = [0] 

        experiments = cls.create_llm_fewshot_experiments(
            datasets=datasets,
            unknown_classes=unknown_classes, 
            random_seeds=random_seeds, 
            models=LLM_ONE_STAGE_MODELS,
        )

        for dataset in datasets:

            for model in models:

                for unknown_class in unknown_classes:

                    for data_scenario in PromptDataScenario.list():

                        if model.startswith('one_stage'):

                            ood_scenario = PromptOODScenario.IMPLICIT

                        elif model.startswith('mixed'):

                            ood_scenario = PromptOODScenario.EXPLICIT

                        else:

                            raise ValueError(f"Unknown model type: {model}")

                        # given that the existing fewshot llms are all implicit and zeroshot
                        # we can skip them
                        if (ood_scenario == PromptOODScenario.IMPLICIT) and (PromptDataScenario.ZEROSHOT == data_scenario):
                            continue

                        scenario_name = PromptScenarioName.create_from_enums(
                            ood_scenario=ood_scenario,
                            data_scenario=data_scenario
                        )

                        exp_name = f'ood__llm__data__{dataset}__model__{model}__scenario__{scenario_name.value}__unknown_classes__{unknown_class}'

                        overrides = get_default_overrides(
                            dataset=dataset,
                            model=model,
                            unknown_class=unknown_class,
                            random_seeds=random_seeds,
                            exp_name=exp_name
                        )

                        if ('one_stage' in model):
                            overrides.append('ml__classifier.params.shuffle_free_llms=true')

                        overrides += [
                            f'ml__classifier.params.unknown_detection_scenario={scenario_name.value}',
                            f'ml__datasplit.params.subset_test={N_SUBSET_TEST}',
                        ]

                        experiments.append(Experiment(name=exp_name, overrides=overrides))

        return experiments

    @classmethod
    def create_all_llm_experiments(cls) -> List[Experiment]:

        experiments = cls.create_llm_fewshot_experiments()
        experiments += cls.create_llm_ood_experiments()

        # filter duplicates by experiment name
        seen_names = set()
        unique_experiments = []

        for exp in experiments:
            if exp.name not in seen_names:
                seen_names.add(exp.name)
                unique_experiments.append(exp)
        
        return unique_experiments

if __name__ == "__main__":

    import re
    from collections import defaultdict
    
    all_experiments = ExperimentFactory.create_all_llm_experiments()

    # sort by name
    all_experiments.sort(key=lambda x: x.name)
    for exp in all_experiments:
        if "llama_70" in exp.name:
            print(exp.name)

    print("Counting models and scenarios...")
    print("Mixed - explicit")

    # Dictionary to store counts: {model: {scenario: count}}
    counts = defaultdict(lambda: defaultdict(int))  # type: ignore

    # Extract model and scenario from each string
    for entry in [exp.name for exp in all_experiments]:
        model_match = re.search(r'model__mixed_([^_]+(?:_[^_]+)?)', entry)
        scenario_match = re.search(r'scenario__explicit_(\w+)', entry)
        if model_match and scenario_match:
            model = model_match.group(1)
            scenario = scenario_match.group(1)
            counts[model][scenario] += 1

    # Print the result
    for model, scenario_counts in counts.items():
        print(f"Model: {model}")
        for scenario, count in scenario_counts.items():
            print(f"  {scenario}: {count}")

    print("-"*10)
    print("OneStage - implicit")
    # Dictionary to store counts: {model: {scenario: count}}
    counts = defaultdict(lambda: defaultdict(int))

    # Extract model and scenario from each string
    for entry in [exp.name for exp in all_experiments]:
        model_match = re.search(r'model__one_stage_([^_]+(?:_[^_]+)?)', entry)
        scenario_match = re.search(r'scenario__implicit_(\w+)', entry)
        if model_match and scenario_match:
            model = model_match.group(1)
            scenario = scenario_match.group(1)
            counts[model][scenario] += 1

    # Print the result
    for model, scenario_counts in counts.items():
        print(f"Model: {model}")
        for scenario, count in scenario_counts.items():
            print(f"  {scenario}: {count}")