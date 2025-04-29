import warnings
import hydra

from omegaconf import DictConfig, OmegaConf
from typing import List

# load package specific code
from src.interface.execution_block import ExecutionBlock
from src.util.constants import DictConfigNames, Directory, File, get_hydra_output_dir
from src.util.logger import console
from src.util.environment import PydanticEnvironment
from src.util.mlflow_checks import get_experiment, get_results_as_str
from src.util.mlflow_columns import id_columns
from src.util.dict_extraction import get_nested_dict_values

env = PydanticEnvironment.from_environment()

warnings.filterwarnings(
    "ignore"
)

@hydra.main(
    config_path=str(Directory.CONFIG),
    config_name=File.CONFIG.stem,
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:

    console.rule(f"Executing experiment: {cfg['name']}")

    experiment = get_experiment(config=cfg)

    # check if experiment run exists
    if experiment is not None:

        console.log(f"Experiment {cfg['name']} exists. Its results are:")
        results = get_results_as_str(config=cfg, data=experiment)

        console.log(results)

        return

    # convert omega configuration dictionary to native python dictionary
    # needed to dynamically edit the dictionary
    cfg = OmegaConf.to_object(cfg)

    # sequence of execution
    sequence = [
        'io__import',
        'ml__datasplit',
        'ml__preprocessing',
        'ml__classifier',
        'ml__evaluation',
        'io__export'
    ]

    experiment_name = get_nested_dict_values(list_of_keys=[id_columns.experiment_name.yaml_keys], dictionary=cfg)[0]

    # output placeholder
    output = {
        DictConfigNames.RANDOM_SEED: cfg[DictConfigNames.RANDOM_SEED],
        'config': cfg,
        'output_dir': get_hydra_output_dir(),
        'experiment_name': experiment_name,
    }

    # init sequence
    event_blocks: List[ExecutionBlock] = [ExecutionBlock(name=key, block=cfg[key]) for key in sequence]

    if env.is_dryrun_mode():
        return

    # execute sequence
    for el in event_blocks:
        output = el.execute(**output)


if __name__ == "__main__":
    main()