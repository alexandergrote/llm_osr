import hydra
import warnings
from omegaconf import DictConfig, OmegaConf

# load package specific code
from interface.execution_block import ExecutionBlock
from util.constants import Directory, File
from util.logging import console

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

    # output placeholder
    output = {}

    # init sequence
    sequence = [ExecutionBlock(name=key, block=cfg[key]) for key in sequence]
    
    # execute sequence
    for el in sequence:
        output[el.name] = el.execute(**output)


if __name__ == "__main__":
    main()