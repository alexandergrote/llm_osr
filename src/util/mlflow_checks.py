from omegaconf import DictConfig

from src.io.data_import.mlflow_engine import QueryEngine
from src.util.mlflow_columns import id_columns
from src.util.dict_extraction import get_nested_dict_values
from src.util.mlflow_columns import f1_analysis_columns, MLFlowColumn
from src.util.environment import PydanticEnvironment


def check_if_experiment_run_exists(config: DictConfig) -> bool:
    
    experiment_name = get_nested_dict_values(list_of_keys=[id_columns.experiment_name.yaml_keys], dictionary=config)[0]

    # check if name has been initialized in database        
    experiment_names = QueryEngine.get_experiment_names()

    if experiment_name not in experiment_names:
        return False

    # check if required run exists
    data = QueryEngine.get_results_of_single_experiment(experiment_name, n=999)

    if len(data) == 0:
        return False
    
    mlflow_columns = id_columns.get_columns()

    for mlflow_column in mlflow_columns:

        if mlflow_column.column_name not in data.columns:
            raise ValueError(f"Column {mlflow_column.column_name} must be present in experiment.")
        
        unique_values = [str(v) for v in data[mlflow_column.column_name].unique()]
        value = get_nested_dict_values(list_of_keys=[mlflow_column.yaml_keys], dictionary=config)[0]

        if str(value) not in unique_values:
            return False

    return True

def get_results_as_str(config: DictConfig) -> str:

    # get list of keys for nested dict
    list_values = []

    columns = id_columns.get_columns()

    env = PydanticEnvironment.from_environment()

    if not env.is_dryrun_mode():
        columns += f1_analysis_columns.get_columns()

    assert all(isinstance(el, MLFlowColumn) for el in columns)

    for el in columns:

        assert isinstance(el, MLFlowColumn)

        values = get_nested_dict_values(
            list_of_keys=[el.yaml_keys], 
            dictionary=config
        )

        assert len(values) == 1

        list_values.append(f"{el.verbose_str}: {values[0]}")

    return '\n'.join(list_values)