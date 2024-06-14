import mlflow
import os
from pathlib import Path
from pydantic import BaseModel, validate_arguments
from omegaconf import DictConfig, ListConfig

from src.io.data_export.base import BaseExporter
from src.util.constants import Directory


class Exporter(BaseModel, BaseExporter):

    experiment_name: str

    @staticmethod
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def _log_artifacts(path: Path):

        for file in list(path.glob("*")):

            # ignore hydra output
            if file.name in [".hydra", "main.log", r"\\"]:
                continue

            mlflow.log_artifact(file)

    @staticmethod
    @validate_arguments
    def _log_metrics_from_dict(metrics: dict):

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    @staticmethod
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def _log_params_from_omegaconf_dict(params: DictConfig):
        """
        This function logs all parameters from a dictionary
        """
        for param_name, element in params.items():

            if not isinstance(element, DictConfig):
                mlflow.log_param(param_name, element)
                continue

            Exporter._log_params_recursive(param_name, element)

    @staticmethod
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def _log_params_recursive(parent_name: str, element: DictConfig):

        """
        This function recursively checks if the passed element is a dictionary.
        If yes, it selects all keys and executes itself again until a parameter
        is found. The parameter is then logged
        """

        for k, v in element.items():

            if isinstance(v, DictConfig):
                Exporter._log_params_recursive(f"{parent_name}.{k}", v)

            elif isinstance(v, ListConfig):

                # log each DictConfig element in list

                # store elements that are not a dictconfig to log them later
                leftovers = []

                for el in v:

                    if isinstance(el, DictConfig):
                        idx = v.index(el)
                        Exporter._log_params_recursive(f"{parent_name}.{k}.{idx}", el)
                    else:
                        leftovers.append(el)

                mlflow.log_param(key=f"{parent_name}.{k}", value=','.join(leftovers))

            else:
                mlflow.log_param(key=f"{parent_name}.{k}", value=v)

    def export(self, **kwargs):

        assert "config" in kwargs, "kwargs must contain config"
        assert "metrics" in kwargs, "kwaargs must contain metrics"
        assert "output_dir" in kwargs, "kwargs must contain output"
        
        config = kwargs["config"]
        metrics = kwargs["metrics"]
        output_dir = kwargs["output_dir"]

        assert isinstance(config, dict)
        assert isinstance(metrics, dict)
        assert isinstance(output_dir, Path)

        
        # define tracking uri
        tracking_uri = Directory.ROOT / "mlruns"

        # overwrite uri if running on windows
        if os.name == 'nt':
            tracking_uri = Path(rf"file:\\{str(Directory.ROOT)}\mlruns")

        # set tracking uri
        mlflow.set_tracking_uri(str(tracking_uri))

        # set experiment name
        mlflow.set_experiment(self.experiment_name)

        # initialize mlflow run
        with mlflow.start_run():

            omega_conf = DictConfig(config)

            # log all params
            Exporter._log_params_from_omegaconf_dict(omega_conf)

            # log all metrics
            Exporter._log_metrics_from_dict(metrics)

            # log all artifacts
            Exporter._log_artifacts(path=output_dir)


__all__ = ["Exporter"]