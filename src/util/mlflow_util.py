import os
import mlflow
from mlflow import MlflowClient
from pathlib import Path
from src.util.constants import Directory


def get_tracking_uri():

    if os.name == 'nt':
        return Path(rf"file:\\{str(Directory.ROOT)}\mlruns")
    
    return Path(rf"file:{str(Directory.ROOT)}/mlruns")


def get_mlflow_client() -> MlflowClient:

    tracking_uri = Directory.ROOT / "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    return client
