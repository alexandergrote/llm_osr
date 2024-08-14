import mlflow
from mlflow import MlflowClient
from src.util.constants import Directory

def get_mlflow_client() -> MlflowClient:

    tracking_uri = Directory.ROOT / "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    return client