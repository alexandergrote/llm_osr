import mlflow
import os
import re

from argparse import ArgumentParser
from pathlib import Path

from src.util.constants import Directory
from src.util.mlflow_util import get_tracking_uri, get_mlflow_client

def delete_experiments(pattern: str):
    """
    Delete MLflow experiments matching the given pattern.

    :param pattern: Regular expression pattern to search for in experiment names
    """

    """tracking_uri = get_tracking_uri()
    print(tracking_uri)
    mlflow.set_tracking_uri(str(tracking_uri))
    """
    client = get_mlflow_client()

    # Get a list of all experiments
    experiments = client.search_experiments()

    # Filter experiments based on the given pattern
    filtered_experiments = [experiment for experiment in experiments if re.search(pattern, experiment.name)]

    if not filtered_experiments:
        print(f"No experiments found matching the pattern: {pattern}")
        return
    
    # Print the list of experiments to be deleted
    print(f"The following experiments will be deleted:")
    for experiment in filtered_experiments:
        print(f"ID: {experiment.experiment_id}, Name: {experiment.name}")

    # Ask for user confirmation before deletion
    confirmation = input(f"Are you sure you want to delete {len(filtered_experiments)} experiments matching the pattern: {pattern}? (y/n): ")
    if confirmation.lower() != 'y':
        print("Deletion cancelled.")
        return

    # Iterate through the filtered experiments and delete them
    for experiment in filtered_experiments:
        exp_id = experiment.experiment_id
        print(f"Deleting experiment ID: {exp_id}, Name: {experiment.name}")
        mlflow.delete_experiment(exp_id)

        # apply garbage cleaner of mlflow to permanently delete experiment data
        os.environ['MLFLOW_TRACKING_URI'] = str(get_tracking_uri())
        os.system(f"mlflow gc --experiment-ids {exp_id}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Delete MLflow experiments matching the given pattern.")
    parser.add_argument("pattern", type=str, help="Regular expression pattern to search for in experiment names")
    
    args = parser.parse_args()
    delete_experiments(args.pattern)