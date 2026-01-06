# Prompting for the Unknown: Leveraging In-Context-Learning for Few-Shot Open Set Classification

This repository is associated with the paper "Prompting for the Unknown: Leveraging In-Context-Learning for Few-Shot Open Set Classification".

Kindly be aware that the code is subject to change as the paper may be developed further.

## Explaining the Code Structure and High-Level Usage

![Repository structure](/static/images/ml2mlflow2vis.png)

Broadly speaking, the repository contains two pipelines that serve different purposes:
1) Executing the machine learning experiments and storing the result of each run in a MlFlow database
2) Afterwards, we can query the results from the MlFlow database, analyse and visualize them.

You can run both pipelines together or indepently. This happens by means of `python src/experiments/cli.py <pipeline_name>`. You also have the possibility to skip either the execution or anaylsis/visualization of the pipelines:

- Only Machine Learning Experiment: `python src/experiments/cli.py <pipeline_name> --skip-visualization`  
- Only Analysis / Visualization: `python src/experiments/cli.py <pipeline_name> --skip-execution` 

Overall, there are three different setups, as defined in `src/experiments/factory.py`:
1) Benchmark Analysis (pipeline name "benchmark")
2) Prompting Strategy Analysis (pipeline name "ood")
3) Error Analysis (pipeline name "error")

### Machine Learning Pipeline

This pipeline executes machine learning experiments using different models and configurations. The results are stored in an MlFlow database. Its steps are as follows:

1) Fetch data `src/io/data_import` with interface defined in `src/io/data_import/base.py`
2) Preprocess data `src/ml/datasplit` with interface defined in `src/ml/datasplit/base.py`
3) Preprocess data `src/ml/preprocessing` with interface defined in `src/ml/preprocessing/base.py`
4) Train models `src/ml/classifier` with interface defined in `src/ml/classifier/base.py`
5) Evaluate models `src/ml/evaluation` with interface defined in `src/ml/evaluation/base.py`
6) Export results `src/io/data_export` with interface defined in `src/io/data_export/base.py`


### Analysis / Visualization Pipeline

The different analysing scsripts including the visualizations are located in `src/experiments/analysis` and are named accordingly. 

## Usage 

### Installation

To use this package, you need to install the required packages and have a version of Python 3 installed. The python package has been tested with Python 3.10.16.

First, clone the repository:

`git clone git@github.com:alexandergrote/llm_osr.git`

Next, navigate to the repository directory and install the package using pip:

- `pip install -r requirements.txt`
- `pip install -r dev-requirements.tx`
- `pip install -r test-requirements.txt`
- `pip install .`
- `pre-commit install`


### Commands

To reproduce all experiments, including the figures and tables, you need to execute these function calls, which correspond to the four different setups mentioned above. These function calls are:
- `python src/experiments/cli.py benchmark`
- `python src/experiments/cli.py ood`
- `python src/experiments/cli.py error`


## Tests

To run unit tests, simply execute: ``python -m unittest``

- If you wish to execute only the unit tests and not the integration tests, you can execute: ``python -m unittest discover tests/unit``
- If you wish to execute only the integration tests and not the unit tests, you can execute: ``python -m unittest discover tests/integration``
- If you wish to execute only the end2end tests and not the remaining tests, you can execute: ``python -m unittest discover tests/end2end``

## Additional Remarks for Source Code Usage

This repository contains some opionated snippets of code. For instance, it uses `hydra` for configuration management, ``mlflow`` to keep track of the machine learning runs and a custom cli interface for administering the different pipelines. With these remarks, we will hopefully make it easier for someone new to use this codebase.

### Env File

At the writing of this README file, you are required to provide API keys as specified in `example.env`. Simply execute `mv example.env .env` and edit the API keys.

### Hydra

First and foremost, the main entry script for each machine learning run is `src/main.py`. You can override the default configuration by passing in command line arguments. For example, to run the churn prediction pipeline with a different dataset, you can use the following command: ` python src/main.py io__import=20News`. For more information see the official [hydra documentation](https://hydra.cc/docs/intro/). 

### Mlflow

Start the mlflow gui with 

```
mlflow ui --port 5000
```
and inspect your results visually.