import pandas as pd
import numpy as np
import optuna
import yaml
import os

from abc import ABC, abstractmethod

from typing import Union, Type, Any, Dict
from pydantic import BaseModel, field_validator, model_validator
from pydantic.v1.utils import deep_update

from src.util.dynamic_import import DynamicImport
from src.ml.classifier.base import BaseClassifier
from src.ml.evaluation.osr import Evaluator
from src.util.constants import Directory
from src.util.types import MLPrediction


class BaseHyperParams(ABC):

    @staticmethod
    @abstractmethod
    def get_params_for_study(trial: optuna.Trial) -> Dict[Any, Any]:
        raise NotImplementedError()


class DOCHyperParams(BaseHyperParams):

    @staticmethod
    def get_params_for_study(trial: optuna.Trial) -> Dict[Any, Any]:

        params = {
            'params': {
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
                'learning_rate': trial.suggest_float('learning_rate', 0.0, 1.0),
        }}

        return params
    

class HyperTuner(BaseModel, BaseClassifier):

    n_trials: int 
    timeout: int = 600

    model: Union[dict, Type[BaseClassifier]]
    hyperparams: Union[dict, Type[BaseHyperParams]]
    evaluator: Union[dict, Type[Evaluator]]

    @model_validator(mode="before")
    def init_params(data: dict) -> dict:
        
        if 'model' not in data:
            raise ValueError("model is required")
        
        model = data['model']

        if isinstance(model, str):

            yaml_path = Directory.CONFIG / os.path.join("ml__classifier", model)

            with open(yaml_path, 'r') as f:
                model_dict = yaml.safe_load(f)

            data['model'] = model_dict

        return data


    @field_validator('evaluator', 'hyperparams')
    def _set_model(cls, v):
        return DynamicImport.init_class_from_dict(dictionary=v)
    

    def _objective(self, trial: optuna.Trial, model: Union[Dict[Any, Any], Type[BaseClassifier]], x_train: pd.DataFrame, x_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series, **kwargs):

        if not isinstance(self.hyperparams, BaseHyperParams):
            raise ValueError("Hyperparams is not an instance of BaseHyperParams")
        
        if isinstance(model, BaseClassifier):
            raise ValueError("Model needs to be a dictionary")

        params = self.hyperparams.get_params_for_study(trial)

        # update model with new params
        model = deep_update(model, params)

        # create model
        model = DynamicImport.init_class_from_dict(dictionary=model) 

        if not isinstance(model, BaseClassifier):
            raise ValueError("Model is not an instance of BaseClassifier")

        model.fit(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid
        )

        y_pred = model.predict(x_valid)

        prediction = MLPrediction(
            y_pred=pd.Series(y_pred), 
            y_test=pd.Series(y_valid),
            classes_in_training=set(list(np.unique(y_train)))
        )

        result = self.evaluator.execute(
            prediction=prediction,
            **kwargs
        )

        return result['metrics']['f1_avg']

    def _run_hyperparameter_search(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ) -> optuna.study.Study:


        study = optuna.create_study(
            direction="maximize",
            study_name="Hyperparameter Tuning",
        )

        study.optimize(
            lambda trial: self._objective(
                trial, 
                self.model,
                x_train,
                x_valid,
                y_train,
                y_valid,
                **kwargs
            ),

            n_trials=self.n_trials,
            timeout=self.timeout
        )

        
        return study
    
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):
        
        if not isinstance(self.hyperparams, BaseHyperParams):
            raise ValueError("Hyperparams is not an instance of BaseHyperParams")
        
        if not isinstance(self.model, dict):
            raise ValueError("Model needs to be a dictionary")

        study = self._run_hyperparameter_search(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            **kwargs
        )

        # get best params
        params = self.hyperparams.get_params_for_study(study.best_trial)

        # update model with new params
        model = deep_update(self.model, params)

        # refit model
        self.model = DynamicImport.init_class_from_dict(dictionary=model)

        if not isinstance(self.model, BaseClassifier):
            raise ValueError("Model is not an instance of BaseClassifier")

        self.model.fit(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid
        )

        # store best params in kwargs
        kwargs['best_params'] = params

        return kwargs

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:

        if not isinstance(self.model, BaseClassifier):
            raise ValueError("Model is not an instance of BaseClassifier")

        return self.model.predict(x, **kwargs)
    
    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:

        if not isinstance(self.model, BaseClassifier):
            raise ValueError("Model is not an instance of BaseClassifier")

        return self.model.predict_proba(x, **kwargs)
