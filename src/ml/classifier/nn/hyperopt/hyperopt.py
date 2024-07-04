import pandas as pd
import numpy as np
import optuna
import yaml
import os

from typing import Union, Type, Any, Dict, Tuple
from pydantic import BaseModel, field_validator, model_validator
from pydantic.v1.utils import deep_update

from src.util.dynamic_import import DynamicImport
from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.nn.cls.base import BaseBenchmark
from src.ml.evaluation.osr import Evaluator
from src.util.constants import Directory
from src.util.types import MLPrediction
    

class HyperTuner(BaseModel, BaseClassifier):

    n_trials: int 
    timeout: int = 600

    model: Union[dict, Type[BaseBenchmark]]
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


    @field_validator('evaluator')
    def _set_model(cls, v):
        return DynamicImport.init_class_from_dict(dictionary=v)
    
    @staticmethod
    def init_model_from_params(model: dict, params: dict) -> Type[BaseBenchmark]:

        # update model with new params
        model = deep_update(model, params)

        # create model
        return DynamicImport.init_class_from_dict(dictionary=model) 
    

    def _objective(self, trial: optuna.Trial, model: Dict[Any, Any], x_train: pd.DataFrame, x_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series, **kwargs):
        
        if not isinstance(self.evaluator, Evaluator):
            raise ValueError("Evaluator is not an instance of Evaluator")

        model_cls = self.init_model_from_params(model, model)

        params = model_cls.get_hyperparameters(trial)

        model_cls = self.init_model_from_params(model, params)
        
        if not isinstance(model_cls, BaseBenchmark):
            raise ValueError("Model is not an instance of BaseBenchmark")

        model_cls.fit(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            **kwargs
        )

        y_pred = model_cls.predict(x=x_valid)

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
        model: Dict[Any, Any],
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
                model,
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
        
        if not isinstance(self.model, dict):
            raise ValueError("Model needs to be a dictionary")

        study = self._run_hyperparameter_search(
            model=self.model,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            **kwargs
        )

        model_cls = self.init_model_from_params(self.model, self.model)

        # get best params
        params = model_cls.get_hyperparameters(study.best_trial)

        # refit model
        self.model = self.init_model_from_params(self.model, params)

        if not isinstance(self.model, BaseBenchmark):
            raise ValueError("Model is not an instance of BaseBenchmark")

        self.model.fit(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid
        )

        # store best params in kwargs
        kwargs['best_params'] = params

        return kwargs

    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        if not isinstance(self.model, BaseBenchmark):
            raise ValueError("Model is not an instance of BaseBenchmark")
        
        if include_outlierscore:
            raise ValueError("Outlier score not implemented yet")

        return self.model.predict(x, include_outlierscore=include_outlierscore, **kwargs)
    
    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:

        if not isinstance(self.model, BaseBenchmark):
            raise ValueError("Model is not an instance of BaseBenchmark")

        return self.model.predict_proba(x, **kwargs)
