import optuna

from abc import abstractmethod
from typing import Dict, Any

from src.ml.classifier.base import BaseClassifier

class BaseBenchmark(BaseClassifier):
    """Base class for Neural Network models. To be implemented"""
    
    @staticmethod
    @abstractmethod
    def get_hyperparameters(trial: optuna.Trial) -> Dict[Any, Any]:
        raise NotImplementedError()
        
