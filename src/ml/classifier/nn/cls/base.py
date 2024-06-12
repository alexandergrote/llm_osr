from pydantic import BaseModel

from src.ml.classifier.base import BaseClassifier

class BaseNN(BaseModel, BaseClassifier):
    """Base class for Neural Network models. To be implemented"""
    pass
