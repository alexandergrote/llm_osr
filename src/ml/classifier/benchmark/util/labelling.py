import numpy as np

from pydantic.v1 import validate_arguments
from typing import Any, Tuple


class LabellingUtilities:

    @staticmethod
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def create_label_mapping(y: np.ndarray) -> Tuple[dict, dict]:

        label2idx = {label: idx for idx, label in enumerate(np.unique(y))}
        idx2label = {idx: label for label, idx in label2idx.items()}

        return label2idx, idx2label

    @staticmethod
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def map_labels(y: np.ndarray, mapping: dict, target_dtype: str, unknown_value: Any) -> np.ndarray:
        
        dtype_mapping = {
            'int': int,
            'str': np.object_
        }   

        dtype_final = dtype_mapping[target_dtype]

        y_relabelled = np.copy(y)
        y_relabelled = y_relabelled.astype(np.object_)

        mask_transformed = np.zeros_like(y)

        for key, value in mapping.items():
            mask_treatedformed_sub = y == key
            y_relabelled[mask_treatedformed_sub] = value
            mask_transformed[mask_treatedformed_sub] = 1

        y_relabelled[mask_transformed == 0] = unknown_value

        y_relabelled = y_relabelled.astype(dtype_final)
        
        return y_relabelled