import numpy as np
import threading

from abc import abstractmethod
from tqdm import tqdm
from copy import copy
from pydantic import BaseModel
from pydantic.config import ConfigDict
from numpy import ndarray
from typing import Optional, Union, Tuple, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.ml.classifier.llm.util.prediction import Prediction
from src.ml.classifier.llm.util.rest import AbstractLLM
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.util.constants import ErrorValues
from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.llm.util.cosine_selector import CosineSelector
from src.ml.classifier.llm.util.prompt import PromptExample, PromptScenarioName
from src.util.constants import Directory


class AbstractClassifierLLM(BaseModel, BaseClassifier):

    # placeholders, values will be set later
    x_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    x_valid: Optional[np.ndarray] = None
    y_valid: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None

    unknown_detection_scenario: PromptScenarioName
    unknown_detection_model_name: str 

    n_unknown_examples: int = 5
    
    # use cache
    use_cache: bool = False

    # fewshot selection of data points
    selector: Union[CosineSelector, dict]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, **kwargs):
        raise NotImplementedError("Method must be implemented in subclass")

    @abstractmethod
    def _single_predict(self, text: str, use_cache: bool = False, **kwargs) -> Tuple[str, float]:
        raise NotImplementedError("Method must be implemented in subclass")


    def _get_examples(self, text_to_classify: str) -> List[PromptExample]:

        if self.y_train is None:
            raise ValueError("Model is not fitted")
        
        if self.y_valid is None:
            raise ValueError("Model is not fitted")
        
        if isinstance(self.selector, dict):
            raise ValueError("Selector not set")

        examples = self.selector.get_examples(
            text=text_to_classify,
            x_train=self.x_train,
            y_train=self.y_train
        )

        return examples

    
    def _get_outlier_examples(self, outlier_value: str) -> List[PromptExample]:

        if self.classes is None:
            raise ValueError("Model not fitted")

        if self.y_valid is None:
            raise ValueError("Model not fitted")

        if self.x_valid is None:
            raise ValueError("Model not fitted")
        
        # get mask of all unknown classes in y_valid
        mask = ~np.isin(self.y_valid, self.classes)

        x_valid_unknown = self.x_valid[mask]
        x_valid_unknown = x_valid_unknown.reshape(-1)

        examples = [
            PromptExample(text=el, label=outlier_value) for el in x_valid_unknown
        ]

        if (len(examples) == 0):
            raise ValueError("No unknown classes found in validation set")

        if len(examples) > self.n_unknown_examples:
            examples = examples[:self.n_unknown_examples]

        return examples


    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # assert if array is 2D
        if len(x.shape) != 2:
            raise ValueError("Input should be 2D array")

        result_text_list = [None] * len(x)
        result_score_list = [None] * len(x)
        
        # Mutex für den Zugriff auf Prediction.valid_labels
        prediction_lock = threading.Lock()
        
        # Funktion für die Thread-Ausführung
        def process_single_prediction(idx, el_str):
            exp_name = kwargs.get('experiment_name', 'None')
            config = kwargs.get('config', 'None')
            seed = 'None'
            if isinstance(config, dict):
                seed = config.get('random_seed', 'None')

            base_str = f"{exp_name} - {seed} - {el_str}"
            
            result_text = ErrorValues.PARSING_STR.value
            result_score = float(ErrorValues.PARSING_NUM.value)
            
            try:
                # Kritischer Abschnitt mit Lock
                with prediction_lock:
                    result = self._single_predict(text=el_str, use_cache=self.use_cache)
                
                if result is not None:
                    result_text, result_score = result
                    print_str = f"{base_str} - {result_text}"
                else:
                    print_str = f"{base_str} - None"
                    
            except Exception as e:
                print_str = f"{base_str} - Error: {e}"
            
            # Speichern der Ergebnisse
            result_text_list[idx] = result_text
            result_score_list[idx] = result_score
            
            # Logging
            output_file = Directory.ROOT / 'tmp/logs'
            output_file.mkdir(parents=True, exist_ok=True)
            output_file = output_file / f'{exp_name}.log'
            datetime_now = str(datetime.now())
            with open(output_file, 'a') as f:
                f.write(datetime_now + ' - ' + print_str + '\n')
            
            return idx, print_str
        
        # Anzahl der Worker-Threads (anpassen nach Bedarf)
        max_workers = min(10, len(x))  # Begrenzen Sie die Anzahl der Threads
        
        # Fortschrittsanzeige
        pbar = tqdm(total=len(x), desc="LLM Prediction")
        
        # Thread-Pool erstellen und Aufgaben einreichen
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Einreichen der Aufgaben
            futures = []
            for i, el in enumerate(x):
                future = executor.submit(process_single_prediction, i, el[0])
                futures.append(future)
            
            # Verarbeiten der Ergebnisse
            for future in as_completed(futures):
                try:
                    idx, print_str = future.result()
                    pbar.write(print_str)
                except Exception as e:
                    pbar.write(f"Error in thread: {e}")
                finally:
                    pbar.update(1)
        
        pbar.close()
        
        if include_outlierscore:
            return np.array(result_text_list), np.array(result_score_list)
        
        return np.array(result_text_list)
    
    def predict_proba(self, x: ndarray, **kwargs) -> ndarray:
        pass


class LLMClassifierMixin:

    def _get_parsed_output(self, model: AbstractLLM, text: str, valid_labels: List[str], use_cache: bool = False, retries: int = 5, **kwargs) -> Optional[LogProbScore]:

        # work on copy
        prompt_copy = copy(text)
        
        Prediction.valid_labels = valid_labels
        
        for _ in range(retries):

            try:

                result: LogProbScore = model(
                    text=prompt_copy,
                    pydantic_model=Prediction, 
                    use_cache=use_cache, 
                    **kwargs
                )

                return result

            except Exception:
                pass

        return None
