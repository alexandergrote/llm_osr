import numpy as np
import os
import json

from abc import abstractmethod
from tqdm import tqdm
from copy import copy
from pydantic import BaseModel
from pydantic.config import ConfigDict
from numpy import ndarray
from typing import Optional, Union, Tuple, List, Callable, Dict, Any, Sized
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.ml.classifier.llm.util.prediction import Prediction
from src.ml.classifier.llm.util.rest import AbstractLLM
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.util.constants import ErrorValues
from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.llm.util.cosine_selector import CosineSelector
from src.ml.classifier.llm.util.prompt import PromptExample, PromptScenarioName, PromptCreator
from src.util.constants import Directory
from src.util.constants import UnknownClassLabel


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
    def _single_predict(self, text: str, use_cache: bool = False, is_prompt: bool = False, **kwargs) -> Tuple[str, float]:
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

    def _get_number_of_workers(self, x: Optional[Sized] = None) -> int:

        n = 10
        if x is None:
            return n

        if len(x) == 0:
            return n

        return min(n, len(x))

    def parallel_predict(self, x: np.ndarray, is_prompt: bool = False, **kwargs):

        result_text_list = [None] * len(x)
        result_score_list = [None] * len(x)

        # Mutex für den Zugriff auf Prediction.valid_labels
        #prediction_lock = threading.Lock()

        # Funktion für die Thread-Ausführung
        def process_single_prediction(idx, el_str):
            exp_name = kwargs.get('experiment_name', 'None')
            config = kwargs.get('config', 'None')
            seed = 'None'
            if isinstance(config, dict):
                seed = config.get('random_seed', 'None')

            base_str = f"{exp_name} - {seed}"
            
            result_text = ErrorValues.PARSING_STR.value
            result_score = float(ErrorValues.PARSING_NUM.value)
            
            try:
                
                result = self._single_predict(text=el_str, use_cache=self.use_cache, is_prompt=is_prompt, **kwargs)
                
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
        n_workers =  self._get_number_of_workers(x) # Begrenzen Sie die Anzahl der Threads

        # Fortschrittsanzeige
        pbar = tqdm(total=len(x), desc="LLM Prediction")
        
        # Thread-Pool erstellen und Aufgaben einreichen
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Einreichen der Aufgaben
            futures = []
            for i, el in enumerate(x):
                future = executor.submit(process_single_prediction, i, el[0])
                futures.append(future)
            
            # Verarbeiten der Ergebnisse
            for future in futures:
                try:
                    idx, print_str = future.result()
                    pbar.write(print_str)
                except Exception as e:
                    pbar.write(f"Error in thread: {e}")
                finally:
                    pbar.update(1)
        
        pbar.close()

        return result_text_list, result_score_list

    def get_cache(self) -> Dict[str, Any]:

        cache: dict = {}

        cache_dir = Directory.CACHING_DIR / os.path.join(
            "src", "ml", "classifier", "llm", "util", "rest.py", "rest", self.unknown_detection_model_name
        ) 

        # load jsons into dict
        n_worker = self._get_number_of_workers()

        def load_json(filename: Path):

            with open(filename, 'r') as file:
                data = json.load(file)

            return filename.stem, data

        x = list(cache_dir.glob("*.json"))

        # Fortschrittsanzeige
        pbar = tqdm(total=len(x), desc="Loading Cache")
        
        # Thread-Pool erstellen und Aufgaben einreichen
        with ThreadPoolExecutor(max_workers=n_worker) as executor:
            # Einreichen der Aufgaben
            futures = []
            for el in x:
                future = executor.submit(load_json, el)
                futures.append(future)
            
            # Verarbeiten der Ergebnisse
            for future in as_completed(futures):
                try:

                    json_name, json_data = future.result()
                    cache[json_name] = json_data

                except Exception as e:
                    pbar.write(f"Error in thread: {e}")
                finally:
                    pbar.update(1)
        
        pbar.close()

        return cache

    def get_active_and_hash_list(self, x: np.ndarray, cache: Dict[str, Any], prompt_function: Callable) -> Tuple[List[str], List[str]]:

        active_list = []
        hash_list = []
        
        def get_hash_value(text: str) -> Tuple[str, np.ndarray]:

            prompt = prompt_function(text)  # Generate
            hash_value = PromptCreator.hash_prompt(prompt)
            
            return hash_value, np.array([prompt])

        # Anzahl der Worker-Threads (anpassen nach Bedarf)
        n_workers =  self._get_number_of_workers(x) # Begrenzen Sie die Anzahl der Threads

        # Fortschrittsanzeige
        pbar = tqdm(enumerate(x), desc="Create active list and prompt hash", total=len(x))
        
        # Thread-Pool erstellen und Aufgaben einreichen
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Einreichen der Aufgaben
            futures = []
            for el in x:
                future = executor.submit(get_hash_value, el[0])
                futures.append(future)
            
            # Verarbeiten der Ergebnisse
            for future in futures:

                try:

                    hash_value, text = future.result()
                    hash_list.append(hash_value)

                    if hash_value not in cache:
                        active_list.append(text)

                except Exception as e:
                    pbar.write(f"Error in thread: {e}")

                finally:
                    pbar.update(1)
        
        pbar.close()

        return active_list, hash_list

    def get_result_from_logprobscore(self, logprob: LogProbScore) -> Tuple[str, float]:
        prediction_label = logprob.answer.label

        unknown_score = 0.0 if prediction_label != UnknownClassLabel.UNKNOWN_STR.value else 1.0

        return prediction_label, unknown_score

    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        # assert if array is 2D
        if len(x.shape) != 2:
            raise ValueError("Input should be 2D array")

        key = "get_prompt"

        if not hasattr(self, key):

            text_list, score_list = self.parallel_predict(x, is_prompt=False,  **kwargs)

            if include_outlierscore:
                return np.array(text_list), np.array(score_list)
        
            return np.array(text_list)

        prompt_function = getattr(self, key)

        if not isinstance(prompt_function, Callable):  # type: ignore
            raise ValueError(f"{key} must be callable")

        # use prompt function to generate hashes, which are needed for cache lookup
        cache = {}  # Store index -> value pairs
        active_list: List[str] = []  # list with texts to be predicted
        text_list = [None] * len(x)  # holds text results
        score_list = [None] * len(x)  # holds score results

        # load cache
        cache = self.get_cache()

        # get active and hash list
        active_list, hash_list = self.get_active_and_hash_list(x, cache, prompt_function)
            
        # Process only the active items that have not been cached
        # This speeds up the overall process since only a fraction of the datapoints need to be passed to the api provider
        sub_text_list, sub_score_list  = self.parallel_predict(active_list, is_prompt=True, **kwargs)

        # Restore cached items at their original positions
        active_index = 0

        # merge them all together and maintain order of original list
        for i in range(len(x)):

            cache_result = cache.get(hash_list[i], None)

            if cache_result is not None:
                logprob_score = LogProbScore(**cache_result)
                text_list[i], score_list[i] = self.get_result_from_logprobscore(logprob_score)

            else:
                text_list[i] = sub_text_list[active_index]  # Place processed item
                score_list[i] = sub_score_list[active_index]  # Place processed item
                active_index += 1
        
        if include_outlierscore:
            return np.array(text_list), np.array(score_list)
        
        return np.array(text_list)
    
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
