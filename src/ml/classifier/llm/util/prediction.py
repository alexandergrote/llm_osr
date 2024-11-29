import warnings

from pydantic import BaseModel, Field
from pydantic.warnings import PydanticDeprecatedSince20
from langchain import pydantic_v1
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Callable

from src.ml.util.job_queue import Job
from src.ml.classifier.llm.util.request import RequestOutput

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)


class Prediction(BaseModel):

    reasoning: str = Field(description="The reasoning behind the prediction")
    label: str = Field(description="The predicted label")
    
    @classmethod
    def from_llm_job(cls, filename: str, class_labels: List[str], request_output_fun: Callable) -> "Prediction":

        job = Job.from_json_file(filename)

        output: RequestOutput = request_output_fun(job.request_output)
        
        PredictionV1.valid_labels = class_labels
        
        parser = PydanticOutputParser(pydantic_object=PredictionV1)
        prediction: PredictionV1 = parser.parse(output.text)

        return Prediction(**prediction.dict())


class PredictionV1(pydantic_v1.BaseModel):
    
    reasoning: str = pydantic_v1.Field(description="The reasoning behind the prediction")
    label: str = pydantic_v1.Field(description="The predicted label")

    @pydantic_v1.validator("label")
    def label_must_be_valid(cls, label: str):

        if not hasattr(cls, 'valid_labels'):
            raise Exception("No class labels supplied")
        
        valid_labels = cls.valid_labels

        if label.lower() not in [valid_label.lower() for valid_label in valid_labels]:
            raise ValueError(f"Label must be one of {valid_labels}")
        
        return label.lower()
    
if __name__ == '__main__':

    Prediction.valid_labels = ["label1", "label2", "label3"]

    prediction = Prediction(reasoning="Some reasoning", label="label1")
    
    print(prediction)

    print(prediction.valid_labels)

