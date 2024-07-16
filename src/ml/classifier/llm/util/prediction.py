from pydantic import BaseModel, Field, validator
from langchain import pydantic_v1


class Prediction(BaseModel):

    reasoning: str = Field(description="The reasoning behind the prediction")
    label: str = Field(description="The predicted label")

    @validator("label", allow_reuse=True)
    def label_must_be_valid(cls, label: str):

        if not hasattr(cls, 'valid_labels'):
            raise Exception("No class labels supplied")
        
        valid_labels = cls.valid_labels

        if label.lower() not in [valid_label.lower() for valid_label in valid_labels]:
            raise ValueError(f"Label must be one of {valid_labels}")
        
        return label.lower()

class PredictionV1(pydantic_v1.BaseModel):
    
    reasoning: str = pydantic_v1.Field(description="The reasoning behind the prediction")
    label: str = pydantic_v1.Field(description="The predicted label")

    @pydantic_v1.validator("label", allow_reuse=True)
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

    PredictionV1.create_from_pydantic_v2(prediction)