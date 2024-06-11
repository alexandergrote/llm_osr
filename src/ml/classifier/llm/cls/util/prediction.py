from langchain_core import pydantic_v1

class Prediction(pydantic_v1.BaseModel):
    reasoning: str = pydantic_v1.Field(description="The reasoning behind the prediction")
    label: str = pydantic_v1.Field(description="The predicted label")

    @pydantic_v1.validator("label", allow_reuse=True)
    def label_must_be_valid(cls, label: str):

        if not hasattr(cls, 'valid_labels'):
            raise Exception("No class labels supplied")
        
        valid_labels = cls.valid_labels

        if label.lower() not in [valid_label.lower() for valid_label in valid_labels]:
            raise ValueError(f"Label must be one of {valid_labels}")
        return label