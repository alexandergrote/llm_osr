import random

from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

from src.ml.classifier.llm.util.outlier import OutlierValue
from src.util.constants import Directory, UnknownClassLabel
from src.ml.classifier.llm.util.prediction import Prediction
from src.util.hashing import Hash


CHAIN_OF_THOUGHT = "Let's think step by step"
UNKNOWN_CLASS_LABEL = UnknownClassLabel.UNKNOWN_STR.value


class PromptExample(BaseModel):
    text: str
    label: str


class PromptDataScenario(Enum):
    ZEROSHOT = "zeroshot"
    FEWSHOT = "fewshot"

    @classmethod
    def list(cls) -> List["PromptDataScenario"]:
        return [member for member in cls]

class PromptOODScenario(Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


class PromptScenarioName(Enum):
    EXPLICIT_WITH_LABELS = f"{PromptOODScenario.EXPLICIT.value}_{PromptDataScenario.FEWSHOT.value}"
    EXPLICIT_WITHOUT_LABELS = f"{PromptOODScenario.EXPLICIT.value}_{PromptDataScenario.ZEROSHOT.value}"
    IMPLICIT_WITH_LABELS = f"{PromptOODScenario.IMPLICIT.value}_{PromptDataScenario.FEWSHOT.value}"
    IMPLICIT_WITHOUT_LABELS = f"{PromptOODScenario.IMPLICIT.value}_{PromptDataScenario.ZEROSHOT.value}"
    MULTICLASS = "multiclass"

    @classmethod
    def create_from_enums(cls, ood_scenario: PromptOODScenario, data_scenario: PromptDataScenario) -> "PromptScenarioName":
        return cls(f"{ood_scenario.value}_{data_scenario.value}")

    @classmethod
    def is_implict(cls, value: str) -> bool:
        return PromptOODScenario.IMPLICIT.value in value.lower()
    

class PromptTemplate(Enum):
    ND_WITH_CLASSES = "nd_with_classes.txt"
    OSR = "osr.txt"
    MULTCLASS = "multiclass.txt"
    

class PromptCreator(BaseModel):

    name: PromptScenarioName
    template: PromptTemplate
    use_outlier_in_examples: bool

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return Hash.hash(prompt)

    @staticmethod
    def create_prompt(
        template_content: str, 
        text_to_classify: str, 
        examples: List[PromptExample],
        parser: PydanticOutputParser,
        outlier_examples: Optional[List[PromptExample]] = None, 
        additional_formatting: Optional[Dict[str, str]] = None,
        shuffle_examples: bool = True,
        use_classes_in_examples: bool = True,
        use_outlier_in_examples: bool = False, 
        ) -> str:

        if additional_formatting is None:
            additional_formatting = {}

        if not hasattr(parser.pydantic_object, "valid_labels"):
            raise ValueError("Pydantic model must have a 'valid_labels' attribute.")

        # work on copy
        all_examples = examples.copy()

        if use_outlier_in_examples:
            
            if (outlier_examples is None) or (len(outlier_examples) == 0):
                raise ValueError("Outlier examples must be provided when using use_outlier_in_examples.")
            
            all_examples.extend(outlier_examples)

        if shuffle_examples:
            random.seed(42)
            random.shuffle(all_examples)

        if use_classes_in_examples:
            examples_msg = '\n'.join([f"{example.text} -> {example.label}" for example in all_examples])
        else:
            examples_msg = '\n'.join([f"{example.text}" for example in all_examples])


        instructions = "You must provide your reasoning first and then the label."
        instructions += parser.get_format_instructions()

        classes_msg = "Valid labels:\n"
        classes_msg += '\n'.join(parser.pydantic_object.valid_labels)

        formatting_fields = {
            "examples": examples_msg,
            "text": text_to_classify,
            "json_instructions": instructions,
            "classes": classes_msg,
            "chain_of_thought": CHAIN_OF_THOUGHT,
            **additional_formatting
        }

        # check if fields are in template
        for field in formatting_fields.keys():

            field = "{" + field + "}"

            if field not in template_content:
                raise ValueError(f"Field '{field}' not found in template.")

        prompt = template_content.format(**formatting_fields)

        return prompt

    def create(self, text_to_classify: str, examples: List[PromptExample], parser: PydanticOutputParser, outlier_examples: Optional[List[PromptExample]]) -> str:

        with open(Directory.PROMPT_DIR / self.template.value, "r") as f:
            template_content = f.read()

        additional_formatting: dict = {}

        if self.name.value.startswith("implicit"):

            additional_formatting = {
                "unknown_label": UNKNOWN_CLASS_LABEL
            }

        elif self.name.value.startswith("explicit"):

            new_examples = []

            for el in examples:

                new_example = PromptExample(
                    label=OutlierValue.INLIER.value,
                    text=el.text
                )
                new_examples.append(new_example)

            examples = new_examples

        elif self.name.value.startswith("multiclass"):
            pass

        else:
            raise ValueError(f"Unknown template name '{self.name}'.")

        prompt = PromptCreator.create_prompt(
            template_content=template_content, 
            text_to_classify=text_to_classify,
            examples=examples,
            parser=parser,
            additional_formatting=additional_formatting,
            use_outlier_in_examples=self.use_outlier_in_examples,
            outlier_examples=outlier_examples
        )

        return prompt


explicit_with_labels = PromptCreator(
    name=PromptScenarioName.EXPLICIT_WITH_LABELS,
    template=PromptTemplate.ND_WITH_CLASSES,
    use_outlier_in_examples=True,
)

explicit_without_labels = PromptCreator(
    name=PromptScenarioName.EXPLICIT_WITHOUT_LABELS,
    template=PromptTemplate.ND_WITH_CLASSES,
    use_outlier_in_examples=False,
)

implicit_with_labels = PromptCreator(
    name=PromptScenarioName.IMPLICIT_WITH_LABELS,
    template=PromptTemplate.OSR,
    use_outlier_in_examples=True,
)

implicit_without_labels = PromptCreator(
    name=PromptScenarioName.IMPLICIT_WITHOUT_LABELS,
    template=PromptTemplate.OSR,
    use_outlier_in_examples=False,
)

multiclass = PromptCreator(
    name=PromptScenarioName.MULTICLASS,
    template=PromptTemplate.MULTCLASS,
    use_outlier_in_examples=False,
)

PROMPT_SCENARIOS: Dict[PromptScenarioName, PromptCreator] = {
    PromptScenarioName.EXPLICIT_WITH_LABELS: explicit_with_labels,
    PromptScenarioName.EXPLICIT_WITHOUT_LABELS: explicit_without_labels,
    PromptScenarioName.IMPLICIT_WITH_LABELS: implicit_with_labels,
    PromptScenarioName.IMPLICIT_WITHOUT_LABELS: implicit_without_labels,
    PromptScenarioName.MULTICLASS: multiclass,
}

if __name__ == '__main__':

    examples = [                                                                                                                               
        PromptExample(text="It's raining heavily.", label="weather"),                                                                               
        PromptExample(text="The final score was 3-1.", label="sports"),                                                                             
        PromptExample(text="The president signed a new law.", label="politics")                                                                     
    ]

    text = 'What is the capital of France?'

    ### scenarios
    # explizit, with labels (fewshot)
    # explizit, without labels (zeroshot)
    # implizit, with labels (fewshot)
    # implizit, without labels (zeroshot)

    # models
    # llama 3.3 70b
    # llama 3.1 8b
    # qwen 25b --> samba nova
    # phi4 14b

    

    scenarios = [
        explicit_with_labels,
        explicit_without_labels,
        implicit_with_labels,
        implicit_without_labels,
        multiclass
    ]

    for scenario in scenarios:

        print('-'*20)
        print(scenario.name,"\n")

        valid_labels = [el.label for el in examples]

        outlier_examples = [                                                                                                                      
            PromptExample(text="The cake is delicious.", label="outlier"),                                                                                                                                                     
        ]

        if scenario.name.value.startswith("implicit"):

            valid_labels.append(UnknownClassLabel.UNKNOWN_STR.value)

            outlier_examples = [                                                                                                                      
                PromptExample(text="The cake is delicious.", label=UnknownClassLabel.UNKNOWN_STR.value),                                                                                                                                                     
            ]

        elif scenario.name.value.startswith("explicit"):

            valid_labels = ['inlier', 'outlier']

        elif scenario.name.value.startswith("multiclass"):
            pass

        else:
            raise ValueError(f"Unknown scenario name: {scenario.name}")

        Prediction.valid_labels = valid_labels

        parser = PydanticOutputParser(
            pydantic_object=Prediction, 
        )

        prompt = scenario.create(
            text_to_classify=text,
            parser=parser, 
            examples=examples,
            outlier_examples=outlier_examples
        )

        print(prompt)