import random

from typing import List, Dict, Optional
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

from src.ml.classifier.llm.util.outlier import OutlierValue
from src.util.constants import Directory, UnknownClassLabel
from src.ml.classifier.llm.util.prediction import Prediction


CHAIN_OF_THOUGHT = "Let's think step by step"
UNKNOWN_CLASS_LABEL = UnknownClassLabel.UNKNOWN_STR.value

class PromptExample(BaseModel):
    text: str
    label: str


def create_prompt(
    template: str, 
    text_to_classify: str, 
    examples: List[PromptExample],
    parser: PydanticOutputParser,
    shuffle_examples: bool = False,
    use_classes_in_examples: bool = True,
    binarize_labels: bool = False,
    use_outlier_in_examples: bool = False, 
    outlier_examples: Optional[List[PromptExample]] = None, 
    additional_formatting: Optional[Dict[str, str]] = None
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

        if field not in template:
            raise ValueError(f"Field '{field}' not found in template.")

    prompt = template.format(**formatting_fields)

    return prompt

if __name__ == '__main__':


    examples = [                                                                                                                               
        PromptExample(text="It's raining heavily.", label="weather"),                                                                               
        PromptExample(text="The final score was 3-1.", label="sports"),                                                                             
        PromptExample(text="The president signed a new law.", label="politics")                                                                     
    ]

    outlier_examples = [                                                                                                                      
        PromptExample(text="The cake is delicious.", label="outlier"),                                                                                                                                                     
    ]

    text = 'What is the capital of France?'

    for file in Directory.PROMPT_DIR.glob('*.txt'):

        print(file)

        with open(file, 'r') as f:
            template = f.read()

        valid_labels = []
        additional_formatting = {}
        use_classes_in_examples = True

        if file.name.startswith("multiclass") or file.name.startswith("osr"):
            valid_labels = ['weather', 'sports', 'politics']

        if file.name.startswith('osr'):
            valid_labels += [UnknownClassLabel.UNKNOWN_STR.value]
            additional_formatting['unknown_label'] = UnknownClassLabel.UNKNOWN_STR.value

        if file.name.startswith('ood') or file.name.startswith("nd"):
            valid_labels = OutlierValue.list()

        if file.name.startswith('nd_no_classes'):
            use_classes_in_examples=False

        Prediction.valid_labels = valid_labels

        parser = PydanticOutputParser(
            pydantic_object=Prediction, 
        )

        prompt = create_prompt(
            template=template,
            text_to_classify=text,
            examples=examples,
            parser=parser,
            use_classes_in_examples=use_classes_in_examples,
            additional_formatting=additional_formatting,
            shuffle_examples=True
        )

        print('-'*25)
        print(prompt)

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

    class Scenario(BaseModel):

        name: str
        template: str
        use_outlier_in_examples: bool

        def create(self, parser: PydanticOutputParser, additional_formatting: dict = {}) -> str:

            with open(Directory.PROMPT_DIR / self.template, "r") as f:
                template_content = f.read()

            prompt = create_prompt(
                template=template_content, 
                text_to_classify=text,
                examples=examples,
                parser=parser,
                additional_formatting=additional_formatting
            )

            return prompt



    explicit_with_labels = Scenario(
        name="explicit_with_labels",
        template="nd_with_classes.txt",
        use_outlier_in_examples=True,
    )

    explicit_without_labels = Scenario(
        name="explicit_without_labels",
        template="nd_with_classes.txt",
        use_outlier_in_examples=False,
    )

    implicit_with_labels = Scenario(
        name="implicit_with_labels",
        template="osr.txt",
        use_outlier_in_examples=True,
    )

    implicit_without_labels = Scenario(
        name="implicit_without_labels",
        template="osr.txt",
        use_outlier_in_examples=False,
    )

    scenarios = [
        explicit_with_labels,
        explicit_without_labels,
        implicit_with_labels,
        implicit_without_labels,
    ]

    for scenario in scenarios:

        print('-'*20)
        print(scenario.name)

        Prediction.valid_labels = valid_labels

        parser = PydanticOutputParser(
            pydantic_object=Prediction, 
        )

        if scenario.name.startswith("implicit"):

            additional_formatting = {
                "unknown_label": UnknownClassLabel.UNKNOWN_STR.value
            }
        else:
            additional_formatting = {}

        prompt = scenario.create(
            examples=examples,
            additional_formatting=additional_formatting,
        )

        print(prompt)