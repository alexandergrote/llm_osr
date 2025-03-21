from pydantic import BaseModel, field_validator
from pydantic.config import ConfigDict
from typing import Any, List, Dict
from langchain.output_parsers import PydanticOutputParser
from src.constants import Directory


CHAIN_OF_THOUGHT = "Let's think step by step"
UNKNOWN_CLASS_LABEL = UnknownClassLabel.UNKNOWN_STR.value


def create_prompt(
    template: str, 
    text_to_classify: str, 
    available_classes: List[str], 
    examples: List[Dict[str, str]], 
    use_classes_in_examples: bool, 
    parser: PydanticOutputParser,
    ) -> str:

    instructions = parser.get_format_instructions()

    classes_msg = '\n'.join([el["output"] for el in examples])

    if use_classes_in_examples:
        examples_msg = '\n'.join([f"{example['input']} -> {example['output']}" for example in examples])
    else:
        examples_msg = '\n'.join([f"{example['input']}" for example in examples])


    prompt = template.format(
        examples_msg=examples_msg,
        text=text_to_classify,
        instructions=instructions,
        classes_msg=classes_msg,
        unknown_label=UnknownClassLabel.UNKNOWN_STR.value
    )

    return prompt

if __name__ == '__main__':

    for file in 

    text = "example text"
