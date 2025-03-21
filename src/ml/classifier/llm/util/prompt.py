from typing import List, Dict, Optional
from langchain.output_parsers import PydanticOutputParser
from src.util.constants import Directory, UnknownClassLabel
from src.ml.classifier.llm.util.prediction import Prediction


CHAIN_OF_THOUGHT = "Let's think step by step"
UNKNOWN_CLASS_LABEL = UnknownClassLabel.UNKNOWN_STR.value


def create_prompt(
    template: str, 
    text_to_classify: str, 
    examples: List[Dict[str, str]], 
    parser: PydanticOutputParser,
    use_classes_in_examples: bool = True, 
    additional_formatting: Optional[Dict[str, str]] = None
    ) -> str:

    if additional_formatting is None:
        additional_formatting = {}

    if not hasattr(parser.pydantic_object, "valid_labels"):
        raise ValueError("Pydantic model must have a 'valid_labels' attribute.")

    if use_classes_in_examples:
        examples_msg = '\n'.join([f"{example['input']} -> {example['output']}" for example in examples])
    else:
        examples_msg = '\n'.join([f"{example['input']}" for example in examples])


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
        {"input": "It's raining heavily.", "output": "weather"},                                                                               
        {"input": "The final score was 3-1.", "output": "sports"},                                                                             
        {"input": "The president signed a new law.", "output": "politics"}                                                                     
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
            valid_labels = ['outlier', 'inlier']

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
            additional_formatting=additional_formatting
        )

        print('-'*25)
        print(prompt)
