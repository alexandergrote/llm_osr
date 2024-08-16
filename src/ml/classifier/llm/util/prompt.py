from pydantic import BaseModel
from pydantic.config import ConfigDict
from typing import Any, List, Dict
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts.few_shot import FewShotPromptTemplate, PromptTemplate
        
        
class PromptCreator(BaseModel):

    text: str
    classes: set
    parser: Any

    _system_msg: str = "You are a classifier for an Open Set Recognition problem."
    _task_prompt: str = "Let's think step by step! Question: {query} -> <your response>"
    _suffix_prompt: str = "Provide your final desired output in the following format: {format_instructions}"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def classes_prompt(self) -> str:

        classes_msg = '\n'.join(self.classes)

        prompt = f"You must reply with one of these classes: \n{classes_msg}\nIf the query does not belong to any of these classes, answer with 'unknown'"

        return prompt
    
    @property
    def prefix_prompt(self) -> str:
        return f"{self._system_msg}\n{self.classes_prompt}\n{self._task_prompt}"
    
    @property
    def suffix_prompt(self) -> str:
        return self._suffix_prompt
    
    @property
    def task_prompt(self) -> str:
        return self._task_prompt

    @staticmethod
    def get_chain_input_field_name() -> str:
        return "query"
    
    def create_zero_shot_prompt(self) -> PromptTemplate:

        template_msg = f"{self.prefix_prompt}\n{self.suffix_prompt}"

        prompt = PromptTemplate(
            template=template_msg,
            input_variables=[PromptCreator.get_chain_input_field_name()],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            validate_template=True
        )

        return prompt

    def create_few_shot_prompt(self, examples: List[Dict[str, str]]) -> FewShotPromptTemplate:

        # assert structure of examples
        for example in examples:
            assert "input" in example
            assert "output" in example

        example_prompt = PromptTemplate(
            template="{input} -> {output}",
            input_variables=["input", "output"]
        )
        
        prompt = FewShotPromptTemplate(
            prefix=self.prefix_prompt,
            suffix=self.suffix_prompt,
            examples=examples,
            example_prompt=example_prompt,
            input_variables=[PromptCreator.get_chain_input_field_name()],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            validate_template=True
        )

        return prompt


if __name__ == '__main__':

    from langchain_core import pydantic_v1

    class Greeting(pydantic_v1.BaseModel):
        text: str = pydantic_v1.Field(description="Text")
        label: str = pydantic_v1.Field(description="Label")

    parser = PydanticOutputParser(pydantic_object=Greeting)

    prompt_creator = PromptCreator(
        text="Some question",
        classes=set(["class1", "class2"]),
        parser=parser
    )
    
    prompt = prompt_creator.create_single_shot_prompt()
    print(prompt.format(query="Go away!"))

    examples = [
        {"input": "Hello", "output": "Greeting"},
        {"input": "Goodbye", "output": "Farewell"}
    ]

    prompt = prompt_creator.create_few_shot_prompt(examples=examples)
    print(prompt.format(query="Go away!"))