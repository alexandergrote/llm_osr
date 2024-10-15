from pydantic import BaseModel, field_validator
from pydantic.config import ConfigDict
from typing import Any, List, Dict
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts.few_shot import FewShotPromptTemplate, PromptTemplate
        
        
class FewshotPromptCreator(BaseModel):

    parser: Any

    prefix_prompt: str 
    suffix_prompt: str

    text_to_classify: str
    examples: List[Dict[str, str]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def get_chain_input_field_name() -> str:
        return "query"
    
    @staticmethod
    def get_format_instruction_field() -> str:
        return "format_instructions"

    @field_validator("prefix_prompt")
    @classmethod
    def validate_prefix_prompt(cls, v):

        field = "{" + cls.get_chain_input_field_name() + "}"
        assert field in v, f"prefix_prompt must contain the field {field}"

        return v
    
    @field_validator("suffix_prompt")
    @classmethod
    def validate_suffix(cls, v):

        field = "{" + cls.get_format_instruction_field() + "}"
        assert field in v, f"suffix_prompt must contain the field {field}"

        return v

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, v):

        for example in v:
            assert "input" in example
            assert "output" in example

        return v
    
    @field_validator("parser")
    @classmethod
    def validate_parser(cls, v):

        assert hasattr(v, "get_format_instructions")

        return v
    

    def create(self) -> str:

        instructions = self.parser.get_format_instructions()

        example_prompt = PromptTemplate(
            template="{input} -> {output}",
            input_variables=["input", "output"]
        )
        
        prompt = FewShotPromptTemplate(
            prefix=self.prefix_prompt,
            suffix=self.suffix_prompt,
            examples=self.examples,
            example_prompt=example_prompt,
            input_variables=[FewshotPromptCreator.get_chain_input_field_name()],
            partial_variables={FewshotPromptCreator.get_format_instruction_field(): instructions},
            validate_template=True
        )

        return prompt.format(**{FewshotPromptCreator.get_chain_input_field_name(): self.text_to_classify}) 



if __name__ == '__main__':

    from langchain_core import pydantic_v1

    class Greeting(pydantic_v1.BaseModel):
        text: str = pydantic_v1.Field(description="Text")
        label: str = pydantic_v1.Field(description="Label")

    parser = PydanticOutputParser(pydantic_object=Greeting)

    examples = [
        {"input": "Hello", "output": "Greeting"},
        {"input": "Goodbye", "output": "Farewell"}
    ]

    prompt_creator = FewshotPromptCreator(
        examples=examples,
        text_to_classify="Hola",
        prefix_prompt=f"You are a classifier. {{{FewshotPromptCreator.get_chain_input_field_name()}}} -> <your response>",
        suffix_prompt=f"Use this format for your response: {{{FewshotPromptCreator.get_format_instruction_field()}}}",
        parser=parser
    )

    prompt = prompt_creator.create()
    print(prompt)