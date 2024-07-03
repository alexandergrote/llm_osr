from pydantic import BaseModel
from typing import Any

from src.util.environment import PydanticEnvironment


env = PydanticEnvironment()


class RequestInput(BaseModel):

    url: str
    headers: dict = dict()
    data: dict = dict()
    prompt_key: str


class RequestOutput(BaseModel):
    
    output_key: str

    @staticmethod
    def get_response_from_single_obj(x: Any, output_key: str, **kwargs) -> str:

        assert isinstance(x, dict), "Response should be a list of dictionaries"
        assert output_key in x, f"{output_key} is required for this function"

        result = x[output_key]

        assert isinstance(result, str), "Response should be a string"

        return result
    
    @staticmethod
    def get_response_from_list_of_obj(x: Any, output_key: str, **kwargs) -> str:

        assert isinstance(x, list), "Response should be a list"
        assert len(x) == 1, "Response should be a list of length 1"
        
        el = x[-1]

        text = RequestOutput.get_response_from_single_obj(x=el, output_key=output_key) 

        return text

    def format_response(self, x: Any, **kwargs) -> str:

        if isinstance(x, list):
            return self.get_response_from_list_of_obj(x=x, output_key=self.output_key, **kwargs)

        if isinstance(x, dict):
            return self.get_response_from_single_obj(x=x, output_key=self.output_key, **kwargs)
        
        raise ValueError("Response should be a list or a dictionary")
    

class LLamaRequestOutput(RequestOutput):

    def format_response(self, x: Any, **kwargs) -> str:
        
        raw_result = super().format_response(x, **kwargs)

        # extract json_str from the answer
        # Extract the JSON part from the text
        start_index = raw_result.rfind('{')
        end_index = raw_result.rfind('}') + 1

        return raw_result[start_index:end_index]


class RequestFactory(BaseModel):

    @classmethod
    def create_hf_request_input(cls, url):

        return RequestInput(
            url=url,
            prompt_key='inputs',
            output_key='generated_text',
            data={'parameters': {
                    "temperature": 0.01, 
                    "max_new_tokens": None,
                    "stop": ["}"],
                    'details': True,
                }},
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {env.hf_token}"}
        )