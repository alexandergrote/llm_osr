from typing import List, Optional, Callable
from pydantic import BaseModel

from src.util.types import LogProb
from src.util.environment import PydanticEnvironment


env = PydanticEnvironment()


def add_prompt_to_data_hf(*, data: dict, prompt_value: str, **kwargs) -> dict:
    """
    Add the prompt to the data dictionary
    """

    data['inputs'] = prompt_value

    return data


def add_prompt_to_data_openai(*, data: dict, prompt_value: str, **kwargs) -> dict:
    """
    Add the prompt to the data dictionary
    """

    data['messages'] = [{"role": "user", "content": prompt_value}]

    return data



class RequestInputData(BaseModel):

    url: str  # url to send the request to
    headers: dict = dict()
    data: dict = dict()
    data_modifying_function: Optional[Callable] = None  # function to modify the data before sending the request

    @classmethod
    def create_openai_request_input(cls, name) -> "RequestInputData":
        
        return RequestInputData(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {env.openai_key}"
                },
            data={
                "model": name,
                "temperature": 0,
                "max_tokens": 1000,
                "stop": None,
                "logprobs": True,
                "top_logprobs": 0,
                "tools": None
            },
            data_modifying_function=add_prompt_to_data_openai
        )
        

    @classmethod
    def create_hf_llama_request_input(cls, url) -> "RequestInputData":

        return RequestInputData(
            url=url,
            data={'parameters': {
                    "temperature": 0.01, 
                    "max_new_tokens": 1000,
                    "stop": ["}"],
                    'details': True,
                }},
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {env.hf_token}"},
            data_modifying_function=add_prompt_to_data_hf
        )

class RequestOutput(BaseModel):
    
    text: str
    logprobas: List[LogProb]
    

    @classmethod
    def from_llama_hf_request(cls, x: List[dict], **kwargs) -> "RequestOutput":

        assert isinstance(x, list), "Response should be a list"
        assert len(x) == 1, "Response should be a list of length 1"
        
        el = x[-1]

        assert 'generated_text' in el, "Response should contain 'generated_text' key"
        assert 'details' in el, "Response should contain 'details' key"

        generated_text = el['generated_text']
        details = el['details']

        # extract logproba_output from the answer
        assert 'tokens' in details, "Details should contain 'tokens' key"
        
        tokens = details['tokens']

        assert isinstance(tokens, list), "Tokens should be a list"
        assert len(tokens) > 0, "Tokens should not be empty"

        for token in tokens:
            assert 'text' in token, "Token should contain 'text' key"
            assert 'logprob' in token, "Token should contain 'logprob' key"

        # extract json_str from the answer
        # Extract the JSON part from the text
        start_index = generated_text.rfind('{')
        end_index = generated_text.rfind('}') + 1
        text_response = generated_text[start_index:end_index]

        # extract logproba_output from the answer
        logprobas = [LogProb(text=el['text'], logprob=el['logprob']) for el in tokens]


        return cls(
            text=text_response,
            logprobas=logprobas
        )

    @classmethod
    def from_openai_request(cls, x: dict, **kwargs) -> "RequestOutput":
        
        assert 'choices' in x, 'Completion does not have choices'
        assert len(x['choices']) == 1, 'Completion does not have one choice'
        assert 'message' in x['choices'][0], 'Completion choice does not have message'
        assert 'content' in x['choices'][0]['message'] , 'Completion choice message does not have content'

        text = x['choices'][0]['message']['content']
        logprob_list = [LogProb(text=token['token'], logprob=token['logprob']) for token in x['choices'][0]['logprobs']['content']]

        return cls(
            text=text,
            logprobas=logprob_list
        )