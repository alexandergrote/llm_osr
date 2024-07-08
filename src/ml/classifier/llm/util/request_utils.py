from typing import List
from langchain_core import pydantic_v1

from src.util.types import LogProb
from src.util.environment import PydanticEnvironment


env = PydanticEnvironment()



class RequestInputData(pydantic_v1.BaseModel):

    url: str  # url to send the request to
    headers: dict = dict()
    data: dict = dict()
    prompt_key: str  # to add the prompt to the data dynamically


class RequestOutput(pydantic_v1.BaseModel):
    
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


class RequestFactory(pydantic_v1.BaseModel):

    @classmethod
    def create_hf_llama_request_input(cls, url):

        return RequestInputData(
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