from typing import List, Callable, Optional
from pydantic import BaseModel

from src.util.types import LogProb
from src.util.environment import PydanticEnvironment


env = PydanticEnvironment.from_environment()


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


def get_prompt_from_data_hf(*, data: dict, **kwargs) -> str:

    return data["data"]['inputs']


def get_prompt_from_data_openai(*, data: dict, **kwargs) -> str:

    return data["data"]['messages'][0]['content']


class RequestInput(BaseModel):

    data: dict

    @staticmethod
    def get_request_dict(url: str, headers: dict, prompt: str, data: dict, data_modifying_function: Callable ) -> dict:

        request_dict = {
            "url": url,
            "headers": headers,
        }

        request_dict['data'] = data_modifying_function(
            data=data,
            prompt_value=prompt
        )

        return request_dict

    @classmethod
    def create_openai_request_input(cls, prompt: str, url: str = "https://api.openai.com/v1/chat/completions", payload: Optional[dict] = None ) -> "RequestInput":
        
        if payload is None:
            payload = {}
        
        assert 'name' in payload, "Payload must contain a 'name' key"

        headers={
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {env.openai_key}"
        }

        final_payload = {
            "temperature": 0,
            "max_tokens": 1000,
            "stop": None,
            "logprobs": True,
            "top_logprobs": 0,
            "tools": None
        }

        for k, v in payload.items():
            final_payload[k] = v    

        request_dict = RequestInput.get_request_dict(
            url=url,
            headers=headers,
            prompt=prompt,
            data=final_payload,
            data_modifying_function=add_prompt_to_data_openai
        )    

        return RequestInput(
            data=request_dict
        )
        
    @classmethod
    def create_hf_llama_request_input(cls, prompt: str, url, payload: Optional[dict] = None) -> "RequestInput":

        if payload is None:
            payload = {}

        headers={"Content-Type": "application/json", "Authorization": f"Bearer {env.hf_token}"}

        final_payload = {
            "temperature": 0.01, 
            "max_new_tokens": 1000,
            "stop": ["}"],
            'details': True,
            "return_full_text": False,
        }

        for k, v in payload.items():
            final_payload[k] = v 

        request_dict = RequestInput.get_request_dict(
            url=url,
            headers=headers,
            prompt=prompt,
            data={'parameters': final_payload},
            data_modifying_function=add_prompt_to_data_hf
        )    

        return RequestInput(
            data=request_dict
        )

    @classmethod
    def create_groq_request_input(cls, prompt: str, url: str = 'https://api.groq.com/openai/v1/chat/completions', payload: Optional[dict] = None) -> "RequestInput":

        if payload is None:
            payload = {}
            
        assert 'model' in payload, "Payload must contain a 'name' key"

        headers={
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {env.groq_key}"
        }

        final_payload = {
            "temperature": 0,
            "max_tokens": 1000,
        }

        for k, v in payload.items():
            final_payload[k] = v        

        request_dict = RequestInput.get_request_dict(
            url=url,
            headers=headers,
            prompt=prompt,
            data=final_payload,
            data_modifying_function=add_prompt_to_data_openai
        )    

        return RequestInput(
            data=request_dict
        )

    @classmethod
    def create_sambanova_request_input(cls, prompt: str, url: str = 'https://api.sambanova.ai/v1/chat/completions', payload: Optional[dict] = None) -> "RequestInput":

        if payload is None:
            payload = {}
            
        assert 'model' in payload, "Payload must contain a 'model' key"

        headers={
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {env.sambanova_key}"
        }

        final_payload = {
            "temperature": 0,
            "max_tokens": 1000,
        }

        for k, v in payload.items():
            final_payload[k] = v    

        request_dict = RequestInput.get_request_dict(
            url=url,
            headers=headers,
            prompt=prompt,
            data=final_payload,
            data_modifying_function=add_prompt_to_data_openai
        )    

        return RequestInput(
            data=request_dict
        )
    
    @classmethod
    def create_openrouter_request_input(cls, prompt: str, url: str = 'https://openrouter.ai/api/v1/chat/completions', payload: Optional[dict] = None) -> "RequestInput":

        if payload is None:
            payload = {}


        assert 'model' in payload, "Payload must contain a 'model' key"

        headers={
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {env.openrouter_key}"
        }

        final_payload = {
            "temperature": 0,
            "max_tokens": 1000,
        }

        for k, v in payload.items():
            final_payload[k] = v    

        request_dict = RequestInput.get_request_dict(
            url=url,
            headers=headers,
            prompt=prompt,
            data=final_payload,
            data_modifying_function=add_prompt_to_data_openai
        )    

        return RequestInput(
            data=request_dict
        )

    @classmethod
    def create_cerebras_request_input(cls, prompt: str, url: str = 'https://api.cerebras.ai/v1/chat/completions', payload: Optional[dict] = None) -> "RequestInput":

        if payload is None:
            payload = {}

        assert 'model' in payload, "Payload must contain a 'model' key"

        headers={
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {env.cerebras_key}"
        }

        final_payload = {
            "temperature": 0,
        }

        for k, v in payload.items():
            final_payload[k] = v    

        request_dict = RequestInput.get_request_dict(
            url=url,
            headers=headers,
            prompt=prompt,
            data=final_payload,
            data_modifying_function=add_prompt_to_data_openai
        )    

        return RequestInput(
            data=request_dict
        )

    @staticmethod
    def get_prompt_from_hf_data(dictionary):

        return get_prompt_from_data_hf(
            data=dictionary
        )
    
    @staticmethod
    def get_prompt_from_openai_data(dictionary):

        return get_prompt_from_data_openai(
            data=dictionary
        )


class RequestOutput(BaseModel):
    
    text: str
    logprobas: List[LogProb]
    num_tokens: int
    error: bool = False
    
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
        # needed because otherwise pydantic cannot parse the json
        start_index = generated_text.rfind('{')
        end_index = generated_text.rfind('}') + 1
        text_response = generated_text[start_index:end_index]

        # extract logproba_output from the answer
        logprobas = [LogProb(text=el['text'], logprob=el['logprob']) for el in tokens]

        return cls(
            text=text_response,
            logprobas=logprobas,
            num_tokens=len(tokens),
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
    
    @classmethod
    def from_groq_request(cls, x: dict, **kwargs) -> "RequestOutput":
        
        assert 'choices' in x, 'Completion does not have choices'
        assert len(x['choices']) == 1, 'Completion does not have one choice'
        assert 'message' in x['choices'][0], 'Completion choice does not have message'
        assert 'content' in x['choices'][0]['message'] , 'Completion choice message does not have content'

        text = x['choices'][0]['message']['content']
        logprob_list = [LogProb(text='{"label": "LogProb not supported by Groq"}', logprob=0)]

        return cls(
            text=text,
            logprobas=logprob_list,
            num_tokens=x["usage"]["completion_tokens"]
        )
    
    @classmethod
    def from_sambanova_request(cls, x: dict, **kwargs) -> "RequestOutput":

        assert 'choices' in x, 'Completion does not have choices'
        assert len(x['choices']) == 1, 'Completion does not have one choice'
        assert 'message' in x['choices'][0], 'Completion choice does not have message'
        assert 'content' in x['choices'][0]['message'] , 'Completion choice message does not have content'

        text = x['choices'][0]['message']['content']
        logprob_list = [LogProb(text='{"label": "LogProb not supported by SambaNova"}', logprob=0)]

        return cls(
            text=text,
            logprobas=logprob_list,
            num_tokens=-1
        )
    
    @classmethod
    def from_openrouter_request(cls, x: dict, **kwargs) -> "RequestOutput":

        assert 'choices' in x, 'Completion does not have choices'
        assert len(x['choices']) == 1, 'Completion does not have one choice'
        assert 'message' in x['choices'][0], 'Completion choice does not have message'
        assert 'content' in x['choices'][0]['message'] , 'Completion choice message does not have content'

        text = x['choices'][0]['message']['content']
        logprob_list = [LogProb(text='{"label": "LogProb not supported by OpenRouter"}', logprob=0)]

        return cls(
            text=text,
            logprobas=logprob_list,
            num_tokens=-1
        )
    
    @classmethod
    def from_cerebras_request(cls, x: dict, **kwargs) -> "RequestOutput":

        assert 'choices' in x, 'Completion does not have choices'
        assert len(x['choices']) == 1, 'Completion does not have one choice'
        assert 'message' in x['choices'][0], 'Completion choice does not have message'
        assert 'content' in x['choices'][0]['message'] , 'Completion choice message does not have content'

        text = x['choices'][0]['message']['content']
        logprob_list = [LogProb(text='{"label": "LogProb not supported by Cerebras"}', logprob=0)]

        return cls(
            text=text,
            logprobas=logprob_list,
            num_tokens=x["usage"]["completion_tokens"]
        )