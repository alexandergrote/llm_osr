
from src.ml.classifier.llm.util.request import RequestOutput, RequestInputData
from src.ml.classifier.llm.util.rest import LLM as RestLLM
from src.util.lazy_dict import LazyDict
from src.util.constants import LLMModels, RESTAPI_URLS

Llama3_8B_Remote_HF_Input = RequestInputData.create_hf_llama_request_input(url=RESTAPI_URLS[LLMModels.LLAMA_3_8B_Remote_HF])
Llama3_70B_Remote_HF_Input = RequestInputData.create_hf_llama_request_input(url=RESTAPI_URLS[LLMModels.LLAMA_3_70B_Remote_HF])

GPT3_5_Turbo_Input = RequestInputData.create_openai_request_input(name='gpt-3.5-turbo-0125')

LLM_Mapping = LazyDict({
    LLMModels.OAI_GPT3: (RestLLM, {'name': 'gpt-3.5-turbo', 'request_input': GPT3_5_Turbo_Input, 'request_output_formatter': RequestOutput.from_openai_request}),
    LLMModels.LLAMA_3_8B_Remote_HF: (RestLLM, {'name':'llama-8b', 'request_input': Llama3_8B_Remote_HF_Input, 'request_output_formatter': RequestOutput.from_llama_hf_request}),
    LLMModels.LLAMA_3_70B_Remote_HF: (RestLLM, {'name':'llama-70b', 'request_input': Llama3_70B_Remote_HF_Input, 'request_output_formatter': RequestOutput.from_llama_hf_request})
})


if __name__ == '__main__':

    model = LLM_Mapping[LLMModels.LLAMA_3_8B_Remote_HF]

    prompt = """

    "Is this a greeting? 'hello'"

    You must say either köljafdjklöas if 'yes' or abc if 'no'.
    You must provide reasoning for your answer.

    Responde with the following schema at the end of your response:

    {
        'answer': <your reply>,
        'reasoning': <your reasoning>
    }

"""

    result = model(prompt=prompt)

    print(result)