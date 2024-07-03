
from src.ml.classifier.llm.api import OpenAIWrapper, HFWrapper, Llama
from src.ml.classifier.llm.util.langchain_wrappers import LangchainWrapper, LangchainWrapperRemote
from src.ml.classifier.llm.util.request_utils import RequestInput, RequestOutput, LLamaRequestOutput, RequestFactory
from src.util.lazy_dict import LazyDict
from src.util.constants import LLMModels


Localhost_Remote_Input = RequestInput(
    url='http://localhost:1234/chat',
    prompt_key='prompt',
    output_key='response',
)

Localhost_Remote_Output = RequestOutput(output_key='response')

Llama3_8B_Remote_HF_Input = RequestFactory.create_hf_request_input(url='https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8b-Instruct')
Llama3_70B_Remote_HF_Input = RequestFactory.create_hf_request_input(url='https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70b-Instruct')

Llama3_Remote_HF_Output = LLamaRequestOutput(output_key='generated_text')


LLM_Mapping = LazyDict({
    LLMModels.OAI_GPT4: (LangchainWrapper, {'name': 'gpt-3.5', 'custom_model': OpenAIWrapper(name='gpt-3.5-turbo-0125')}),
    LLMModels.OAI_GPT3: (LangchainWrapper, {'name':'gpt-3.5', 'custom_model': OpenAIWrapper(name='gpt-3.5-turbo-0125')}),
    LLMModels.OAI_GPT2: (LangchainWrapper, {'name':'gpt-2', 'custom_model': HFWrapper(name='gpt2')}),
    LLMModels.LLAMA_3B_Local: (LangchainWrapper, {'name':'llama-3b', 'custom_model': Llama(name='llama')}),
    LLMModels.LLAMA_3_8B_Remote: (LangchainWrapperRemote, {'name':'llama-3b', 'request_input': Localhost_Remote_Input, 'request_output': Localhost_Remote_Output}),
    LLMModels.LLAMA_3_8B_Remote_HF: (LangchainWrapperRemote, {'name':'llama-3b', 'request_input': Llama3_8B_Remote_HF_Input, 'request_output': Llama3_Remote_HF_Output}),
    LLMModels.LLAMA_3_70B_Remote_HF: (LangchainWrapperRemote, {'name':'llama-3b', 'request_input': Llama3_70B_Remote_HF_Input, 'request_output': Llama3_Remote_HF_Output})
})


if __name__ == '__main__':

    model = LLM_Mapping[LLMModels.LLAMA_3_8B_Remote_HF]

    prompt = """

    "Is this a greeting? 'hello'"

    You must say either 'yes' or 'no'.
    You must provide reasoning for your answer.

    Responde with the following schema at the end of your response:

    {
        'answer': <your reply>,
        'reasoning': <your reasoning>
    }

"""

    result = model._call(prompt=prompt)

    print(result)