
from src.ml.classifier.llm.api import OpenAIWrapper, HFWrapper, Llama
from src.ml.classifier.llm.api.base import LangchainWrapper, LangchainWrapperRemote
from src.util.lazy_dict import LazyDict
from src.util.constants import LLMModels


LLM_Mapping = LazyDict({
    LLMModels.OAI_GPT4: (LangchainWrapper, {'name': 'gpt-3.5', 'custom_model': OpenAIWrapper(name='gpt-3.5-turbo-0125')}),
    LLMModels.OAI_GPT3: (LangchainWrapper, {'name':'gpt-3.5', 'custom_model': OpenAIWrapper(name='gpt-3.5-turbo-0125')}),
    LLMModels.OAI_GPT2: (LangchainWrapper, {'name':'gpt-2', 'custom_model': HFWrapper(name='gpt2')}),
    LLMModels.LLAMA_3B: (LangchainWrapper, {'name':'llama-3b', 'custom_model': Llama(name='llama')}),
    LLMModels.LLAMA_3B_Remote: (LangchainWrapperRemote, {'name':'llama-3b', 'url': 'http://localhost:1234/chat'})
})