import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


from src.ml.classifier.llm.util.mapping import LLM_Mapping as ModelsDict
from src.util.constants import LLMModels as ModelsEnum

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str


class ChatRequestBatch(BaseModel):
    prompts: List[str]


class ChatResponseBatch(BaseModel):
    responses: List[str]


@app.post("/chat", response_model=ChatResponse)
async def generate_text(request: ChatRequest):

    response = llama._call(request.prompt)

    return ChatResponse(response=response)


@app.post("/chat/batch", response_model=ChatResponseBatch)
async def generate_text_batch(request: ChatRequestBatch):

    responses = llama.custom_model.batch(request.prompts)

    return ChatResponseBatch(responses=responses)


if __name__ == "__main__":

    llama = ModelsDict[ModelsEnum.LLAMA_3B]

    llama.setup()

    uvicorn.run("src.ml.classifier.service:app", host="127.0.0.1", port=1234, workers=10)


