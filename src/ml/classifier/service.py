import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel


from src.ml.classifier.llm.util.mapping import LLM_Mapping as ModelsDict
from src.util.constants import LLMModels as ModelsEnum

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def generate_text(request: ChatRequest):

    response = model._call(request.prompt)

    return ChatResponse(response=response)


if __name__ == "__main__":

    model = ModelsDict[ModelsEnum.LLAMA_3B_Local]

    if hasattr(model, 'setup'):
        model.setup()

    uvicorn.run("src.ml.classifier.service:app", host="127.0.0.1", port=1234, workers=10)


