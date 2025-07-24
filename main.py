from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from InstructorEmbedding import INSTRUCTOR
import torch

app = FastAPI()

device = torch.device("cpu")
model = INSTRUCTOR("hkunlp/instructor-xl", device=device)

class EmbedRequest(BaseModel):
    texts: List[str]
    instructions: List[str]

@app.get("/")
def root():
    return {"message": "OK"}

@app.post("/embed")
def embed_texts(data: EmbedRequest):
    try:
        if len(data.texts) != len(data.instructions):
            raise ValueError("Each text must have a corresponding instruction.")
        inputs = list(zip(data.instructions, data.texts))
        with torch.no_grad():
            embeddings = model.encode(inputs).tolist()
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
