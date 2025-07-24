from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import gc
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8000",                   
    "https://nexa-ai-2hd6.onrender.com",      
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("onnx/model.onnx", providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("onnx/")

class EmbedRequest(BaseModel):
    texts: List[str]

@app.get("/")
def root():
    return {"message": "OK"}

@app.post("/embed")
def embed(req: EmbedRequest):
    texts = req.texts[:10] 

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128, 
        return_tensors="np"
    )

    ort_inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }
    outputs = session.run(None, ort_inputs)
    last_hidden_state = outputs[0]

    mask = np.expand_dims(encoded["attention_mask"], -1).astype(np.float32)
    summed = np.sum(last_hidden_state * mask, axis=1)
    norm = np.clip(mask.sum(1), a_min=1e-9, a_max=None)
    mean_pooled = summed / norm

    norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    mean_pooled = mean_pooled / np.clip(norms, a_min=1e-9, a_max=None)

    del encoded, outputs, last_hidden_state, mask, summed, norm, norms
    gc.collect()

    return {"embeddings": mean_pooled.tolist()}
