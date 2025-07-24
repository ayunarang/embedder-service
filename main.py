from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import gc

app = FastAPI()

# Load model and tokenizer once
session = ort.InferenceSession("onnx/model.onnx", providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
def embed(req: EmbedRequest):
    # Limit batch size for lower RAM footprint
    texts = req.texts[:10]

    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,  
        return_tensors="np"
    )

    # Prepare ONNX input
    ort_inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }

    # Inference
    outputs = session.run(None, ort_inputs)
    last_hidden_state = outputs[0]

    # Mean Pooling
    mask = np.expand_dims(encoded["attention_mask"], -1).astype(np.float32)
    summed = np.sum(last_hidden_state * mask, axis=1)
    norm = np.clip(mask.sum(1), a_min=1e-9, a_max=None)
    mean_pooled = summed / norm

    # Explicit cleanup
    del encoded, outputs, last_hidden_state, mask, summed, norm
    gc.collect()

    return {"embeddings": mean_pooled.tolist()}
