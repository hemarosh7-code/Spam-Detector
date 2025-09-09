# app.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast
import onnxruntime as ort
import numpy as np
import os, json

MODEL_ONNX = "model-onnx/model_int8.onnx"   # choose quantized ONNX
TOKENIZER_DIR = "hf_model"
API_KEY = os.environ.get("SCAM_API_KEY", "CHANGE_ME")

tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_DIR)
sess = ort.InferenceSession(MODEL_ONNX, providers=['CPUExecutionProvider'])

# load temperature if exists
temp_file = os.path.join(TOKENIZER_DIR, "temperature.json")
temperature = 1.0
if os.path.exists(temp_file):
    temperature = json.load(open(temp_file))["temperature"]

app = FastAPI(title="Scam Score API")

class Item(BaseModel):
    text: str

def predict_proba(texts):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="np")
    inputs = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
    logits = sess.run(None, inputs)[0]  # shape (N,2)
    logits = logits / temperature
    # compute softmax prob of label=1
    max_l = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_l)
    probs = exp[:,1] / exp.sum(axis=1)
    return probs

BANDS = [(0,39,"Safe"),(40,69,"Suspicious"),(70,100,"Likely Scam")]

@app.post("/score")
def score(item: Item, x_api_key: str = Header(None)):
    # simple API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    prob = float(predict_proba([item.text])[0])
    score = int(round(prob * 100))
    band = next(name for lo,hi,name in BANDS if lo <= score <= hi)
    return {"score": score, "prob": prob, "band": band}

