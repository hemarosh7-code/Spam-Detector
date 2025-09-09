# export_to_onnx.py
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = "hf_model"
OUT = "model-onnx/model.onnx"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# dummy inputs
enc = tokenizer("This is a sample input", return_tensors="pt", truncation=True, padding=True, max_length=128)
input_ids = enc['input_ids']
attention_mask = enc['attention_mask']

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    OUT,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=12,
    dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"},
                  "attention_mask": {0: "batch_size", 1: "seq_len"},
                  "logits": {0: "batch_size"}}
)
print("Saved ONNX ->", OUT)

