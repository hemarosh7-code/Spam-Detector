# calibrate.py
import json, numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pandas as pd

MODEL_DIR = "hf_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

df = pd.read_csv("data/scam_data.csv")
# create a validation slice
val_df = df.sample(frac=0.15, random_state=42)
texts = val_df['text'].tolist()
labels = val_df['label'].values

def get_logits(batch_texts):
    logits = []
    with torch.no_grad():
        for i in range(0, len(batch_texts), 32):
            batch = batch_texts[i:i+32]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
            out = model(**enc)
            logits.append(out.logits.cpu().numpy())
    return np.vstack(logits)

logits = get_logits(texts)  # shape (N,2)
# Fit temperature T using simple optimization on validation set
import torch.optim as optim

logits_t = torch.tensor(logits)
labels_t = torch.tensor(labels, dtype=torch.long)
T = torch.nn.Parameter(torch.ones(1) * 1.0)
optimizer = optim.LBFGS([T], lr=0.01, max_iter=200)

def closure():
    optimizer.zero_grad()
    scaled = logits_t / T
    loss = torch.nn.functional.cross_entropy(scaled, labels_t)
    loss.backward()
    return loss

optimizer.step(closure)
temp = float(T.detach().cpu().item())
print("Temperature:", temp)

with open(MODEL_DIR + "/temperature.json", "w") as f:
    json.dump({"temperature": temp}, f)

