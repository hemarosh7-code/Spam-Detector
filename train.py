# train.py
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

MODEL_NAME = "distilbert-base-uncased"
DATA_CSV = "data/scam_data.csv"
OUTDIR = "hf_model"
MAX_LEN = 128
BATCH = 16
EPOCHS = 3
SEED = 42

df = pd.read_csv(DATA_CSV)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df.label, random_state=SEED)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def preprocess(df):
    return tokenizer(df['text'].tolist(), truncation=True, padding='max_length', max_length=MAX_LEN)

train_enc = preprocess(train_df)
val_enc = preprocess(val_df)
test_enc = preprocess(test_df)

train_ds = Dataset.from_dict({**train_enc, "labels": train_df['label'].tolist()})
val_ds = Dataset.from_dict({**val_enc, "labels": val_df['label'].tolist()})
test_ds = Dataset.from_dict({**test_enc, "labels": test_df['label'].tolist()})

model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir=OUTDIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary', zero_division=0)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(OUTDIR)
tokenizer.save_pretrained(OUTDIR)
print("Saved model to", OUTDIR)

