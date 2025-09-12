import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import os

MODEL_NAME = "distilbert-base-uncased"
DATA_CSV = "data/scam_data.csv"
OUTDIR = "hf_model"
MAX_LEN = 128
BATCH = 16
EPOCHS = 3
SEED = 42

# --- FIX/IMPROVEMENT APPLIED HERE ---
# Check if the data directory and CSV file exist.
# If not, create dummy data for a seamless execution.
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists(DATA_CSV):
    print(f"File '{DATA_CSV}' not found. Creating dummy data for demonstration.")
    texts = [
        "Congratulations! You've won a million dollars! Click here to claim your prize.",
        "Your account has been locked. Please verify your details at this link.",
        "Hello, how are you? Would you like to meet up?",
        "Important: We have detected suspicious activity. Please click this link to secure your account.",
        "Hey, let's grab coffee this weekend.",
        "You have received a new payment of $500. Check your wallet balance.",
        "Your Amazon order has shipped.",
        "Special offer for you! 90% off for a limited time.",
    ]
    labels = [1, 1, 0, 1, 0, 1, 0, 1]
    df = pd.DataFrame({'text': texts, 'label': labels})
    df.to_csv(DATA_CSV, index=False)

df = pd.read_csv(DATA_CSV)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)

# Perform a second split to get a separate validation set.
# This gives you a train, validation, and test split.
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
