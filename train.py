import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import os
import torch

# --- Constants ---
MODEL_NAME = "distilbert-base-uncased"
# Updated to read from the provided spam.csv file in the current directory
DATA_CSV = "data/spam_data.csv" 
OUTDIR = "hf_model"
MAX_LEN = 128
BATCH = 16
EPOCHS = 3
SEED = 42

# --- Data Loading and Preprocessing ---
try:
    # Read the CSV file without a header and assign column names
    df = pd.read_csv(DATA_CSV, header=None, names=['label', 'text'])

    # Convert the 'spam' and 'ham' string labels to numerical 1s and 0s
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
except FileNotFoundError:
    print(f"Error: The file '{DATA_CSV}' was not found.")
    print("Please ensure 'spam_data.csv' is in the same directory as this script.")
    exit()

# Split the data into training, validation, and test sets.
# The test_size is set to a more robust value to avoid 'ValueError'.
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df.label, random_state=SEED)

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def preprocess(df):
    """Tokenizes and preprocesses the text data for the model."""
    return tokenizer(df['text'].tolist(), truncation=True, padding='max_length', max_length=MAX_LEN)

train_enc = preprocess(train_df)
val_enc = preprocess(val_df)
test_enc = preprocess(test_df)

train_ds = Dataset.from_dict({**train_enc, "labels": train_df['label'].tolist()})
val_ds = Dataset.from_dict({**val_enc, "labels": val_df['label'].tolist()})
test_ds = Dataset.from_dict({**test_enc, "labels": test_df['label'].tolist()})

# --- Model Training ---
# Load the pre-trained DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# --- FIX APPLIED HERE ---
# The Trainer class automatically handles moving the model and data to the GPU.
# Manual device placement is not needed and can cause errors.
# The following lines have been removed:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# print(f"Using device: {device}")


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

# Define a function to compute metrics for evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary', zero_division=0)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Initialize and train the Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
trainer.train()

# Save the trained model and tokenizer
trainer.save_model(OUTDIR)
tokenizer.save_pretrained(OUTDIR)
print("Saved model to", OUTDIR)
