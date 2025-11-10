"""
train_text_classifier.py

Usage:
    - Put your training data in a CSV file with columns: text,label
      Example rows:
        "Open window to vent smoke","safety"
        "Add 2 cups of water","instruction"

    - Run:
        python train_text_classifier.py --data-file data.csv --output-dir ./saved_model

    - After training, call classifier.predict(text) to get the label.

Notes:
    - This example uses DistilBERT (small & fast). Swap model_name for a bigger model if desired.
    - For real LLM-style few-shot classification or larger models, consider PEFT/LoRA.
"""

import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# -------------------------
# Utility: read CSV file
# -------------------------
def read_csv_as_pairs(csv_path: str) -> List[Tuple[str, str]]:
    import csv

    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # Accept either headers 'text' and 'label', or first two columns
        headers = reader.fieldnames or []
        for r in reader:
            if "text" in r and "label" in r:
                text = r["text"].strip()
                label = r["label"].strip()
            else:
                # fallback: first two columns
                keys = list(r.keys())
                text = r[keys[0]].strip()
                label = r[keys[1]].strip()
            if text == "":
                continue
            pairs.append((text, label))
    return pairs


# -------------------------
# Prepare dataset
# -------------------------


def prepare_dataset(pairs, test_size=0.1, seed=42):
    texts = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]

    # build label <-> id maps
    unique_labels = sorted(set(labels))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    numeric_labels = [label2id[l] for l in labels]

    n_samples = len(texts)
    n_classes = len(unique_labels)

    # Auto-adjust test split
    if n_samples < 15 or int(n_samples * test_size) < n_classes:
        print(f"⚠️ Dataset too small ({n_samples} samples, {n_classes} classes) — using all for training.")
        train_ds = Dataset.from_dict({"text": texts, "label": numeric_labels})
        val_ds = train_ds  # use same data for validation (not ideal, but works)
        return train_ds, val_ds, label2id, id2label

    # If dataset is large enough, use stratified split
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, numeric_labels, test_size=test_size, random_state=seed, stratify=numeric_labels
        )
    except ValueError:
        print("⚠️ Stratified split failed, falling back to random split.")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, numeric_labels, test_size=test_size, random_state=seed, stratify=None
        )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    return train_ds, val_ds, label2id, id2label


# -------------------------
# Tokenize function
# -------------------------
def tokenize_function(examples, tokenizer, max_length=256):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


# -------------------------
# Metrics
# -------------------------
def compute_metrics(pred):
    import evaluate

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    # macro F1 for multi-class
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}


# -------------------------
# Main training flow
# -------------------------
def train(
        data_file: str,
        model_name: str = "distilbert-base-uncased",
        output_dir: str = "./text_classifier",
        epochs: int = 100,
        per_device_batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 256,
        seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Reading CSV...")
    pairs = read_csv_as_pairs(data_file)
    if len(pairs) < 4:
        raise ValueError("Need at least a handful of examples to train. Found: {}".format(len(pairs)))

    train_ds, val_ds, label2id, id2label = prepare_dataset(pairs, test_size=0.1, seed=seed)
    num_labels = len(label2id)
    print(f"Labels ({num_labels}): {label2id}")

    print("Loading tokenizer and model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize datasets
    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)

    # Set format for PyTorch
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_eval=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished. Saving model to", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label maps explicitly
    import json

    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as fh:
        json.dump(label2id, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "id2label.json"), "w", encoding="utf-8") as fh:
        json.dump(id2label, fh, ensure_ascii=False, indent=2)

    print("Model + tokenizer + label maps saved.")
    return output_dir


# -------------------------
# Inference convenience class
# -------------------------
class TextClassifier:
    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        # load label map if present
        import json

        try:
            with open(os.path.join(model_dir, "id2label.json"), "r", encoding="utf-8") as fh:
                self.id2label = json.load(fh)
            # keys in json were strings, convert keys to int
            self.id2label = {int(k): v for k, v in self.id2label.items()}
        except Exception:
            # fallback to model config
            self.id2label = getattr(self.model.config, "id2label", None)
        if self.id2label is None:
            # build default mapping
            self.id2label = {i: str(i) for i in range(self.model.config.num_labels)}

        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, text: str, top_k: int = 1):
        # returns list of (label, score)
        inputs = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # shape: (num_labels,)
        topk_idx = np.argsort(-probs)[:top_k]
        results = [(self.id2label[int(idx)], float(probs[int(idx)])) for idx in topk_idx]
        return results


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a text classifier from csv")
    parser.add_argument("--data-file", default='./input.csv', help="CSV file with columns: text,label")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Pretrained model name")
    parser.add_argument("--output-dir", default="./text_classifier", help="Where to save trained model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    train(
        data_file=args.data_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
