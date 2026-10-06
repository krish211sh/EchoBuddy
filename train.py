"""Fine-tune DistilBERT on GoEmotions (multi-label, 28 emotions).
Run:  python train.py
Output: model/ (weights + tokenizer) and results/metrics.json
"""
import json
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MODEL, OUT, THRESH = "distilbert-base-uncased", "model", 0.3

ds = load_dataset("google-research-datasets/go_emotions", "simplified")
names = ds["train"].features["labels"].feature.names
n = len(names)
tok = AutoTokenizer.from_pretrained(MODEL)


def prep(batch):
    enc = tok(batch["text"], truncation=True, max_length=64)
    y = np.zeros((len(batch["text"]), n), dtype=np.float32)
    for i, labels in enumerate(batch["labels"]):
        y[i, labels] = 1.0
    enc["labels"] = y.tolist()
    return enc


ds = ds.map(prep, batched=True, remove_columns=["text", "labels", "id"])
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=n, problem_type="multi_label_classification",
    id2label=dict(enumerate(names)), label2id={l: i for i, l in enumerate(names)})


def metrics(p):
    probs = 1 / (1 + np.exp(-p.predictions))
    pred, true = (probs > THRESH).astype(int), p.label_ids.astype(int)
    return {"f1_micro": f1_score(true, pred, average="micro", zero_division=0),
            "f1_macro": f1_score(true, pred, average="macro", zero_division=0),
            "precision_macro": precision_score(true, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(true, pred, average="macro", zero_division=0)}


args = TrainingArguments(
    output_dir="checkpoints", num_train_epochs=3, learning_rate=5e-5,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="f1_macro", fp16=torch.cuda.is_available(),
    report_to="none", seed=42)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                  eval_dataset=ds["validation"], data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=metrics)
trainer.train()
test = trainer.evaluate(ds["test"], metric_key_prefix="test")
print(test)
trainer.save_model(OUT)
tok.save_pretrained(OUT)
with open("results/metrics.json", "w") as f:
    json.dump(test, f, indent=2)
