from datasets import load_dataset

dataset = load_dataset("yelp_polarity")
print(dataset)
print(dataset["train"][0])
print(dataset["train"][1])

split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]  # 90%
val_dataset   = split_dataset["test"]   # 10%
test_dataset  = dataset["test"]

model_checkpoint = "distilbert-base-uncased"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True)
val_dataset_tokenized   = val_dataset.map(tokenize_function, batched=True)
test_dataset_tokenized  = test_dataset.map(tokenize_function, batched=True)

print(train_dataset_tokenized["input_ids"][0])
print(train_dataset_tokenized["attention_mask"][0])

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2
)

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="distilbert_yelp_sentiment",
    eval_strategy="epoch",    # <= UWAGA: w nowszych wersjach Transformers
    save_strategy="epoch",    # co epokę zapisujemy checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    logging_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

import evaluate
import numpy as np
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
    }

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    compute_metrics=compute_metrics
)

# --- Faza trenowania ---
trainer.train()

# --- Ewaluacja na zbiorze testowym ---
test_results = trainer.evaluate(test_dataset_tokenized)
print("Wyniki na zbiorze testowym:", test_results)

# --- Zapis wytrenowanego modelu i tokenizera ---
# Możesz użyć dowolnej ścieżki. Tutaj zapisujemy do folderu "my_finetuned_model".
save_directory = "my_finetuned_model"

trainer.save_model(save_directory)
# Zapisuje też konfigurację i tokeny do folderu
tokenizer.save_pretrained(save_directory)

print(f"Model oraz tokenizer zostały zapisane w katalogu: {save_directory}")
