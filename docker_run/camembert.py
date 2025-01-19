import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import optuna
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

# Load train.csv
print("Loading train.csv ...")
df = pd.read_csv("train.csv")

# Encode target labels if they're strings
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

# Split into train (90%) and validation (10%)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Text"],
    df["Label"],
    test_size=0.1,
    random_state=42,
    stratify=df["Label"]
)

# Oversample the minority class to improve coverage
ros = RandomOverSampler(random_state=42)
train_texts_resampled, train_labels_resampled = ros.fit_resample(
    pd.DataFrame(train_texts, columns=["Text"]),
    train_labels
)

train_texts_resampled = train_texts_resampled["Text"].tolist()
train_labels_resampled = train_labels_resampled.tolist()

print(f"Original training distribution: {Counter(train_labels)}")
print(f"Resampled training distribution: {Counter(train_labels_resampled)}")

# Initialize tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-large")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512  # Increased to 512 for more context
    )

# Crate train and validation datasets
train_dataset = Dataset.from_dict({
    "text": train_texts_resampled,
    "label": train_labels_resampled
}).map(tokenize_function, batched=True).remove_columns(["text"])

val_dataset = Dataset.from_dict({
    "text": val_texts,
    "label": val_labels
}).map(tokenize_function, batched=True).remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# Training with class weights
def model_init():
    """Create a fresh model for each hyperparameter trial."""
    return CamembertForSequenceClassification.from_pretrained(
        "camembert/camembert-large",
        num_labels=len(le.classes_)
    )

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(next(model.parameters()).device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Class weights for imbalanced classes
counts = Counter(train_labels_resampled)
num_labels = len(le.classes_)
total_samples = len(train_labels_resampled)

class_weights = []
for label_idx in range(num_labels):
    class_count = counts[label_idx]
    weight = total_samples / (num_labels * class_count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class Weights:", class_weights)

# Base training arguments
base_training_args = TrainingArguments(
    output_dir="./camembert_weighted",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    logging_dir="./logs",
    per_device_eval_batch_size=8,
    save_total_limit=1,  # Only keep 1 best model
    seed=42
)

# Metrics definition
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

# Optuna hyperparameter search space
def hp_space_optuna(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 12),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
    }

trainer_for_search = WeightedTrainer(
    model_init=model_init,
    args=base_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    class_weights=class_weights,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

print("Starting hyperparameter search (Optuna) focusing on accuracy ...")
best_run = trainer_for_search.hyperparameter_search(
    direction="maximize",
    hp_space=hp_space_optuna,
    backend="optuna",
    n_trials=1
)

print("Best hyperparameters found:", best_run.hyperparameters)

# Best training arguments
final_training_args = TrainingArguments(
    output_dir="./camembert_weighted",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=1,
    seed=42,
    learning_rate=best_run.hyperparameters["learning_rate"],
    num_train_epochs=int(best_run.hyperparameters["num_train_epochs"]),
    per_device_train_batch_size=int(best_run.hyperparameters["per_device_train_batch_size"]),
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_first_step=True,
    report_to="none",
)

final_trainer = WeightedTrainer(
    model_init=model_init,
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    class_weights=class_weights,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

final_trainer.train()

# Validation step
val_results = final_trainer.evaluate(eval_dataset=val_dataset)
print("\nValidation set results:", val_results)

# Confusion matrix
predictions = final_trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)
print("\nClassification report:")
print(classification_report(val_labels, preds, target_names=le.classes_))

# Labeling test data
test_df = pd.read_csv("test.csv")
test_texts = list(test_df["Text"])
test_dataset = Dataset.from_dict({"text": test_texts}).map(tokenize_function, batched=True)
test_dataset = test_dataset.remove_columns(["text"])
test_dataset.set_format("torch")

test_preds = final_trainer.predict(test_dataset)
test_labels_idx = np.argmax(test_preds.predictions, axis=1)
test_labels_str = le.inverse_transform(test_labels_idx)

test_df["Label"] = test_labels_str
test_df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv.")