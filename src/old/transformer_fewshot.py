import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
os.environ["USE_TF"] = "0"
import os
os.environ["USE_TF"] = "0"  # forcer PyTorch only

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import os

FRACTIONS = [0.05, 0.10, 0.20, 0.50, 1.00]
MODEL_NAME = "distilbert-base-uncased"

# Charger dataset propre
df = pd.read_csv("../data/processed/twitter_clean.csv")

X = df["clean_text"]
y = df["label"]

# Train / test identique à la baseline
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
os.makedirs("../reports", exist_ok=True)
os.makedirs("../results", exist_ok=True)

results = []

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

for frac in FRACTIONS:
    print(f"\n=== TRANSFORMER avec {int(frac*100)}% des données ===")

    # sous-échantillonnage few-shot
    if frac < 1.0:
        X_train_frac, _, y_train_frac, _ = train_test_split(
            X_train, y_train,
            train_size=frac,
            random_state=42,
            stratify=y_train
        )
    else:
        X_train_frac, y_train_frac = X_train, y_train

    train_df = pd.DataFrame({"text": X_train_frac, "label": y_train_frac})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_ds = Dataset.from_pandas(train_df)
    test_ds  = Dataset.from_pandas(test_df)

    # tokenisation
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)

    # on enlève la colonne texte brute (+ index éventuel)
    cols_to_remove = [c for c in ["text", "__index_level_0__"] if c in train_ds.column_names]
    train_ds = train_ds.remove_columns(cols_to_remove)
    cols_to_remove = [c for c in ["text", "__index_level_0__"] if c in test_ds.column_names]
    test_ds = test_ds.remove_columns(cols_to_remove)

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4
    )

    # ⚠️ Arguments simplifiés pour être compatibles avec ta version de transformers
    args = TrainingArguments(
        output_dir=f"../results/frac_{int(frac*100)}",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16
        # pas de evaluation_strategy / save_strategy ici
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Metrics:", metrics)

    results.append({
        "fraction": frac,
        "n_train": len(X_train_frac),
        "accuracy": metrics["eval_accuracy"],
        "f1": metrics["eval_f1"]
    })

pd.DataFrame(results).to_csv("../reports/fewshot_transformer.csv", index=False)
print("\n✅ Résultats Transformers enregistrés dans ../reports/fewshot_transformer.csv")
