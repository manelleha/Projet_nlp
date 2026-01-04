# src/tl_finetune.py
import os
os.environ["USE_TF"] = "0"  # force Transformers à éviter TensorFlow

import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, f1_score, classification_report


MODEL_NAME_DEFAULT = "bert-base-uncased"
NUM_LABELS = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["clean_text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, digits=4)
    return acc, f1, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME_DEFAULT)
    parser.add_argument("--train_path", type=str, default="../data/twitter_train_clean.csv")
    parser.add_argument("--val_path", type=str, default="../data/twitter_val_clean.csv")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    # option utile si tu veux tester vite / faire low-resource plus tard
    parser.add_argument("--train_frac", type=float, default=1.0, help="Fraction du train (ex: 0.1, 0.2, 0.5, 1.0)")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs("../reports", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    # 1) Load data
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)

    if args.train_frac < 1.0:
        train_df = train_df.sample(frac=args.train_frac, random_state=args.seed).reset_index(drop=True)

    print("Train:", train_df.shape, "| Val:", val_df.shape)

    # 2) Tokenizer + Model (fine-tuning)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS
    )
    model.to(device)

    # 3) Datasets / loaders
    train_ds = TweetDataset(train_df, tokenizer, args.max_length)
    val_ds = TweetDataset(val_df, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 4) Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 5) Training loop
    metrics_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / max(1, (pbar.n + 1)))

        train_loss = running_loss / max(1, len(train_loader))

        # Eval
        val_acc, val_f1, val_report = evaluate(model, val_loader, device)
        print(f"\n[Epoch {epoch}] train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}\n")

        # save report each epoch
        with open("../reports/finetune_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(val_report)

        metrics_rows.append({
            "epoch": epoch,
            "train_frac": args.train_frac,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1
        })

    # 6) Save model + metrics
    out_dir = "../models/bert_finetuned"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("../reports/finetune_metrics.csv", index=False)

    print(f"✅ Modèle sauvegardé dans: {out_dir}")
    print("✅ Metrics sauvegardées dans: ../reports/finetune_metrics.csv")
    print("✅ Report sauvegardé dans: ../reports/finetune_classification_report.txt")


if __name__ == "__main__":
    main()
