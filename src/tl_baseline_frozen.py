# src/tl_baseline_frozen.py
import os
os.environ["USE_TF"] = "0"  # Force Transformers à éviter TensorFlow

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import AutoTokenizer, AutoModel

# Dataset en anglais -> modèle anglais
MODEL_NAME = "bert-base-uncased"

BATCH_SIZE = 64
MAX_LENGTH = 128


def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean pooling en ignorant le padding.
    last_hidden_state: (batch, seq_len, hidden)
    attention_mask: (batch, seq_len)
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def compute_embeddings(texts, tokenizer, model, device, desc="Embedding"):
    """
    Retourne un array numpy (n_samples, hidden_size)
    en utilisant mean pooling (plus stable que CLS sur certains datasets).
    """
    dataset = TensorDataset(torch.arange(len(texts)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    model.to(device)

    all_emb = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            idx = batch[0].numpy()
            batch_texts = [texts[i] for i in idx]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            pooled = mean_pooling(outputs.last_hidden_state, enc["attention_mask"])
            all_emb.append(pooled.cpu().numpy())

    return np.vstack(all_emb)


def main():
    # 1) Charger données clean
    train = pd.read_csv("../data/twitter_train_clean.csv")
    val   = pd.read_csv("../data/twitter_val_clean.csv")

    # MODIFICATION UTILISATEUR: 10% du dataset seulement pour comparaison équitable avec Finetuning
    # (Le finetuning a tourné sur 10% par manque de temps, donc on aligne la baseline)
    train = train.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print(f"📉 SUB-SAMPLING ACTIVÉ: Train size réduit à {len(train)} (10%)")

    X_train = train["clean_text"].astype(str).tolist()
    y_train = train["label"].astype(int).values

    X_val = val["clean_text"].astype(str).tolist()
    y_val = val["label"].astype(int).values

    print("Train:", train.shape, "Val:", val.shape)

    # 2) Charger modèle pré-entraîné
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # 3) Geler les poids
    for p in model.parameters():
        p.requires_grad = False

    # 4) Embeddings train/val
    print("\n== Embeddings TRAIN ==")
    X_train_emb = compute_embeddings(X_train, tokenizer, model, device, desc="Train embeddings")

    print("\n== Embeddings VAL ==")
    X_val_emb = compute_embeddings(X_val, tokenizer, model, device, desc="Val embeddings")

    # 5) Classifieur simple
    clf = LogisticRegression(max_iter=3000, n_jobs=1)
    clf.fit(X_train_emb, y_train)

    # 6) Évaluation
    y_pred = clf.predict(X_val_emb)

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="macro")

    print("\n=== Baseline Transfert Learning (BERT gelé + LogisticRegression) ===")
    print("Accuracy:", round(acc, 4))
    print("F1 macro:", round(f1, 4))

    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    # 7) Sauvegarde résultats
    os.makedirs("../reports", exist_ok=True)
    pd.DataFrame({
        "accuracy": [acc],
        "f1_macro": [f1],
    }).to_csv("../reports/tl_baseline_frozen.csv", index=False)

    # Sauvegarde du modèle (Classifieur) pour la démo
    import joblib
    model_dir = "../models/baseline"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "baseline_model.joblib"))
    print(f"\n✅ Modèle Baseline sauvegardé dans {model_dir}/baseline_model.joblib")
    print("✅ Résultats sauvegardés dans ../reports/tl_baseline_frozen.csv")


if __name__ == "__main__":
    main()
