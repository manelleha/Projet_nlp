import os
os.environ["USE_TF"] = "0"  # On force Transformers à utiliser PyTorch uniquement

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModel

# Modèle de base (anglais)
MODEL_NAME = "bert-base-uncased"

# K-shot par classe (vrai few-shot supervisé)
K_LIST = [1, 5, 10, 20, 50, 100]

BATCH_SIZE = 64


def compute_embeddings(texts, tokenizer, model, device, desc="Embedding"):
    """Calcule les embeddings CLS pour une liste de textes."""
    dataset = TensorDataset(torch.arange(len(texts)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_emb = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            idx = batch[0].numpy()
            batch_texts = [texts[i] for i in idx]

            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}

            outputs = model(**encodings)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_emb.append(cls_emb)

    return np.vstack(all_emb)


def sample_k_per_class(labels, k, random_state=42):
    """
    Retourne les index d'un échantillon K-shot par classe.
    Si une classe contient < K exemples, on prend tout.
    """
    rng = np.random.default_rng(random_state)
    labels = np.array(labels)

    indices_per_class = []
    for lab in np.unique(labels):
        idx_lab = np.where(labels == lab)[0]
        if len(idx_lab) <= k:
            chosen = idx_lab
        else:
            chosen = rng.choice(idx_lab, size=k, replace=False)
        indices_per_class.append(chosen)

    return np.concatenate(indices_per_class)


def main():
    print(">> Chargement des fichiers...")
    train = pd.read_csv("../data/twitter_train_clean.csv")
    val   = pd.read_csv("../data/twitter_val_clean.csv")

    print("Taille train :", train.shape)
    print("Taille val   :", val.shape)

    X_train_text = train["clean_text"].tolist()
    y_train = train["label"].values

    X_val_text = val["clean_text"].tolist()
    y_val = val["label"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)

    print("\n>> Chargement du tokenizer et du modèle pré-entraîné...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    print(">> Gel des poids du modèle (feature-based TL, pas de fine-tuning)...")
    for p in model.parameters():
        p.requires_grad = False

    print("\n>> Calcul des embeddings TRAIN...")
    train_emb = compute_embeddings(X_train_text, tokenizer, model, device, desc="Embedding train")

    print("\n>> Calcul des embeddings VAL...")
    val_emb = compute_embeddings(X_val_text, tokenizer, model, device, desc="Embedding val")

    os.makedirs("../reports", exist_ok=True)
    results = []

    # =======================
    # 1) Vrai few-shot K-shot
    # =======================

    print("\n>> Expériences few-shot (K-shot par classe)...")

    for k in K_LIST:
        print(f"\n=== K-shot (K={k} par classe) ===")
        idx_k = sample_k_per_class(y_train, k, random_state=42)

        X_emb_k = train_emb[idx_k]
        y_k     = y_train[idx_k]

        print(f"  -> Nombre total d'exemples utilisés : {len(y_k)}")

        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X_emb_k, y_k)

        y_pred = clf.predict(val_emb)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="macro")

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 macro : {f1:.4f}")

        results.append({
            "mode": "k-shot",
            "k_per_class": k,
            "n_train": len(y_k),
            "accuracy": acc,
            "f1_macro": f1,
        })

    # =======================
    # 2) Référence full-data
    # =======================

    print("\n>> Entraînement sur TOUT le train (full-data)...")

    clf_full = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf_full.fit(train_emb, y_train)
    y_pred_full = clf_full.predict(val_emb)

    acc_full = accuracy_score(y_val, y_pred_full)
    f1_full  = f1_score(y_val, y_pred_full, average="macro")

    print(f"  [FULL-DATA] Accuracy : {acc_full:.4f}")
    print(f"  [FULL-DATA] F1 macro : {f1_full:.4f}")

    results.append({
        "mode": "full-data",
        "k_per_class": None,
        "n_train": len(y_train),
        "accuracy": acc_full,
        "f1_macro": f1_full,
    })

    # =======================
    # 3) Sauvegarde + delta F1
    # =======================

    results_df = pd.DataFrame(results)

    # F1 de référence = full-data
    f1_ref = acc_full  # ou f1_full si tu veux comparer en F1
    # Si tu préfères comparer en F1 :
    # f1_ref = f1_full

    # Ici je compare en F1 :
    f1_ref = f1_full
    results_df["delta_f1_vs_full"] = results_df["f1_macro"] - f1_ref

    results_df.to_csv("../reports/tl_baseline_kshot_results.csv", index=False)
    print("\n✅ Résultats K-shot baseline TL enregistrés dans ../reports/tl_baseline_kshot_results.csv")
    print(results_df)


if __name__ == "__main__":
    print(">> Lancement tl_baseline_kshot.py (vrai K-shot supervisé sur BERT gelé)")
    main()
