import os
os.environ["USE_TF"] = "0"  # Forcer Transformers à utiliser PyTorch uniquement

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-uncased"
FRACTIONS = [0.10, 0.20, 0.50, 1.00]
BATCH_SIZE = 64  # plus gros batch = plus rapide sur CPU si ça tient en RAM


def compute_embeddings(texts, tokenizer, model, device, desc="Embedding"):
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


def main():
    print(">> Chargement des fichiers...")
    train = pd.read_csv("../data/twitter_train_clean.csv")
    val   = pd.read_csv("../data/twitter_val_clean.csv")

    print("Taille train complete :", train.shape)
    print("Taille val :", val.shape)

    # 🔹 Pour débug : on réduit temporairement la taille du train
    # ➜ Quand tout marche, tu pourras enlever cette ligne.
    train = train.sample(n=5000, random_state=42)
    print("Taille train utilisée pour ce test :", train.shape)

    X_train_text = train["clean_text"].tolist()
    y_train = train["label"].values

    X_val_text = val["clean_text"].tolist()
    y_val = val["label"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé :", device)

    print("\n>> Chargement du tokenizer et du modèle...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    print(">> Gel des poids du modèle (pas de fine-tuning)...")
    for p in model.parameters():
        p.requires_grad = False

    print("\n>> Calcul des embeddings TRAIN...")
    train_emb = compute_embeddings(X_train_text, tokenizer, model, device, desc="Embedding train")

    print("\n>> Calcul des embeddings VAL...")
    val_emb = compute_embeddings(X_val_text, tokenizer, model, device, desc="Embedding val")

    os.makedirs("../reports", exist_ok=True)
    results = []

    print("\n>> Début des entraînements few-shot + full-data...")
    for frac in FRACTIONS:
        print(f"\n=== Baseline TL avec {int(frac*100)}% du train ===")

        if frac < 1.0:
            X_emb_frac, _, y_frac, _ = train_test_split(
                train_emb,
                y_train,
                train_size=frac,
                random_state=42,
                stratify=y_train
            )
        else:
            X_emb_frac, y_frac = train_emb, y_train

        print(f"  -> Nombre d'exemples utilisés : {len(y_frac)}")

        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X_emb_frac, y_frac)

        y_pred = clf.predict(val_emb)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="macro")

        regime = "few-shot" if frac < 1.0 else "full-data"

        print(f"  Régime : {regime}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 macro : {f1:.4f}")

        results.append({
            "fraction": frac,
            "regime": regime,
            "n_train": len(y_frac),
            "accuracy": acc,
            "f1_macro": f1
        })

    results_df = pd.DataFrame(results)

    # Référence full-data
    f1_full = results_df.loc[results_df["fraction"] == 1.0, "f1_macro"].iloc[0]
    results_df["delta_f1_vs_full"] = results_df["f1_macro"] - f1_full

    results_df.to_csv("../reports/tl_baseline_results.csv", index=False)
    print("\n✅ Résultats baseline TL enregistrés dans ../reports/tl_baseline_results.csv")
    print("Résultats récap :")
    print(results_df)


if __name__ == "__main__":
    print(">> Lancement du script tl_baseline.py")
    main()
