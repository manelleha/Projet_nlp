import os
import re
import pandas as pd

RAW_TRAIN_PATH = "../data/twitter_training.csv"
RAW_VAL_PATH   = "../data/twitter_validation.csv"

OUT_TRAIN_PATH = "../data/twitter_train_clean.csv"
OUT_VAL_PATH   = "../data/twitter_val_clean.csv"

# -------------------------
# 1. Fonction de nettoyage
# -------------------------
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)        # URLs
    text = re.sub(r"@\w+", " ", text)          # mentions
    text = re.sub(r"#\w+", " ", text)          # hashtags
    text = re.sub(r"[^a-z\s]", " ", text)      # tout sauf lettres/espaces
    text = re.sub(r"\s+", " ", text).strip()   # espaces multiples
    return text

# -------------------------
# 2. Nettoyage d'un fichier
# -------------------------
def process_file(path_in: str, path_out: str, name: str):
    print(f"\n=== Traitement de {name} ===")

    # Les fichiers Kaggle n'ont pas de header → header=None
    df = pd.read_csv(path_in, header=None)
    print(f"Shape brut {name} :", df.shape)

    # Colonnes Kaggle : tweet_id, entity, sentiment, tweet_text
    df.columns = ["tweet_id", "entity", "sentiment", "tweet_text"]

    # Nettoyage du texte
    df["clean_text"] = df["tweet_text"].astype(str).apply(clean_text)

    # Supprimer les lignes trop courtes
    df = df[df["clean_text"].str.len() > 3]

    # Encodage des labels
    label_map = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
        "Irrelevant": 3
    }

    df["label"] = df["sentiment"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Garder seulement ce qui nous intéresse
    df_clean = df[["clean_text", "label"]]

    print(f"Shape nettoyé {name} :", df_clean.shape)
    print("Distribution des labels :")
    print(df_clean["label"].value_counts())

    # Sauvegarde
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    df_clean.to_csv(path_out, index=False, encoding="utf-8")
    print(f"✅ Fichier sauvegardé dans {path_out}")

def main():
    process_file(RAW_TRAIN_PATH, OUT_TRAIN_PATH, "TRAIN")
    process_file(RAW_VAL_PATH,   OUT_VAL_PATH,   "VALIDATION")

if __name__ == "__main__":
    main()
