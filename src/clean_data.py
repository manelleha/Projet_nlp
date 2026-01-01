import pandas as pd
import re
import os

# Assure qu'on a un dossier processed/
os.makedirs("../data/processed", exist_ok=True)

# Charger les datasets bruts
train = pd.read_csv("../data/twitter_training.csv", header=None)
test  = pd.read_csv("../data/twitter_validation.csv", header=None)

# Renommer les colonnes selon le format Kaggle
train.columns = ["tweet_id", "entity", "sentiment", "tweet_text"]
test.columns  = ["tweet_id", "entity", "sentiment", "tweet_text"]

print("Train shape :", train.shape)
print("Test shape  :", test.shape)

# Fusionner les deux datasets
df = pd.concat([train, test], ignore_index=True)

# Nettoyage du texte
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)        # URLs
    text = re.sub(r"@\w+", " ", text)          # mentions
    text = re.sub(r"#\w+", " ", text)          # hashtags
    text = re.sub(r"[^a-z\s]", " ", text)      # caractères spéciaux/chiffres
    text = re.sub(r"\s+", " ", text).strip()   # espaces multiples
    return text

df["clean_text"] = df["tweet_text"].apply(clean_text)

# Supprimer lignes vides
df = df[df["clean_text"].str.len() > 3]

# Encodage des sentiments
label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
    "Irrelevant": 3
}

df["label"] = df["sentiment"].map(label_map)

# Supprimer les lignes sans label valide
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Colonnes finales
df_final = df[["clean_text", "label"]]

# Sauvegarde
output_path = "../data/processed/twitter_clean.csv"
df_final.to_csv(output_path, index=False, encoding="utf-8")

print("\nSUCCESS 🎉")
print("Fichier généré :", output_path)
print("Shape finale :", df_final.shape)
print(df_final.head())
