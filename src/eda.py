# src/eda.py

import os
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# -------------------------------------------------
# Chargement des données
# -------------------------------------------------
train = pd.read_csv("../data/twitter_train_clean.csv")
val   = pd.read_csv("../data/twitter_val_clean.csv")

df = pd.concat([train, val], ignore_index=True)

os.makedirs("../reports", exist_ok=True)

# -------------------------------------------------
# 1. Infos générales
# -------------------------------------------------
print("Taille train :", train.shape)
print("Taille validation :", val.shape)
print("Taille totale :", df.shape)

print("\nDistribution des classes (train) :")
print(train["label"].value_counts())

print("\nDistribution des classes (validation) :")
print(val["label"].value_counts())

# -------------------------------------------------
# 2. Longueur des tweets
# -------------------------------------------------
df["length"] = df["clean_text"].str.len()

print("\nStatistiques longueur de texte (ensemble complet) :")
print(df["length"].describe())

# Histogramme des longueurs
plt.figure(figsize=(7, 5))
plt.hist(df["length"], bins=60)
plt.title("Distribution des longueurs des tweets")
plt.xlabel("Longueur (nb de caractères)")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig("../reports/length_distribution.png")
plt.close()

# Boxplot des longueurs
plt.figure(figsize=(7, 3))
sns.boxplot(x=df["length"])
plt.title("Boxplot des longueurs des tweets")
plt.tight_layout()
plt.savefig("../reports/length_boxplot.png")
plt.close()

print("Graphiques de longueur sauvegardés.")

# -------------------------------------------------
# 3. Distribution des labels (barplot)
# -------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=df)
plt.title("Distribution des classes (ensemble complet)")
plt.xlabel("Label")
plt.ylabel("Nombre d'exemples")
plt.tight_layout()
plt.savefig("../reports/class_distribution.png")
plt.close()

print("Graphique de distribution des classes sauvegardé.")

# -------------------------------------------------
# 4. Wordcloud par classe
# -------------------------------------------------
for label in sorted(df["label"].unique()):
    text = " ".join(df[df["label"] == label]["clean_text"].tolist())
    if not text.strip():
        continue

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(10, 5))
    img = wc.to_image()  # évite le bug numpy/wordcloud
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Wordcloud — Classe {label}")
    plt.tight_layout()
    plt.savefig(f"../reports/wordcloud_label_{label}.png")
    plt.close()

print("Wordclouds sauvegardés.")

# -------------------------------------------------
# 5. Mots les plus fréquents par classe
# -------------------------------------------------
def clean_tokens(text: str):
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    return [tok for tok in text.split() if len(tok) > 2]  # on vire les tokens trop courts

for label in sorted(df["label"].unique()):
    words = []
    for txt in df[df["label"] == label]["clean_text"]:
        words.extend(clean_tokens(txt))

    most_common = Counter(words).most_common(20)
    if not most_common:
        continue

    freq_df = pd.DataFrame(most_common, columns=["word", "count"])

    plt.figure(figsize=(8, 6))
    sns.barplot(data=freq_df, x="count", y="word")
    plt.title(f"Top 20 mots — Classe {label}")
    plt.xlabel("Fréquence")
    plt.ylabel("Mot")
    plt.tight_layout()
    plt.savefig(f"../reports/top_words_label_{label}.png")
    plt.close()

print("Top mots par classe sauvegardés.")

# -------------------------------------------------
# 6. Exemples par classe (console)
# -------------------------------------------------
print("\nExemples (3) par classe :")
for label in sorted(df["label"].unique()):
    print(f"\n=== Classe {label} ===")
    print(df[df["label"] == label]["clean_text"].head(3).to_string(index=False))

print("\n✅ EDA terminé. Tous les graphiques sont dans le dossier ../reports/")
