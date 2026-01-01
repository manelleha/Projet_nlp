import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Charger le dataset nettoyé
df = pd.read_csv("../data/processed/twitter_clean.csv")

print("Taille totale :", df.shape)
print(df.head())

# 2. Distribution des labels
print("\nDistribution des labels :")
print(df["label"].value_counts())

# 3. Longueur des tweets
df["length"] = df["clean_text"].str.len()
print("\nStats longueur :")
print(df["length"].describe())

# 4. Création du dossier reports si besoin
os.makedirs("../reports", exist_ok=True)

# 5. Histogramme des longueurs
plt.hist(df["length"], bins=50)
plt.title("Distribution de la longueur des tweets")
plt.xlabel("Longueur")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig("../reports/hist_lengths.png")
print("Graphique enregistré dans ../reports/hist_lengths.png")
