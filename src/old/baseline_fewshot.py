import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import os

FRACTIONS = [0.05, 0.10, 0.20, 0.50, 1.00]

# Charger le dataset propre
df = pd.read_csv("../data/processed/twitter_clean.csv")

X = df["clean_text"]
y = df["label"]

# Split global train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

os.makedirs("../reports", exist_ok=True)

results = []

for frac in FRACTIONS:
    print(f"\n=== Baseline avec {int(frac*100)}% des données ===")

    if frac < 1.0:
        X_train_frac, _, y_train_frac, _ = train_test_split(
            X_train, y_train,
            train_size=frac,
            random_state=42,
            stratify=y_train
        )
    else:
        X_train_frac, y_train_frac = X_train, y_train

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train_frac)
    X_test_vec = vectorizer.transform(X_test)

    # Modèle baseline
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_vec, y_train_frac)

    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy:", acc)
    print("F1 macro:", f1)

    results.append({
        "fraction": frac,
        "n_train": len(X_train_frac),
        "accuracy": acc,
        "f1": f1
    })

pd.DataFrame(results).to_csv("../reports/fewshot_baseline.csv", index=False)
print("Résultats enregistrés dans ../reports/fewshot_baseline.csv")
