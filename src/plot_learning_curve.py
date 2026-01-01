import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("reports/fewshot_baseline.csv")
transf   = pd.read_csv("reports/fewshot_transformer.csv")

plt.figure(figsize=(8,5))

plt.plot(baseline["n_train"], baseline["f1"], marker="o", label="Baseline TF-IDF")
plt.plot(transf["n_train"], transf["f1"], marker="o", label="Transformers")

plt.xscale("log")
plt.xlabel("Taille du jeu d'entraînement (log)")
plt.ylabel("F1 macro")
plt.title("Courbe d'apprentissage (Few-Shot Learning)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/learning_curve.png")

print("Courbe enregistrée : reports/learning_curve.png")
