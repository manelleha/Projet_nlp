
### Structure des fichiers nettoyés :

| Colonne      | Type   | Description |
|--------------|---------|-------------|
| `clean_text` | string | Tweet nettoyé (minuscule, sans URL, sans mentions, sans hashtags, sans symboles) |
| `label`      | int    | Classe de sentiment encodée (0–3) |

---

## 🧼 2. Nettoyage effectué

Les transformations appliquées :

- Mise en minuscules  
- Suppression :
  - URLs (`http...`)
  - mentions (`@username`)
  - hashtags (`#topic`)
  - ponctuation et caractères spéciaux  
- Réduction des espaces multiples  
- Filtrage des tweets trop courts  
- Conversion du sentiment textuel en label numérique  

> ⚠️ Le nettoyage permet aux modèles classiques (TF-IDF, logistic regression) et aux modèles Transformers de fonctionner correctement.

---

## 🏷️ 3. Signification des labels

| Label | Nom original | Description |
|-------|---------------|-------------|
| **0** | Negative      | Opinion négative envers l’entité |
| **1** | Neutral       | Information factuelle, sans opinion |
| **2** | Positive      | Opinion favorable |
| **3** | Irrelevant    | Le tweet ne concerne pas réellement l'entité |

---

## 📊 4. Taille et distribution des classes

### **Entraînement (`twitter_train_clean.csv`)**
- **72 051 tweets**
- Distribution :

| Classe | Nombre |
|--------|---------|
| 0 (Negative) | 21 804 |
| 1 (Neutral)  | 17 623 |
| 2 (Positive) | 20 017 |
| 3 (Irrelevant) | 12 607 |

---

### **Validation (`twitter_val_clean.csv`)**
- **994 tweets**
- Distribution proche de celle du train :

| Classe | Nombre |
|--------|---------|
| 1 (Neutral)  | 285 |
| 2 (Positive) | 274 |
| 0 (Negative) | 263 |
| 3 (Irrelevant) | 172 |

> ✔️ Le jeu de validation est bien équilibré et représentatif → parfait pour l’évaluation.

---

## 🔎 5. Pourquoi deux fichiers séparés ?

Nous utilisons :
- **un dataset d'entraînement** → pour l’apprentissage, y compris few-shot  
- **un dataset de validation** → pour l’évaluation uniquement  

Cela garantit une comparaison stable et reproductible entre les modèles :
- baseline transfert learning  
- fine-tuning  
- few-shot learning (10 %, 20 %, 50 %, 100 % du train)

---

## ⚙️ 6. Exemple d'utilisation

### Charger les données :

```python
import pandas as pd

train = pd.read_csv("data/twitter_train_clean.csv")
val = pd.read_csv("data/twitter_val_clean.csv")
