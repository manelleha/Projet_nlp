# 🧠 Projet NLP — Sentiment Analysis with Transfer Learning & Few-Shot

## 📌 Objectif du projet
Ce projet vise à analyser des tweets liés à des entités et prédire leur sentiment parmi 4 classes :
0 Negative, 1 Neutral, 2 Positive, 3 Irrelevant.

Le projet explore :
- le transfert learning
- le fine-tuning
- le few-shot learning (prompting LLM)
- l’impact de la quantité de données

## 🗂️ Structure
Projet_nlp/
│
├── data/
│   ├── twitter_train_clean.csv   # dataset d'entraînement nettoyé
│   └── twitter_val_clean.csv     # dataset de validation nettoyé
│
├── src/
│   ├── eda.py                    # exploration des données
│   ├── tl_baseline_frozen.py     # baseline TL (BERT gelé + LogisticRegression)
│   ├── tl_finetune.py            # fine-tuning BERT
│   ├── fewshot_prompting_groq.py # vrai few-shot (prompting LLM)
│   └── plot_results.py           # (à compléter) visualisations & learning curves
│
├── models/
│   └── bert_finetuned/           # modèle fine-tuné sauvegardé
│
└── reports/
    ├── tl_baseline_frozen.csv
    ├── finetune_metrics.csv
    ├── finetune_classification_report.txt
    ├── fewshot_prompting_groq_results.csv
    └── figures/


## 🧪 Scripts principaux
- eda.py : exploration des données
- tl_baseline_frozen.py : baseline TL gelée
- tl_finetune.py : fine-tuning BERT
- fewshot_prompting_groq.py : few-shot par prompting
- plot_results.py : visualisations finales


## 🧩 Rôle des fichiers principaux du projet
# 📊 eda.py — Exploration des données
Rôle : comprendre le dataset avant de modéliser.
Ce script analyse :
la taille du dataset
la distribution des labels
la longueur des tweets
des exemples de textes par classe
des visualisations enregistrées dans reports/
Pourquoi c’est important :
Il permet de justifier les choix du modèle et d’anticiper les difficultés (ex : classe Irrelevant plus complexe).
# 🧱 tl_baseline_frozen.py — Baseline de Transfert Learning (modèle gelé)
Rôle : établir une référence solide avec du transfert learning sans fine-tuning.
Méthode :
BERT pré-entraîné
poids complètement gelés
extraction d’embeddings
classifieur simple (Logistic Regression)
Pourquoi :
C’est la baseline principale demandée.
Elle permet de mesurer ce que vaut un modèle pré-entraîné sans adaptation au dataset.
# 🧠 tl_finetune.py — Fine-tuning de BERT
Rôle : améliorer la performance en adaptant le modèle au dataset.
Méthode :
même modèle BERT
mais toutes les couches sont entraînées
apprentissage supervisé sur les tweets
Pourquoi :
Permet de comparer :
modèle gelé vs modèle adapté
et de démontrer concrètement l’intérêt du fine-tuning.
# 🎯 fewshot_prompting_groq.py — Few-shot “comme en cours” (prompting)
Rôle : implémenter le vrai few-shot vu en cours (TD Prompt Engineering).
Méthode :
aucun entraînement
quelques exemples (K-shot) injectés dans le prompt
le LLM prédit directement le label
Pourquoi :
Ce fichier montre la capacité d’un grand modèle à apprendre uniquement par le contexte,
et permet de comparer :
apprentissage classique vs prompting sans entraînement.



##  À faire
- tester tl_finetune.py et fewshot_prompting_groq.py
- comparer les modeles -> faire une doc


