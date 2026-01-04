Accuracy = 0.6519, F1 macro = 0.6347
Les classes 0 (Negative) et 2 (Positive) sont plutôt bien captées (F1 ~0.70).
La classe 3 (Irrelevant) est la plus difficile (recall 0.44) : le modèle a du mal à repérer le “hors sujet” → c’est logique car ça demande plus de compréhension du contexte/entité.



La baseline “feature-based transfer learning” (BERT gelé + régression logistique) atteint 0.65 d’accuracy et 0.63 de F1 macro. Les classes sentimentées sont relativement bien séparées, tandis que la classe “irrelevant” est plus difficile, ce qui suggère que le fine-tuning pourrait améliorer la prise en compte du contexte.