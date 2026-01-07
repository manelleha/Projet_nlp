import os
import json
import random
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report
from openai import OpenAI

# =========================
# CONFIG
# =========================
# Modèles Groq possibles (selon dispo): "llama-3.1-8b-instant", "llama-3.1-70b-versatile"
GROQ_MODEL = "llama-3.1-8b-instant"

K_PER_CLASS_LIST = [1, 5, 10, 20]   # vrai few-shot (K-shot)
N_VAL_EVAL = 200                    # limite pour coût/temps
SEED = 42

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
    3: "Irrelevant",
}
INV_LABELS = {v: k for k, v in LABELS.items()}


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GROQ_API_KEY manquant. Fais: export GROQ_API_KEY='...'")

    # Groq est compatible OpenAI API
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )


def sample_k_per_class(train_df: pd.DataFrame, k: int, seed: int = SEED):
    """Sélectionne K exemples par classe dans le train."""
    examples = []
    for lbl in sorted(train_df["label"].unique()):
        subset = train_df[train_df["label"] == lbl].sample(n=k, random_state=seed)
        for _, row in subset.iterrows():
            examples.append((row["clean_text"], LABELS[int(lbl)]))
    random.Random(seed).shuffle(examples)
    return examples


def build_prompt(examples, tweet_text: str) -> str:
    """
    Few-shot prompting (in-context learning).
    On force une sortie JSON pour parser facilement.
    """
    intro = (
        "You are a strict classifier for tweet sentiment about an entity.\n"
        "Possible labels: Negative, Neutral, Positive, Irrelevant.\n"
        "Return ONLY valid JSON, with exactly one key: label.\n"
        "Example: {\"label\": \"Negative\"}\n\n"
        "Here are labeled examples:\n\n"
    )

    shots = ""
    for ex_text, ex_label in examples:
        shots += f'Tweet: "{ex_text}"\nLabel: {ex_label}\n\n'

    query = (
        "Now classify the following tweet.\n"
        f'Tweet: "{tweet_text}"\n'
        "Return JSON only."
    )

    return intro + shots + query


def parse_label(raw: str) -> str:
    """Parse la sortie du LLM pour récupérer un label parmi les 4."""
    raw = raw.strip()

    # 1) Essai JSON strict
    try:
        data = json.loads(raw)
        label = str(data.get("label", "")).strip()
        if label in INV_LABELS:
            return label
    except Exception:
        pass

    # 2) Fallback: cherche le label dans le texte
    low = raw.lower()
    for lab in LABELS.values():
        if lab.lower() in low:
            return lab

    return "Neutral"


def llm_predict(client: OpenAI, examples, tweet_text: str) -> tuple[int, str, str]:
    prompt = build_prompt(examples, tweet_text)

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,  # 'max_output_tokens' n'est pas standard OpenAI, 'max_tokens' l'est
        temperature=0.0
    )
    raw_out = resp.choices[0].message.content.strip()
    label_txt = parse_label(raw_out)
    return INV_LABELS[label_txt], label_txt, raw_out


def main():
    client = get_client()
    random.seed(SEED)

    train = pd.read_csv("../data/twitter_train_clean.csv")
    val   = pd.read_csv("../data/twitter_val_clean.csv")

    # Echantillon de validation pour éviter timeout API (300 est suffisant pour une estimation)
    val_eval = val.sample(n=min(300, len(val)), random_state=SEED).reset_index(drop=True)

    os.makedirs("../reports", exist_ok=True)

    summary_rows = []

    for k in K_PER_CLASS_LIST:
        print(f"\n=== Few-shot prompting (in-context) : K={k} exemples / classe ===")

        examples = sample_k_per_class(train, k, seed=SEED)

        y_true, y_pred = [], []
        rows_out = []

        for i, row in val_eval.iterrows():
            text = row["clean_text"]
            gold = int(row["label"])

            pred_id, pred_txt, raw_out = llm_predict(client, examples, text)

            y_true.append(gold)
            y_pred.append(pred_id)

            rows_out.append({
                "clean_text": text,
                "gold_label": gold,
                "pred_label": pred_id,
                "pred_label_text": pred_txt,
                "raw_output": raw_out
            })

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(val_eval)} traités...")

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 macro:  {f1:.4f}")

        # sauver les prédictions pour analyse d'erreurs
        pred_file = f"../reports/fewshot_prompting_groq_k{k}.csv"
        pd.DataFrame(rows_out).to_csv(pred_file, index=False)

        summary_rows.append({
            "k_per_class": k,
            "n_val_eval": len(val_eval),
            "accuracy": acc,
            "f1_macro": f1,
            "predictions_file": pred_file,
            "model": GROQ_MODEL
        })

    summary_file = "../reports/fewshot_prompting_groq_results.csv"
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    print(f"\n✅ Résultats sauvegardés dans {summary_file}")
    print("➡️ Prédictions détaillées: ../reports/fewshot_prompting_groq_k*.csv")


if __name__ == "__main__":
    main()
