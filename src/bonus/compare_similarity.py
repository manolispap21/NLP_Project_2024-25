import json
import pandas as pd
import unicodedata
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

def strip_tonos(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def extract_ground_truth_all(input_dir="data/inputs"):
    ground_truth_by_sentence = {}
    for i in range(1, 100): 
        orig_path = Path(input_dir) / f"original{i}.txt"
        masked_path = Path(input_dir) / f"masked{i}.txt"
        if not orig_path.exists() or not masked_path.exists():
            break
        with open(orig_path, "r", encoding="utf-8") as f1, open(masked_path, "r", encoding="utf-8") as f2:
            original = f1.read().split()
            masked = f2.read().split()
            ground_truth = [o for o, m in zip(original, masked) if m == "[MASK]"]
            ground_truth_by_sentence[f"sentence_{i}"] = ground_truth
    return ground_truth_by_sentence

def run_similarity_vs_ground_truth(json_path="data/outputs/masked_completion_outputs.json"):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Φόρτωση δυναμικού ground truth
    ground_truth_by_sentence = extract_ground_truth_all()

    # Φόρτωση προβλέψεων
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        return

    results = []
    for model_name, outputs in data.items():
        if not isinstance(outputs, list):
            continue

        for idx, entry in enumerate(outputs, 1):
            key = f"sentence_{idx}"
            gold_tokens = ground_truth_by_sentence.get(key)
            if not gold_tokens:
                continue

            predicted_tokens = [mask["chosen"] for mask in entry["masks"]]
            for i, (pred_raw, gold_raw) in enumerate(zip(predicted_tokens, gold_tokens), 1):
                pred = strip_tonos(pred_raw.strip().lower())
                gold = strip_tonos(gold_raw.strip().lower())
                match = pred == gold
                sim = util.cos_sim(model.encode(pred), model.encode(gold)).item()
                results.append({
                    "model": model_name,
                    "sentence": key,
                    "mask_index": i,
                    "predicted": pred_raw,
                    "ground_truth": gold_raw,
                    "exact_match_no_tonos": match,
                    "cosine_similarity": round(sim, 4)
                })

    df = pd.DataFrame(results)

    # Μέσος όρος per model
    summary = df.groupby("model")["cosine_similarity"].mean().reset_index()
    summary["sentence"] = "SUMMARY"
    summary["mask_index"] = ""
    summary["predicted"] = ""
    summary["ground_truth"] = ""
    summary["exact_match_no_tonos"] = ""
    summary["cosine_similarity"] = summary["cosine_similarity"]
    summary = summary[df.columns]  

    df_final = pd.concat([df, summary], ignore_index=True)

    # Save
    output_path = "data/outputs/mask_similarity_comparison.csv"
    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Εκτύπωση
    avg_df = summary[["model", "cosine_similarity"]].rename(columns={"cosine_similarity": "average_cosine_similarity"})
    print(f"\Saved: {output_path}\n")
    print("Average Cosine Similarity per model:")
    print(avg_df.to_string(index=False))
