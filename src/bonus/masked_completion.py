from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
import time
import json
from pathlib import Path

MODELS_TO_TEST = {
    "Greek BERT (uncased)": "nlpaueb/bert-base-greek-uncased-v1",
    "Multilingual BERT": "bert-base-multilingual-cased",
    "Greek Legal BERT": "spyrosbriakos/greek_legal_bert_v2",
    "XLM-RoBERTa": "xlm-roberta-base"
}

def load_masked_sentences(input_dir="data/inputs"):
    sentences = []
    for i in range(1, 100): 
        path = Path(input_dir) / f"masked{i}.txt"
        if not path.exists():
            break
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                sentences.append(text)
    return sentences

def load_model(model_name):
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=5)
    mask_token = tokenizer.mask_token
    print(f"Using mask token: {mask_token}")
    return pipe, mask_token

def run_predictions(label, pipe, mask_token, sentences):
    print(f"\n{'='*60}\nModel: {label}\n{'='*60}")
    for sent in sentences:
        input_text = sent.replace("[MASK]", mask_token)
        print(f"\nInput: {sent}\n")

        current = input_text
        for i in range(current.count(mask_token)):
            print(f"Predicting MASK #{i + 1} ...")
            try:
                start = time.perf_counter()
                results = pipe(current)
                elapsed = round(time.perf_counter() - start, 2)
                print(f"Took {elapsed}s")
            except Exception as e:
                print(f"Error: {e}")
                break

            if isinstance(results, list) and isinstance(results[0], list):
                results = results[0]

            for res in results:
                print(f"  • {res['token_str']} (score: {round(res['score'], 4)})")

            best = results[0]['token_str']
            current = current.replace(mask_token, best, 1)

        print(f"\nFinal output for {label}: {current}")

def run_multi_model_evaluation():
    all_results = {}
    masked_sentences = load_masked_sentences()

    for label, model_id in MODELS_TO_TEST.items():
        try:
            pipe, mask_token = load_model(model_id)
            model_results = []

            for sent in masked_sentences:
                input_text = sent.replace("[MASK]", mask_token)
                current = input_text
                mask_data = []

                for i in range(current.count(mask_token)):
                    raw_preds = pipe(current)
                    preds = raw_preds[0] if isinstance(raw_preds[0], list) else raw_preds

                    top_5 = [{
                        "token": pred["token_str"],
                        "score": round(pred["score"], 4)
                    } for pred in preds]

                    best_token = top_5[0]["token"]
                    mask_data.append({
                        "mask_index": i + 1,
                        "top_5_predictions": top_5,
                        "chosen": best_token
                    })

                    current = current.replace(mask_token, best_token, 1)

                model_results.append({
                    "input": sent,
                    "completed": current,
                    "masks": mask_data
                })

            all_results[label] = model_results

        except Exception as e:
            all_results[label] = {"error": str(e)}
            print(f"Could not load {label}: {e}")

    save_outputs(all_results)

def save_outputs(results_dict, filename="data/outputs/masked_completion_outputs.json"):
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\nSaved predictions to: {filename}")
