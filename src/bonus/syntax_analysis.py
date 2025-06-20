import json
import os
import pandas as pd

try:
    import stanza
except ImportError:
    print("Stanza is not installed. Run: poetry add stanza")
    raise

def run_syntax_analysis(json_path="data/outputs/masked_completion_outputs.json"):
    stanza.download("el", verbose=False)
    nlp = stanza.Pipeline("el", processors="tokenize,mwt,pos,lemma,depparse", use_gpu=False)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        return

    rows = []
    for model_name, results in data.items():
        if not isinstance(results, list):
            continue

        for idx, entry in enumerate(results, 1):
            sentence = entry["completed"]
            doc = nlp(sentence)
            for sent in doc.sentences:
                for word in sent.words:
                    head_word = sent.words[word.head - 1].text if word.head > 0 else "ROOT"
                    rows.append({
                        "model": model_name,
                        "sentence_index": idx,
                        "token": word.text,
                        "lemma": word.lemma,
                        "upos": word.upos,
                        "xpos": word.xpos,
                        "deprel": word.deprel,
                        "head": head_word
                    })

    df = pd.DataFrame(rows)
    output_path = "data/outputs/syntax_analysis.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSyntax Analysis was saved in: {output_path}")
