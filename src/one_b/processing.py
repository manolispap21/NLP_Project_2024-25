import nltk
from nltk.tokenize import sent_tokenize
from one_b.paraphrasers import paraphrase_t5, back_translate_en, paraphrase_pegasus
from pathlib import Path

nltk.download('punkt')

def load_texts(filenames, input_dir="data/inputs"):
    texts = {}
    for fname in filenames:
        fpath = Path(input_dir) / fname
        with open(fpath, "r", encoding="utf-8") as f:
            texts[fname.replace(".txt", "")] = f.read().strip()
    return texts

def split_sentences(text):
    return sent_tokenize(text, language='english')

def run_all_pipelines(filenames, input_dir="data/inputs"):
    texts = load_texts(filenames, input_dir)
    results = {}
    for key, text in texts.items():
        sents = split_sentences(text)
        p1 = [paraphrase_t5(s) for s in sents]
        p2 = [back_translate_en(s) for s in sents]
        p3 = [paraphrase_pegasus(s) for s in sents]
        results[key] = {
            'original': text,
            'P1_t5': ' '.join(p1),
            'P2_backtranslation': ' '.join(p2),
            'P3_pegasus': ' '.join(p3)
        }
    return results