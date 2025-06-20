import spacy
from pathlib import Path
from one_a.config import SUBJECT_TO_POSSESSIVE

# === Φόρτωση μοντέλου ===
nlp = spacy.load("en_core_web_sm")

# === Συντακτικές βοηθητικές ===

def load_texts(filenames, input_dir="data/inputs"):
    """
    Φορτώνει κείμενα από αρχεία .txt με βάση λίστα ονομάτων αρχείων.
    Επιστρέφει λεξικό {\"όνομα\": \"περιεχόμενο\"} χωρίς το .txt στο όνομα.
    """
    texts = {}
    for fname in filenames:
        fpath = Path(input_dir) / fname
        with open(fpath, "r", encoding="utf-8") as f:
            texts[fname.replace(".txt", "")] = f.read().strip()
    return texts

def has_subject(doc):
    """
    Ελέγχει αν η πρόταση έχει υποκείμενο (nsubj ή nsubjpass).
    """
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    return root and any(t.dep_ in {"nsubj", "nsubjpass"} and t.head == root for t in doc)

def get_subject_possessive(verb_token):
    for child in verb_token.children:
        if child.dep_ == "nsubj" and child.text.lower() in SUBJECT_TO_POSSESSIVE:
            return SUBJECT_TO_POSSESSIVE[child.text.lower()]
    for ancestor in verb_token.ancestors:
        for child in ancestor.children:
            if child.dep_ == "nsubj" and child.text.lower() in SUBJECT_TO_POSSESSIVE:
                return SUBJECT_TO_POSSESSIVE[child.text.lower()]
    return "their"
