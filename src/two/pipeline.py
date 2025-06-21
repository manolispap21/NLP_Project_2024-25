import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in stop_words)

def compute_combined_similarity(data_1a, data_1b):
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = []
    labels = []
    similarities = []

    # Προτάσεις 1A
    for key, versions in data_1a.items():
        orig = preprocess(versions["original"])
        rew = preprocess(versions["rewritten"])
        emb_orig = model.encode(orig)
        emb_rew = model.encode(rew)
        sim = cosine_similarity([emb_orig], [emb_rew])[0][0]
        embeddings.extend([emb_orig, emb_rew])
        labels.extend([f"{key}-original", f"{key}-rewritten"])
        similarities.append({"id": key, "type": "sentence", "similarity": sim})

    # Κείμενα 1B
    for key, versions in data_1b.items():
        orig = preprocess(versions["original"])
        emb_orig = model.encode(orig)
        for name, val in versions.items():
            if name == "original":
                continue
            mod = preprocess(val)
            emb_mod = model.encode(mod)
            sim = cosine_similarity([emb_orig], [emb_mod])[0][0]
            embeddings.extend([emb_orig, emb_mod])
            labels.extend([f"{key}-original", f"{key}-{name}"])
            similarities.append({"id": f"{key}-{name}", "type": "text", "similarity": sim})

    coords = PCA(n_components=2).fit_transform(embeddings)
    return coords, labels, similarities
