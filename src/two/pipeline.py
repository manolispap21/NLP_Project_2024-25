import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessor
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join(t for t in tokens if t not in stop_words)

# Υπολογισμός ενσωματώσεων και cosine similarity
def compute_similarity_and_embeddings(data_dict):
    model = SentenceTransformer("all-mpnet-base-v2")
    results = []
    all_texts = []
    labels = []

    for text_key, versions in data_dict.items():
        original = preprocess(versions["original"])
        emb_orig = model.encode(original)
        row = {"Text": text_key}
        all_texts.append(original)
        labels.append(f"{text_key}-Original")

        for key, val in versions.items():
            if key == "original":
                continue
            processed = preprocess(val)
            emb = model.encode(processed)
            sim = cosine_similarity([emb_orig], [emb])[0][0]
            row[key] = sim
            all_texts.append(processed)
            labels.append(f"{text_key}-{key}")

        results.append(row)

    embeddings = model.encode(all_texts)
    coords = PCA(n_components=2).fit_transform(embeddings)
    return results, coords, labels
