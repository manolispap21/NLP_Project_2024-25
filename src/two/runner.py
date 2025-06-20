# runner.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .pipeline import compute_similarity_and_embeddings

def run_similarity_analysis(data_path="data/outputs/outputs.json"):
    # Φόρτωση JSON
    data = json.load(open(Path(data_path), encoding="utf-8"))

    if "1B" not in data:
        raise ValueError("JSON is empty for 1B.")

    results, coords, labels = compute_similarity_and_embeddings(data["1B"])
    df = pd.DataFrame(results)

    # Εμφάνιση πίνακα στο terminal
    print("\nCosine Similarity (Original vs Reconstructions):")
    print(df.to_string(index=False))

    # Αποθήκευση CSV με similarity scores
    sim_path = Path("data/outputs/similarity_scores.csv")
    df.to_csv(sim_path, index=False)
    print(f"Similarity scores saved to: {sim_path}")

    # Οπτικοποίηση PCA
    plt.figure(figsize=(9, 6))
    for i, label in enumerate(labels):
        x, y = coords[i]
        plt.scatter(x, y)
        plt.annotate(label, (x + 0.015, y + 0.015), fontsize=9)

    plt.title("PCA Projection of Embeddings (Original vs Reconstructions)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()

    # Αποθήκευση εικόνας
    img_path = Path("data/outputs/pca_similarity_plot.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nPCA plot saved to: {img_path}")

    return df
