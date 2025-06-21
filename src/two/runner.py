import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .pipeline import compute_combined_similarity

def run_similarity_analysis(data_path="data/outputs/outputs.json"):
    data = json.load(open(Path(data_path), encoding="utf-8"))

    if "1A" not in data or "1B" not in data:
        print("1A or 1B not found in JSON.")
        return

    print("\nCosine Similarity (Original vs Reconstructions):")

    coords, labels, similarities = compute_combined_similarity(data["1A"], data["1B"])

    # Αποθήκευση similarity scores
    df_sim = pd.DataFrame(similarities)
    sim_path = Path("data/outputs/similarity.csv")
    df_sim.to_csv(sim_path, index=False)
    print(f"Similarity scores saved to: {sim_path}")

    # Οπτικοποίηση
    plt.figure(figsize=(9, 6))
    for i, label in enumerate(labels):
        x, y = coords[i]
        plt.scatter(x, y)
        plt.annotate(label, (x + 0.01, y + 0.01), fontsize=8)

    plt.title("Semantic Space (Originals vs Reconstructions)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()

    img_path = Path("data/outputs/semantic_space.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"PCA unified plot saved to: {img_path}")
