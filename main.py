import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

import pandas as pd

sys.path.append("src")

from one_b.processing import run_all_pipelines as run_one_b
from one_a.pipeline import rewrite_sentence
from one_a.utils import load_texts
from two.runner import run_similarity_analysis
from bonus.masked_completion import run_multi_model_evaluation as run_masked_bonus
from bonus.syntax_analysis import run_syntax_analysis 
from bonus.compare_similarity import run_similarity_vs_ground_truth

INPUT_DIR = "data/inputs"
OUTPUT_DIR = "data/outputs"
INPUT_FILES_1A = ["sentence1.txt", "sentence2.txt"]
INPUT_FILES_1B = ["text1.txt", "text2.txt"]

def pretty_print_results(results, title):
    print(f"\n {title} Results:\n" + "=" * (len(title) + 10))
    for fname, res in results.items():
        print(f"\n {fname}")
        for key, value in res.items():
            print(f"  {key.capitalize()}: {value}")

def run_reconstruction_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_combined = {}

    texts_1a = load_texts(INPUT_FILES_1A, input_dir=INPUT_DIR)
    results_1a = {}
    for fname, text in tqdm(texts_1a.items(), desc="Running 1A Pipeline"):
        results_1a[fname + ".txt"] = {
            "original": text,
            "rewritten": rewrite_sentence(text)
        }
    results_combined["1A"] = results_1a
    pretty_print_results(results_1a, "1A")

    print()
    results_1b = {}
    print("Running 1B Pipeline:")
    for fname in tqdm(INPUT_FILES_1B, desc="Generating paraphrases"):
        result = run_one_b([fname], input_dir=INPUT_DIR)
        results_1b.update(result)
    results_combined["1B"] = results_1b
    pretty_print_results(results_1b, "1B")

    with open(f"{OUTPUT_DIR}/outputs.json", "w", encoding="utf-8") as f:
        json.dump(results_combined, f, indent=2, ensure_ascii=False)
    print("\n Completed 1A and 1B – saved to outputs.json")

def display_menu():
    print("\n" + "=" * 50)
    print("NLP Final Project – Pipeline Runner".center(50))
    print("=" * 50)
    print("1. Run reconstruction pipelines (1A + 1B)")
    print("2. Run similarity analysis (2)")
    print("3. Run masked clause completion (Bonus)")
    print("4. Run syntactic analysis of completed sentences (Bonus)")
    print("5. Compare predictions with ground truth (Bonus)")
    print("Exit")
    print("=" * 50)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    while True:
        display_menu()
        mode = input("Select option (1 / 2 / 3 / 4 / 5 / exit): ").strip().lower()

        if mode == "1":
            run_reconstruction_pipeline()

        elif mode == "2":
            print("\nRunning Similarity Analysis...")
            try:
                run_similarity_analysis(data_path=f"{OUTPUT_DIR}/outputs.json")
            except Exception as e:
                print(f"Error during similarity analysis: {e}")

        elif mode == "3":
            print("\nRunning Masked Clause Completion...")
            try:
                run_masked_bonus()
            except Exception as e:
                print(f"Error during masked clause completion: {e}")

        elif mode == "4":
            print("\nRunning Syntactic Analysis on Completed Outputs...")
            try:
                run_syntax_analysis()
            except Exception as e:
                print(f"Error during syntactic analysis: {e}")
        elif mode == "5":
            print("\nComparing predictions with ground truth using cosine similarity...")
            try:
                run_similarity_vs_ground_truth()
            except Exception as e:
                print(f"Error during comparison: {e}")

        elif mode in {"exit", "quit", "q"}:
            print("Exiting. Goodbye!")
            break

        else:
            print("Invalid input. Please choose 1, 2, 3, 4, 5 or exit.")

if __name__ == "__main__":
    main()