import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any
import argparse
import os

def load_results(file_path: str) -> list[Dict[str, Any]]:
    """Load results from a JSONL file."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def compare_scores(base_results: List[Dict[str, Any]],
                   base_with_rag_results: List[Dict[str, Any]],
                   tuned_results: List[Dict[str, Any]],
                   tuned_with_rag_results: list[Dict[str, Any]]) -> Dict[str, Any]:
    # Ensure both have the same number of samples
    if len(base_results) != len(base_with_rag_results):
        print(f"Warning: Different number of samples - Base: {len(base_results)}, Base with RAG: {len(base_with_rag_results)}")
    if len(base_results) != len(tuned_results):
        print(f"Warning: Different number of samples - Base: {len(base_results)}, Tuned: {len(tuned_results)}")
    if len(base_results) != len(tuned_with_rag_results):
        print(f"Warning: Different number of samples - Base: {len(base_results)}, Tuned with RAG: {len(tuned_with_rag_results)}")
    
    base_scores = [item['score'] for item in base_results]
    base_with_rag_scores = [item['score'] for item in base_with_rag_results]
    tuned_scores = [item['score'] for item in tuned_results]
    tuned_with_rag_scores = [item['score'] for item in tuned_with_rag_results]

    # Base_scores = [14, 9, 15, 10, 12, 15, 8, 11, 7, 14, 14, 6, 11, 11, 11, 10, 14, 7, 8, 10, 12, 13, 10, 10, 11, 12, 11, 8, 3, 4, 10, 11, 12, 14, 13, 10, 9, 6, 7, 3, 11, 14, 10, 10, 6, 5, 12, 15, 5, 11, 4, 9, 5, 13, 13, 8, 12, 9, 10, 5, 8, 15, 7, 11, 14, 6, 3, 7, 12, 9, 10, 15, 8, 10, 7, 15, 4, 10, 9, 15, 12, 9, 8, 10, 11, 8, 5, 7, 11, 8, 7, 9, 14, 8, 8, 14, 9, 7, 15, 5]
    # Fine_tuned_scores = [14, 10, 14, 12, 12, 15, 11, 9, 13, 12, 13, 7, 3, 9, 10, 8, 14, 6, 15, 13, 14, 14, 11, 3, 6, 8, 8, 11, 9, 4, 12, 11, 15, 14, 9, 7, 5, 5, 3, 4, 9, 11, 12, 11, 8, 8, 6, 5, 7, 10, 10, 7, 9, 10, 10, 9, 15, 10, 14, 9, 11, 15, 9, 14, 10, 5, 10, 11, 10, 3, 12, 6, 5, 11, 9, 7, 7, 9, 9, 10, 15, 5, 6, 14, 3, 8, 13, 4, 13, 3, 11, 12, 6, 9, 7, 13, 11, 7, 13, 6]
    # Base_RAG_scores = [14, 9, 13, 10, 11, 15, 14, 13, 7, 12, 12, 15, 10, 9, 8, 13, 8, 13, 15, 13, 8, 14, 14, 8, 9, 11, 10, 12, 12, 10, 8, 15, 12, 13, 15, 12, 13, 11, 13, 3, 13, 11, 11, 12, 12, 10, 9, 13, 14, 13, 12, 13, 15, 13, 14, 9, 14, 14, 14, 13, 8, 13, 6, 13, 14, 12, 9, 13, 14, 6, 13, 15, 10, 10, 15, 14, 7, 14, 15, 5, 15, 12, 12, 14, 10, 8, 14, 11, 11, 8, 11, 14, 14, 13, 9, 15, 9, 13, 15, 12]
    # Fine_tuned_RAG_scores = [15, 10, 7, 10, 11, 15, 13, 10, 5, 13, 12, 13, 10, 10, 11, 10, 10, 10, 15, 14, 10, 15, 13, 9, 11, 4, 12, 9, 8, 8, 10, 13, 14, 12, 14, 12, 10, 9, 11, 3, 9, 14, 14, 12, 12, 11, 9, 14, 12, 10, 12, 13, 15, 13, 14, 7, 14, 14, 10, 12, 10, 13, 6, 6, 10, 9, 8, 10, 9, 5, 14, 14, 9, 11, 11, 14, 13, 15, 14, 10, 14, 12, 10, 14, 12, 10, 5, 4, 10, 11, 11, 14, 14, 13, 9, 15, 8, 8, 11, 14]

    data = [base_scores, base_with_rag_scores, tuned_scores, tuned_with_rag_scores]
    labels = ["Base Model", "Base Model + RAG", "Fine-tuned Model", "Fine-tuned Model + RAG"]

    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=labels, showmeans=True)

    plt.ylabel("Scores")
    plt.title("Model scores")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("model_scores_boxplot.png", dpi=300, bbox_inches="tight")

    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_results', type=str, default="output_base.jsonl")
    parser.add_argument('--base_with_rag_results', type=str, default="output_base_with_rag.jsonl")
    parser.add_argument('--tuned_results', type=str, default="output_final_no_rag.jsonl")
    parser.add_argument('--tuned_with_rag_results', type=str, default="output_final_with_rag.jsonl")
    
    args = parser.parse_args()
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_file = os.path.join(script_dir, args.base_results)
    base_with_rag_file = os.path.join(script_dir, args.base_with_rag_results)
    tuned_file = os.path.join(script_dir, args.tuned_results)
    tuned_with_rag_file = os.path.join(script_dir, args.tuned_with_rag_results)
    
    base_results = load_results(base_file)
    base_with_rag_results = load_results(base_with_rag_file)
    tuned_results = load_results(tuned_file)
    tuned_with_rag_results = load_results(tuned_with_rag_file)
    
    compare_scores(base_results, base_with_rag_results, tuned_results, tuned_with_rag_results)
