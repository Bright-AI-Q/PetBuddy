"""
After running llm_grader.py to generate model responses for the test samples and score them using LLM,
this module compares the result to output_base.jsonl.
"""
import argparse
import json
from typing import List, Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from a JSONL file."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def compare_scores(base_results: List[Dict[str, Any]], 
                   final_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare scores between base and final model results."""
    
    # Ensure both have the same number of samples
    if len(base_results) != len(final_results):
        print(f"Warning: Different number of samples - Base: {len(base_results)}, Final: {len(final_results)}")
    
    # Calculate statistics
    base_scores = [item['score'] for item in base_results]
    final_scores = [item['score'] for item in final_results]
    
    base_avg = np.mean(base_scores)
    final_avg = np.mean(final_scores)
    
    base_std = np.std(base_scores)
    final_std = np.std(final_scores)
    
    improvement = final_avg - base_avg
    improvement_pct = (improvement / base_avg) * 100 if base_avg > 0 else 0
    
    # Count wins/losses/ties
    wins = sum(1 for b, f in zip(base_scores, final_scores) if f > b)
    losses = sum(1 for b, f in zip(base_scores, final_scores) if f < b)
    ties = sum(1 for b, f in zip(base_scores, final_scores) if f == b)
    
    # Per-question comparison
    comparisons = []
    for base_item, final_item in zip(base_results, final_results):
        comparisons.append({
            'breed': base_item['breed'],
            'question': base_item['question'],
            'base_score': base_item['score'],
            'final_score': final_item['score'],
            'improvement': final_item['score'] - base_item['score'],
            'base_reason': base_item['reason'],
            'final_reason': final_item['reason']
        })
    
    return {
        'base_avg': base_avg,
        'final_avg': final_avg,
        'base_std': base_std,
        'final_std': final_std,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'total_samples': len(base_scores),
        'comparisons': comparisons
    }


def generate_comparison_report(comparison: Dict[str, Any]) -> str:
    """Generate the comparison report content as a string."""
    lines = []
    
    # Header
    lines.append("MODEL COMPARISON REPORT: Base vs Final")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall Statistics
    lines.append("📊 Overall Statistics:")
    lines.append("─" * 80)
    lines.append(f"  Base Model Average Score:   {comparison['base_avg']:.2f} ± {comparison['base_std']:.2f}")
    lines.append(f"  Final Model Average Score:  {comparison['final_avg']:.2f} ± {comparison['final_std']:.2f}")
    lines.append(f"  Improvement:                {comparison['improvement']:+.2f} ({comparison['improvement_pct']:+.1f}%)")
    lines.append("")
    
    # Win/Loss Record
    lines.append("🏆 Win/Loss Record:")
    lines.append("─" * 80)
    lines.append(f"  Final Model Wins:   {comparison['wins']:3d} ({comparison['wins']/comparison['total_samples']*100:.1f}%)")
    lines.append(f"  Final Model Losses: {comparison['losses']:3d} ({comparison['losses']/comparison['total_samples']*100:.1f}%)")
    lines.append(f"  Ties:               {comparison['ties']:3d} ({comparison['ties']/comparison['total_samples']*100:.1f}%)")
    lines.append(f"  Total Samples:      {comparison['total_samples']:3d}")
    lines.append("")
    
    # Top 5 Improvements
    lines.append("📈 Top 5 Improvements:")
    lines.append("─" * 80)
    sorted_improvements = sorted(comparison['comparisons'], key=lambda x: x['improvement'], reverse=True)
    for i, item in enumerate(sorted_improvements[:5], 1):
        lines.append(f"  {i}. {item['breed']} ({item['base_score']} → {item['final_score']}, +{item['improvement']})")
        lines.append(f"     Q: {item['question'][:70]}...")
    lines.append("")
    
    # Top 5 Declines
    lines.append("📉 Top 5 Declines:")
    lines.append("─" * 80)
    sorted_declines = sorted(comparison['comparisons'], key=lambda x: x['improvement'])
    for i, item in enumerate(sorted_declines[:5], 1):
        lines.append(f"  {i}. {item['breed']} ({item['base_score']} → {item['final_score']}, {item['improvement']})")
        lines.append(f"     Q: {item['question'][:70]}...")
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def print_comparison_report(comparison: Dict[str, Any]):
    """Print a detailed comparison report."""
    report = generate_comparison_report(comparison)
    print("\n" + report + "\n")


def save_detailed_comparison(comparison: Dict[str, Any], output_path: str = "detailed_comparison.json"):
    """Save detailed comparison to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"💾 Detailed comparison saved to: {output_path}")


def save_report_as_markdown(comparison: Dict[str, Any], output_path: str = "comparison_report.md"):
    """Save comparison report as a markdown file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Model Comparison Report: Base vs Final\n\n")
        f.write("*You can generate your own report using `llm_grade_comparer.py` with `--save_report` argument*\n\n")
        f.write("---\n\n")
        
        f.write("## 📊 Overall Statistics\n\n")
        f.write(f"- **Base Model Average Score:** {comparison['base_avg']:.2f} ± {comparison['base_std']:.2f}\n")
        f.write(f"- **Final Model Average Score:** {comparison['final_avg']:.2f} ± {comparison['final_std']:.2f}\n")
        f.write(f"- **Improvement:** {comparison['improvement']:+.2f} ({comparison['improvement_pct']:+.1f}%)\n\n")
        
        f.write("## 🏆 Win/Loss Record\n\n")
        f.write(f"- **Final Model Wins:** {comparison['wins']} ({comparison['wins']/comparison['total_samples']*100:.1f}%)\n")
        f.write(f"- **Final Model Losses:** {comparison['losses']} ({comparison['losses']/comparison['total_samples']*100:.1f}%)\n")
        f.write(f"- **Ties:** {comparison['ties']} ({comparison['ties']/comparison['total_samples']*100:.1f}%)\n")
        f.write(f"- **Total Samples:** {comparison['total_samples']}\n\n")
        
        f.write("## 📈 Top 5 Improvements\n\n")
        sorted_improvements = sorted(comparison['comparisons'], key=lambda x: x['improvement'], reverse=True)
        for i, item in enumerate(sorted_improvements[:5], 1):
            f.write(f"{i}. **{item['breed']}** ({item['base_score']} → {item['final_score']}, +{item['improvement']})\n")
            f.write(f"   - *Question:* {item['question']}\n\n")
        
        f.write("## 📉 Top 5 Declines\n\n")
        sorted_declines = sorted(comparison['comparisons'], key=lambda x: x['improvement'])
        for i, item in enumerate(sorted_declines[:5], 1):
            f.write(f"{i}. **{item['breed']}** ({item['base_score']} → {item['final_score']}, {item['improvement']})\n")
            f.write(f"   - *Question:* {item['question']}\n\n")
    
    print(f"📄 Comparison report saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model', type=str, default="output_final.jsonl")
    parser.add_argument('--verbose', action='store_true', help='Save detailed comparison to JSON file')
    parser.add_argument('--save_report', action='store_true', help='Save comparison report as markdown file')
    args = parser.parse_args()

    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_file = os.path.join(script_dir, "output_base.jsonl")
    final_file = os.path.join(script_dir, args.output_model)
    
    # Load results
    print("Loading results...")
    base_results = load_results(base_file)
    final_results = load_results(final_file)
    
    print(f"Loaded {len(base_results)} base results and {len(final_results)} final results")
    
    # Compare scores
    comparison = compare_scores(base_results, final_results)
    
    # Print report
    print_comparison_report(comparison)
    
    # Save detailed comparison only if verbose flag is set
    if args.verbose:
        save_detailed_comparison(comparison, os.path.join(script_dir, "detailed_comparison.json"))
    
    # Save report as markdown if save-report flag is set
    if args.save_report:
        save_report_as_markdown(comparison, os.path.join(script_dir, "comparison_report.md"))