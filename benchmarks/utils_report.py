"""
Report generation utilities.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def save_metrics_json(metrics: Dict[str, Any], output_path: Path):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def plot_latency_histogram(latencies: List[float], output_path: Path, title: str = "Latency Distribution"):
    """Plot latency histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(latencies, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.axvline(np.percentile(latencies, 50), color='r', linestyle='--', label='p50')
    ax.axvline(np.percentile(latencies, 95), color='g', linestyle='--', label='p95')
    ax.axvline(np.percentile(latencies, 99), color='b', linestyle='--', label='p99')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_recall_vs_scale(
    scales: List[int],
    recall_values: Dict[str, List[float]],
    output_path: Path,
    metric_name: str = "Recall@10"
):
    """Plot recall vs scale for multiple systems."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for system_name, recalls in recall_values.items():
        ax.plot(scales, recalls, marker='o', label=system_name, linewidth=2, markersize=8)
    ax.set_xlabel('Dataset Size (vectors)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs Dataset Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_compression_vs_scale(
    scales: List[int],
    compression_ratios: Dict[str, List[float]],
    output_path: Path
):
    """Plot compression ratio vs scale."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for system_name, ratios in compression_ratios.items():
        ax.plot(scales, ratios, marker='o', label=system_name, linewidth=2, markersize=8)
    ax.set_xlabel('Dataset Size (vectors)')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Ratio vs Dataset Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_recall_over_time(
    time_steps: List[int],
    recall_values: List[float],
    output_path: Path,
    metric_name: str = "Recall@10"
):
    """Plot recall over time steps (for forever-memory simulation)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps, recall_values, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Over Time (Forever-Memory Simulation)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_latency_over_time(
    time_steps: List[int],
    latency_p95: List[float],
    output_path: Path
):
    """Plot latency over time steps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps, latency_p95, marker='o', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('p95 Latency (ms)')
    ax.set_title('Latency Over Time (Forever-Memory Simulation)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_compression_over_time(
    time_steps: List[int],
    compression_ratios: List[float],
    output_path: Path
):
    """Plot compression ratio over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_steps, compression_ratios, marker='o', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Ratio Over Time (Forever-Memory Simulation)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_results_md(
    suite_results: Dict[str, Dict],
    output_path: Path,
    config: Dict[str, Any]
):
    """Generate RESULTS.md summary."""
    lines = [
        "# MCN v1 Evaluation Results",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Seed**: {config.get('seed', 'N/A')}",
        f"**Embedding Model**: {config.get('embedding_model', 'all-MiniLM-L6-v2')}",
        "",
        "## Summary",
        "",
    ]
    
    for suite_name, results in suite_results.items():
        lines.append(f"### {suite_name}")
        lines.append("")
        
        # Create table
        if "scales" in results:
            # Multi-scale results
            scales = results["scales"]
            systems = list(results["metrics"].keys())
            
            lines.append("| System | Scale | Recall@10 | Recall@100 | MRR@10 | p50 (ms) | p95 (ms) | Compression |")
            lines.append("|--------|-------|-----------|------------|--------|----------|----------|-------------|")
            
            for system in systems:
                for scale_idx, scale in enumerate(scales):
                    metrics = results["metrics"][system][scale_idx]
                    recall_10 = metrics.get("recall@10", 0.0)
                    recall_100 = metrics.get("recall@100", 0.0)
                    mrr = metrics.get("mrr@10", 0.0)
                    p50 = metrics.get("latency_p50", 0.0)
                    p95 = metrics.get("latency_p95", 0.0)
                    compression = metrics.get("compression_ratio", 0.0)
                    
                    lines.append(
                        f"| {system} | {scale:,} | {recall_10:.4f} | {recall_100:.4f} | "
                        f"{mrr:.4f} | {p50:.2f} | {p95:.2f} | {compression:.2f}:1 |"
                    )
        else:
            # Single result
            lines.append("| System | Recall@10 | Recall@100 | MRR@10 | p50 (ms) | p95 (ms) | Compression |")
            lines.append("|--------|-----------|------------|--------|----------|----------|-------------|")
            
            for system, metrics in results["metrics"].items():
                recall_10 = metrics.get("recall@10", 0.0)
                recall_100 = metrics.get("recall@100", 0.0)
                mrr = metrics.get("mrr@10", 0.0)
                p50 = metrics.get("latency_p50", 0.0)
                p95 = metrics.get("latency_p95", 0.0)
                compression = metrics.get("compression_ratio", 0.0)
                
                lines.append(
                    f"| {system} | {recall_10:.4f} | {recall_100:.4f} | {mrr:.4f} | "
                    f"{p50:.2f} | {p95:.2f} | {compression:.2f}:1 |"
                )
        
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def generate_details_md(
    suite_results: Dict[str, Dict],
    output_path: Path,
    config: Dict[str, Any],
    commands: List[str]
):
    """Generate DETAILS.md with full configuration."""
    lines = [
        "# Evaluation Details",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Commands",
        "",
    ]
    
    for cmd in commands:
        lines.append(f"```bash")
        lines.append(cmd)
        lines.append("```")
        lines.append("")
    
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: {config.get('python_version', 'N/A')}")
    lines.append(f"- OS: {config.get('os', 'N/A')}")
    lines.append("")
    
    lines.append("## Detailed Results")
    lines.append("")
    
    for suite_name, results in suite_results.items():
        lines.append(f"### {suite_name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(results, indent=2))
        lines.append("```")
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

