#!/usr/bin/env python3
"""
Main CLI runner for all evaluation suites.

Usage:
    python benchmarks/run_all.py --out reports/mcn_eval_20240101 --scales 100000 200000 300000
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
import json
import platform
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import suites
sys.path.insert(0, str(Path(__file__).parent))
from suite_a_beir import run_suite_a
from suite_b_conversational import run_suite_b
from suite_c_forever_memory import run_suite_c
from utils_report import (
    save_metrics_json, generate_results_md, generate_details_md,
    plot_latency_histogram, plot_recall_vs_scale, plot_compression_vs_scale,
    plot_recall_over_time, plot_latency_over_time, plot_compression_over_time
)


def main():
    parser = argparse.ArgumentParser(description="Run MCN v1 evaluation suites")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--scales", type=int, nargs="+", default=[100000, 200000, 300000],
                        help="Dataset scales to test")
    parser.add_argument("--beir", type=str, default="nfcorpus",
                        help="BEIR dataset name (default: nfcorpus)")
    parser.add_argument("--conv", type=str, default="openassistant",
                        help="Conversational dataset (default: openassistant)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-queries", type=int, default=200, help="Max queries per suite")
    parser.add_argument("--concurrent", type=int, default=1, help="Concurrent queries")
    parser.add_argument("--skip-a", action="store_true", help="Skip Suite A (BEIR)")
    parser.add_argument("--skip-b", action="store_true", help="Skip Suite B (Conversational)")
    parser.add_argument("--skip-c", action="store_true", help="Skip Suite C (Forever-Memory)")
    parser.add_argument("--qdrant", action="store_true", help="Include Qdrant baseline (requires Docker)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MCN v1 Comprehensive Evaluation")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"Scales: {args.scales}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    # Configuration
    config = {
        "scales": args.scales,
        "seed": args.seed,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "python_version": sys.version,
        "os": platform.system(),
        "max_queries": args.max_queries,
        "concurrent": args.concurrent,
    }
    
    commands = [f"python benchmarks/run_all.py {' '.join(sys.argv[1:])}"]
    
    # Run suites
    suite_results = {}
    
    # Suite A: BEIR
    if not args.skip_a:
        print("\n" + "="*80)
        print("Running Suite A: BEIR Dataset")
        print("="*80)
        try:
            suite_a_results = run_suite_a(
                dataset_name=args.beir,
                scales=args.scales,
                cache_dir=cache_dir,
                seed=args.seed,
                max_queries=args.max_queries,
                concurrent=args.concurrent
            )
            suite_results["Suite_A_BEIR"] = suite_a_results
        except Exception as e:
            print(f"Error in Suite A: {e}")
            import traceback
            traceback.print_exc()
    
    # Suite B: Conversational
    if not args.skip_b:
        print("\n" + "="*80)
        print("Running Suite B: Conversational Dataset")
        print("="*80)
        try:
            suite_b_results = run_suite_b(
                dataset_name=args.conv,
                scales=args.scales,
                cache_dir=cache_dir,
                seed=args.seed,
                max_queries=args.max_queries,
                concurrent=args.concurrent
            )
            suite_results["Suite_B_Conversational"] = suite_b_results
        except Exception as e:
            print(f"Error in Suite B: {e}")
            import traceback
            traceback.print_exc()
    
    # Suite C: Forever-Memory
    if not args.skip_c:
        print("\n" + "="*80)
        print("Running Suite C: Forever-Memory Simulation")
        print("="*80)
        try:
            suite_c_results = run_suite_c(
                total_vectors=args.scales[-1] if args.scales else 100000,
                wave_size=5000,
                dim=384,
                cache_dir=cache_dir,
                seed=args.seed,
                n_queries_per_step=50,
                concurrent=args.concurrent
            )
            suite_results["Suite_C_ForeverMemory"] = suite_c_results
        except Exception as e:
            print(f"Error in Suite C: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate reports
    print("\n" + "="*80)
    print("Generating Reports")
    print("="*80)
    
    # Save metrics JSON
    metrics_json = {
        "config": config,
        "suites": suite_results,
        "timestamp": datetime.now().isoformat()
    }
    save_metrics_json(metrics_json, output_dir / "metrics.json")
    
    # Generate RESULTS.md
    generate_results_md(suite_results, output_dir / "RESULTS.md", config)
    
    # Generate DETAILS.md
    generate_details_md(suite_results, output_dir / "DETAILS.md", config, commands)
    
    # Generate plots
    print("Generating plots...")
    
    # Latency histograms (from first suite with latencies)
    for suite_name, suite_data in suite_results.items():
        if "metrics" in suite_data and "mcn" in suite_data["metrics"]:
            # Get latencies from first scale
            if suite_data["metrics"]["mcn"]:
                # We'd need to store latencies in results - simplified for now
                pass
    
    # Recall vs scale plots
    for suite_name, suite_data in suite_results.items():
        if "scales" in suite_data and "metrics" in suite_data:
            scales = suite_data["scales"]
            recall_values = {}
            compression_values = {}
            
            for system in ["mcn", "faiss", "brute_force"]:
                if system in suite_data["metrics"]:
                    recalls = [m.get("recall@10", 0.0) for m in suite_data["metrics"][system]]
                    compressions = [m.get("compression_ratio", 1.0) for m in suite_data["metrics"][system]]
                    recall_values[system] = recalls
                    compression_values[system] = compressions
            
            if recall_values:
                plot_recall_vs_scale(
                    scales, recall_values,
                    output_dir / f"{suite_name}_recall_vs_scale.png"
                )
            
            if compression_values:
                plot_compression_vs_scale(
                    scales, compression_values,
                    output_dir / f"{suite_name}_compression_vs_scale.png"
                )
    
    # Forever-memory plots
    if "Suite_C_ForeverMemory" in suite_results:
        suite_c = suite_results["Suite_C_ForeverMemory"]
        if "results" in suite_c:
            results = suite_c["results"]
            time_steps = results["time_steps"]
            
            plot_recall_over_time(
                time_steps, results["recall@10"],
                output_dir / "forever_memory_recall_over_time.png"
            )
            
            plot_latency_over_time(
                time_steps, results["latency_p95"],
                output_dir / "forever_memory_latency_over_time.png"
            )
            
            plot_compression_over_time(
                time_steps, results["compression_ratio"],
                output_dir / "forever_memory_compression_over_time.png"
            )
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - RESULTS.md: Summary tables")
    print(f"  - DETAILS.md: Full configuration and results")
    print(f"  - metrics.json: Machine-readable metrics")
    print(f"  - *.png: Plots")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

