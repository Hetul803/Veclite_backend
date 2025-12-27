#!/usr/bin/env python3
"""
Run all evaluations: BEIR datasets at multiple scales.

Tests MCN on:
- SciFact (small): 5k, 10k vectors
- FiQA (medium): 10k, 50k vectors  
- Synthetic (large): 100k vectors

Outputs comprehensive results.
"""
import sys
import os
import time
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_command(cmd, description):
    """Run a command and capture output."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        elapsed = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return {
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "elapsed": 1800,
            "stdout": "",
            "stderr": "Timeout after 30 minutes"
        }
    except Exception as e:
        return {
            "success": False,
            "elapsed": time.time() - start_time,
            "stdout": "",
            "stderr": str(e)
        }

def main():
    """Run all evaluations."""
    print("="*80)
    print("MCN v1 Comprehensive Real-World Evaluation")
    print("="*80)
    
    results = {}
    
    # Test 1: Real-world smoke test
    print("\n[1/6] Real-World Smoke Test (SciFact 2k)")
    results["smoke"] = run_command(
        "python tests/test_realworld_smoke.py",
        "Smoke test on SciFact (2k docs, 50 queries)"
    )
    
    # Test 2: BEIR SciFact 5k
    print("\n[2/6] BEIR SciFact Evaluation (5k vectors)")
    results["scifact_5k"] = run_command(
        "python scripts/beir_eval.py --dataset scifact --max_docs 5000 --max_queries 200 --output BENCH_SCIFACT_5K.md",
        "SciFact 5k vectors evaluation"
    )
    
    # Test 3: BEIR SciFact 10k
    print("\n[3/6] BEIR SciFact Evaluation (10k vectors)")
    results["scifact_10k"] = run_command(
        "python scripts/beir_eval.py --dataset scifact --max_docs 10000 --max_queries 200 --output BENCH_SCIFACT_10K.md",
        "SciFact 10k vectors evaluation"
    )
    
    # Test 4: BEIR FiQA 10k
    print("\n[4/6] BEIR FiQA Evaluation (10k vectors)")
    results["fiqa_10k"] = run_command(
        "python scripts/beir_eval.py --dataset fiqa --max_docs 10000 --max_queries 200 --output BENCH_FIQA_10K.md",
        "FiQA 10k vectors evaluation"
    )
    
    # Test 5: BEIR FiQA 50k
    print("\n[5/6] BEIR FiQA Evaluation (50k vectors)")
    results["fiqa_50k"] = run_command(
        "python scripts/beir_eval.py --dataset fiqa --max_docs 50000 --max_queries 200 --output BENCH_FIQA_50K.md",
        "FiQA 50k vectors evaluation"
    )
    
    # Test 6: Large-scale synthetic 100k
    print("\n[6/6] Large-Scale Synthetic Test (100k vectors)")
    results["synthetic_100k"] = run_command(
        "python scripts/large_scale_test.py --vectors 100000 --queries 200",
        "Synthetic 100k vectors evaluation"
    )
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{name}: {status} ({result['elapsed']:.1f}s)")
        if not result["success"]:
            print(f"  Error: {result['stderr'][:200]}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

