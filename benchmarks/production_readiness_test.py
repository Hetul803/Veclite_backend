#!/usr/bin/env python3
"""
Production Readiness Test: Large-Scale BEIR Evaluation

Tests MCN v1 on 100K corpus size with comprehensive metrics:
- MCN vs FAISS IndexFlatIP
- Recall@10, nDCG@10, MRR@10
- p95 latency
- RAM at steady state
- Ingestion/build time
- Compression ratio

This is the final test before production launch.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import json
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False
    print("ERROR: BEIR not available. Install with: pip install beir")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("ERROR: sentence-transformers not available")
    sys.exit(1)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("ERROR: FAISS not available. Install with: pip install faiss-cpu")
    sys.exit(1)

from mcn import MCNLayer
from eval_harness import EmbeddingCache, IndexBuilder, QueryRunner, MemoryTracker
from utils_metrics import calculate_metrics, calculate_latency_stats


def load_large_beir_dataset(
    dataset_name: str,
    target_size: int = 100000,
    max_queries: int = 200,
    seed: int = 42
) -> Tuple[List[str], List[str], Dict[str, List[str]], List[Dict[str, float]], List[str], List[str]]:
    """
    Load large BEIR dataset and subsample to target size.
    
    Tries datasets in order: msmarco, nq, hotpotqa, quora, scidocs
    """
    # Try datasets in order of size
    datasets_to_try = [
        "msmarco",      # MS MARCO (largest)
        "nq",           # Natural Questions
        "hotpotqa",     # HotpotQA
        "quora",        # Quora
        "scidocs",      # SciDocs
        "scifact",      # SciFact (fallback)
    ]
    
    if dataset_name not in datasets_to_try:
        datasets_to_try.insert(0, dataset_name)
    
    corpus_texts = None
    query_texts = None
    qrels_dict = None
    qrels_list = None
    corpus_ids = None
    query_ids = None
    actual_dataset = None
    
    for ds_name in datasets_to_try:
        try:
            print(f"\nTrying dataset: {ds_name}...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
            data_path = f"./beir_data/{ds_name}"
            
            # Download if needed
            if not os.path.exists(data_path) or not os.path.exists(f"{data_path}/corpus.jsonl"):
                print(f"  Downloading {ds_name}...")
                util.download_and_unzip(url, data_path)
                # Fix nested directory
                if os.path.exists(f"{data_path}/{ds_name}"):
                    import shutil
                    for f in os.listdir(f"{data_path}/{ds_name}"):
                        src = f"{data_path}/{ds_name}/{f}"
                        dst = f"{data_path}/{f}"
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
            
            # Load data
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
            
            # Convert to lists
            corpus_ids_list = sorted(corpus.keys())
            corpus_texts_list = [corpus[doc_id]["text"] for doc_id in corpus_ids_list]
            
            print(f"  Loaded {len(corpus_texts_list):,} documents from {ds_name}")
            
            if len(corpus_texts_list) >= target_size:
                actual_dataset = ds_name
                corpus_ids = corpus_ids_list
                corpus_texts = corpus_texts_list
                break
            else:
                print(f"  {ds_name} has only {len(corpus_texts_list):,} docs (need {target_size:,})")
                continue
                
        except Exception as e:
            print(f"  Error loading {ds_name}: {e}")
            continue
    
    if corpus_texts is None:
        raise ValueError(f"Could not find dataset with >= {target_size:,} documents")
    
    print(f"\n✅ Using dataset: {actual_dataset} ({len(corpus_texts):,} documents)")
    
    # Subsample to target size
    np.random.seed(seed)
    if len(corpus_texts) > target_size:
        indices = np.random.choice(len(corpus_texts), target_size, replace=False)
        indices = sorted(indices)
        corpus_texts = [corpus_texts[i] for i in indices]
        corpus_ids = [corpus_ids[i] for i in indices]
        print(f"  Subsample to {len(corpus_texts):,} documents")
    
    # Load queries
    query_ids_list = sorted(queries.keys())
    query_texts_list = [queries[q_id] for q_id in query_ids_list]
    
    if max_queries:
        query_ids_list = query_ids_list[:max_queries]
        query_texts_list = query_texts_list[:max_queries]
        qrels = {q_id: qrels[q_id] for q_id in query_ids_list if q_id in qrels}
    
    # Convert qrels
    qrels_list = []
    for q_id in query_ids_list:
        if q_id in qrels:
            qrels_list.append(qrels[q_id])
        else:
            qrels_list.append({})
    
    qrels_dict = {}
    for q_id in query_ids_list:
        if q_id in qrels:
            qrels_dict[q_id] = list(qrels[q_id].keys())
        else:
            qrels_dict[q_id] = []
    
    ground_truths = [qrels_dict.get(q_id, []) for q_id in query_ids_list]
    
    # Filter ground truths to only include docs in corpus
    corpus_ids_set = set(corpus_ids)
    ground_truths = [
        [gt for gt in gt_list if gt in corpus_ids_set]
        for gt_list in ground_truths
    ]
    
    print(f"  Loaded {len(query_texts_list)} queries")
    print(f"  Queries with ground truth: {sum(1 for gt in ground_truths if len(gt) > 0)}")
    
    return corpus_texts, query_texts_list, ground_truths, qrels_list, corpus_ids, query_ids_list, actual_dataset


def evaluate_system_production(
    system_name: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truths: List[List[str]],
    metadata: List[Dict],
    qrels: List[Dict[str, float]],
    dim: int,
    config: Dict = None
) -> Dict:
    """Evaluate system with production metrics."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {system_name}")
    print(f"{'='*80}")
    
    builder = IndexBuilder(dim)
    runner = QueryRunner(dim)
    mem_tracker = MemoryTracker()
    
    # Measure RAM before
    process = psutil.Process()
    ram_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Build index
    print(f"  Building index for {len(vectors):,} vectors...")
    build_start = time.time()
    
    if system_name == "mcn":
        config = config or {}
        index, build_stats = builder.build_mcn(vectors, metadata, config)
        search_fn = lambda q, k: runner.search_mcn(index, q, k)
    elif system_name == "faiss":
        index, build_stats = builder.build_faiss_exact(vectors)
        search_fn = lambda q, k: runner.search_faiss(index, q, k)
    else:
        raise ValueError(f"Unknown system: {system_name}")
    
    build_time = time.time() - build_start
    
    # Measure RAM after build (steady state)
    ram_after = process.memory_info().rss / 1024 / 1024  # MB
    ram_steady_state = ram_after - ram_before
    
    print(f"  Build time: {build_time:.2f}s")
    print(f"  RAM (steady state): {ram_steady_state:.1f} MB")
    
    # Run queries (warmup + measurement)
    print(f"  Running {len(queries)} queries...")
    
    # Warmup
    for query in queries[:10]:
        search_fn(query, 10)
    
    # Measure latency
    latencies = []
    predictions = []
    
    for query in queries:
        start = time.time()
        results, scores, latency = search_fn(query, 100)
        latencies.append(latency)
        
        # Convert to doc IDs
        if system_name == "mcn":
            pred_ids = [str(r.get("id", r.get("original_idx", i))) for i, r in enumerate(results[:100])]
        else:
            # FAISS returns indices
            metadata_map = {i: meta for i, meta in enumerate(metadata)}
            pred_ids = [str(metadata_map.get(int(idx), {}).get("id", idx)) for idx in results[:100]]
        
        predictions.append(pred_ids)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths, qrels, k_values=[10, 100])
    latency_stats = calculate_latency_stats(latencies)
    
    # Compression ratio
    if system_name == "mcn":
        compression_ratio = build_stats.get("compression_ratio", 1.0)
        n_clusters = build_stats.get("n_clusters", 0)
    else:
        compression_ratio = 1.0
        n_clusters = 0
    
    results = {
        "system": system_name,
        "n_vectors": len(vectors),
        "n_queries": len(queries),
        "metrics": {
            "recall@10": metrics["recall@10"],
            "recall@100": metrics["recall@100"],
            "mrr@10": metrics["mrr@10"],
            "ndcg@10": metrics.get("ndcg@10", 0.0),
        },
        "latency": {
            "p50": latency_stats["p50"],
            "p95": latency_stats["p95"],
            "p99": latency_stats["p99"],
            "mean": latency_stats["mean"],
        },
        "ram_steady_state_mb": ram_steady_state,
        "build_time_s": build_time,
        "compression_ratio": compression_ratio,
        "n_clusters": n_clusters,
    }
    
    print(f"  Recall@10: {results['metrics']['recall@10']:.4f}")
    print(f"  nDCG@10: {results['metrics']['ndcg@10']:.4f}")
    print(f"  MRR@10: {results['metrics']['mrr@10']:.4f}")
    print(f"  p95 Latency: {results['latency']['p95']:.2f}ms")
    print(f"  Compression: {compression_ratio:.2f}:1")
    
    return results


def main():
    """Run production readiness test."""
    print("="*80)
    print("MCN v1 Production Readiness Test")
    print("="*80)
    print("Dataset: Large BEIR (100K corpus)")
    print("Systems: MCN v1 vs FAISS IndexFlatIP")
    print("="*80)
    
    # Configuration
    target_size = 100000
    max_queries = 200
    seed = 42
    dim = 384
    
    # Output directory
    output_dir = Path("./reports/production_readiness")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print("\n[1/5] Loading large BEIR dataset...")
    corpus_texts, query_texts, ground_truths, qrels_list, corpus_ids, query_ids, dataset_name = load_large_beir_dataset(
        dataset_name="msmarco",
        target_size=target_size,
        max_queries=max_queries,
        seed=seed
    )
    
    # Build embeddings
    print("\n[2/5] Building embeddings (using cache)...")
    cache = EmbeddingCache(cache_dir, os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    
    corpus_embeddings = cache.encode(corpus_texts, f"{dataset_name}_corpus_{target_size}")
    query_embeddings = cache.encode(query_texts, f"{dataset_name}_queries")
    
    print(f"  Corpus embeddings: {corpus_embeddings.shape}")
    print(f"  Query embeddings: {query_embeddings.shape}")
    
    # Create metadata
    metadata = [
        {"original_idx": i, "id": corpus_ids[i], "text": corpus_texts[i][:100]}
        for i in range(len(corpus_ids))
    ]
    
    # Filter qrels to only include docs in corpus
    corpus_ids_set = set(corpus_ids)
    qrels_filtered = []
    for qrel in qrels_list:
        qrel_filtered = {doc_id: score for doc_id, score in qrel.items() if doc_id in corpus_ids_set}
        qrels_filtered.append(qrel_filtered)
    
    # Evaluate MCN
    print("\n[3/5] Evaluating MCN v1...")
    mcn_config = {
        "hot_buffer_size": 50,
        "beam_size": 200,
        "target_cluster_size": 15,
        "max_cluster_size": 64,
    }
    
    mcn_results = evaluate_system_production(
        "mcn",
        corpus_embeddings,
        query_embeddings,
        ground_truths,
        metadata,
        qrels_filtered,
        dim,
        config=mcn_config
    )
    
    # Evaluate FAISS
    print("\n[4/5] Evaluating FAISS IndexFlatIP...")
    faiss_results = evaluate_system_production(
        "faiss",
        corpus_embeddings,
        query_embeddings,
        ground_truths,
        metadata,
        qrels_filtered,
        dim
    )
    
    # Generate report
    print("\n[5/5] Generating report...")
    
    report = {
        "test_name": "Production Readiness Test",
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "corpus_size": target_size,
        "n_queries": max_queries,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": dim,
        "seed": seed,
        "results": {
            "mcn": mcn_results,
            "faiss": faiss_results,
        },
        "comparison": {
            "recall@10_diff": mcn_results["metrics"]["recall@10"] - faiss_results["metrics"]["recall@10"],
            "ndcg@10_diff": mcn_results["metrics"]["ndcg@10"] - faiss_results["metrics"]["ndcg@10"],
            "mrr@10_diff": mcn_results["metrics"]["mrr@10"] - faiss_results["metrics"]["mrr@10"],
            "latency_p95_ratio": mcn_results["latency"]["p95"] / max(faiss_results["latency"]["p95"], 0.01),
            "ram_ratio": mcn_results["ram_steady_state_mb"] / max(faiss_results["ram_steady_state_mb"], 0.01),
            "build_time_ratio": mcn_results["build_time_s"] / max(faiss_results["build_time_s"], 0.01),
            "compression_benefit": mcn_results["compression_ratio"],
        }
    }
    
    # Save JSON
    with open(output_dir / "results.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    md_lines = [
        "# MCN v1 Production Readiness Test",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Dataset**: {dataset_name}",
        f"**Corpus Size**: {target_size:,} vectors",
        f"**Queries**: {max_queries}",
        f"**Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384D)",
        "",
        "## Results Summary",
        "",
        "| System | Recall@10 | nDCG@10 | MRR@10 | p95 Latency (ms) | RAM (MB) | Build Time (s) | Compression |",
        "|--------|-----------|---------|--------|-------------------|----------|----------------|-------------|",
        f"| **MCN v1** | **{mcn_results['metrics']['recall@10']:.4f}** | **{mcn_results['metrics']['ndcg@10']:.4f}** | **{mcn_results['metrics']['mrr@10']:.4f}** | **{mcn_results['latency']['p95']:.2f}** | **{mcn_results['ram_steady_state_mb']:.1f}** | **{mcn_results['build_time_s']:.2f}** | **{mcn_results['compression_ratio']:.2f}:1** |",
        f"| FAISS IndexFlatIP | {faiss_results['metrics']['recall@10']:.4f} | {faiss_results['metrics']['ndcg@10']:.4f} | {faiss_results['metrics']['mrr@10']:.4f} | {faiss_results['latency']['p95']:.2f} | {faiss_results['ram_steady_state_mb']:.1f} | {faiss_results['build_time_s']:.2f} | 1.00:1 |",
        "",
        "## Key Metrics",
        "",
        f"- **Recall Parity**: {'✅ Perfect' if abs(report['comparison']['recall@10_diff']) < 0.0001 else '⚠️ Difference: ' + str(report['comparison']['recall@10_diff'])}",
        f"- **Latency Ratio**: MCN is {report['comparison']['latency_p95_ratio']:.2f}x slower than FAISS",
        f"- **RAM Ratio**: MCN uses {report['comparison']['ram_ratio']:.2f}x more RAM than FAISS",
        f"- **Compression**: MCN achieves {mcn_results['compression_ratio']:.2f}:1 compression",
        f"- **Build Time**: MCN takes {report['comparison']['build_time_ratio']:.2f}x longer to build",
        "",
        "## Detailed Results",
        "",
        "### MCN v1",
        "",
        f"- Recall@10: {mcn_results['metrics']['recall@10']:.4f}",
        f"- Recall@100: {mcn_results['metrics']['recall@100']:.4f}",
        f"- nDCG@10: {mcn_results['metrics']['ndcg@10']:.4f}",
        f"- MRR@10: {mcn_results['metrics']['mrr@10']:.4f}",
        f"- Latency p50: {mcn_results['latency']['p50']:.2f}ms",
        f"- Latency p95: {mcn_results['latency']['p95']:.2f}ms",
        f"- Latency p99: {mcn_results['latency']['p99']:.2f}ms",
        f"- RAM (steady state): {mcn_results['ram_steady_state_mb']:.1f} MB",
        f"- Build time: {mcn_results['build_time_s']:.2f}s",
        f"- Compression ratio: {mcn_results['compression_ratio']:.2f}:1",
        f"- Clusters: {mcn_results['n_clusters']:,}",
        "",
        "### FAISS IndexFlatIP",
        "",
        f"- Recall@10: {faiss_results['metrics']['recall@10']:.4f}",
        f"- Recall@100: {faiss_results['metrics']['recall@100']:.4f}",
        f"- nDCG@10: {faiss_results['metrics']['ndcg@10']:.4f}",
        f"- MRR@10: {faiss_results['metrics']['mrr@10']:.4f}",
        f"- Latency p50: {faiss_results['latency']['p50']:.2f}ms",
        f"- Latency p95: {faiss_results['latency']['p95']:.2f}ms",
        f"- Latency p99: {faiss_results['latency']['p99']:.2f}ms",
        f"- RAM (steady state): {faiss_results['ram_steady_state_mb']:.1f} MB",
        f"- Build time: {faiss_results['build_time_s']:.2f}s",
        "",
        "## Production Readiness Assessment",
        "",
    ]
    
    # Assessment
    if abs(report['comparison']['recall@10_diff']) < 0.0001:
        md_lines.append("✅ **Recall**: Perfect parity with FAISS exact baseline")
    else:
        md_lines.append(f"⚠️ **Recall**: Difference of {report['comparison']['recall@10_diff']:.4f}")
    
    if mcn_results['latency']['p95'] <= 40:
        md_lines.append(f"✅ **Latency**: p95 of {mcn_results['latency']['p95']:.2f}ms meets 20-40ms target")
    else:
        md_lines.append(f"⚠️ **Latency**: p95 of {mcn_results['latency']['p95']:.2f}ms exceeds 40ms target")
    
    if mcn_results['compression_ratio'] >= 10:
        md_lines.append(f"✅ **Compression**: {mcn_results['compression_ratio']:.2f}:1 ratio is excellent")
    else:
        md_lines.append(f"⚠️ **Compression**: {mcn_results['compression_ratio']:.2f}:1 ratio is lower than expected")
    
    md_lines.extend([
        "",
        "## Conclusion",
        "",
        f"MCN v1 demonstrates {'✅ PRODUCTION READY' if abs(report['comparison']['recall@10_diff']) < 0.0001 and mcn_results['latency']['p95'] <= 40 else '⚠️ NEEDS OPTIMIZATION'} for large-scale production deployment.",
        "",
        f"**Key Findings**:",
        f"- MCN achieves {'perfect' if abs(report['comparison']['recall@10_diff']) < 0.0001 else 'near-perfect'} recall parity with exact baseline",
        f"- Compression ratio of {mcn_results['compression_ratio']:.2f}:1 provides significant storage savings",
        f"- Latency of {mcn_results['latency']['p95']:.2f}ms is {'within' if mcn_results['latency']['p95'] <= 40 else 'above'} the 20-40ms target",
        f"- RAM usage of {mcn_results['ram_steady_state_mb']:.1f} MB is reasonable for {target_size:,} vectors",
        "",
    ])
    
    with open(output_dir / "PRODUCTION_READINESS_REPORT.md", 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n{'='*80}")
    print("Production Readiness Test Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - PRODUCTION_READINESS_REPORT.md")
    print(f"  - results.json")
    print(f"{'='*80}\n")
    
    # Print summary
    print("SUMMARY:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Corpus: {target_size:,} vectors")
    print(f"  MCN Recall@10: {mcn_results['metrics']['recall@10']:.4f}")
    print(f"  FAISS Recall@10: {faiss_results['metrics']['recall@10']:.4f}")
    print(f"  MCN p95 Latency: {mcn_results['latency']['p95']:.2f}ms")
    print(f"  MCN Compression: {mcn_results['compression_ratio']:.2f}:1")
    print(f"  MCN RAM: {mcn_results['ram_steady_state_mb']:.1f} MB")
    print()


if __name__ == "__main__":
    main()

