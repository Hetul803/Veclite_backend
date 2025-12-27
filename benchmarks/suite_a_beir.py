"""
Test Suite A: Harder BEIR Dataset Evaluation

Uses NFCorpus or TREC-COVID (harder BEIR datasets) with scaling to 100K-300K vectors.
"""
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False

from eval_harness import EmbeddingCache, evaluate_system


def load_beir_dataset(
    dataset_name: str,
    max_docs: Optional[int] = None,
    max_queries: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], List[str], Dict[str, List[str]], List[Dict[str, float]], List[str], List[str]]:
    """
    Load BEIR dataset.
    
    Returns:
        corpus_texts, query_texts, qrels_dict, qrels_list, corpus_ids, query_ids
    """
    if not BEIR_AVAILABLE:
        raise ImportError("BEIR not available. Install with: pip install beir")
    
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = f"./beir_data/{dataset_name}"
    
    # Download if needed
    if not os.path.exists(data_path) or not os.path.exists(f"{data_path}/corpus.jsonl"):
        print(f"Downloading {dataset_name}...")
        util.download_and_unzip(url, data_path)
        # Fix nested directory
        if os.path.exists(f"{data_path}/{dataset_name}"):
            import shutil
            for f in os.listdir(f"{data_path}/{dataset_name}"):
                src = f"{data_path}/{dataset_name}/{f}"
                dst = f"{data_path}/{f}"
                if not os.path.exists(dst):
                    shutil.move(src, dst)
    
    # Load data
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # Convert to lists
    corpus_ids = sorted(corpus.keys())
    corpus_texts = [corpus[doc_id]["text"] for doc_id in corpus_ids]
    
    query_ids = sorted(queries.keys())
    query_texts = [queries[q_id] for q_id in query_ids]
    
    # Limit size
    if max_docs:
        np.random.seed(seed)
        if len(corpus_ids) > max_docs:
            indices = np.random.choice(len(corpus_ids), max_docs, replace=False)
            indices = sorted(indices)
            corpus_ids = [corpus_ids[i] for i in indices]
            corpus_texts = [corpus_texts[i] for i in indices]
    
    if max_queries:
        query_ids = query_ids[:max_queries]
        query_texts = query_texts[:max_queries]
        # Filter qrels
        qrels = {q_id: qrels[q_id] for q_id in query_ids if q_id in qrels}
    
    # Convert qrels to list format
    qrels_list = []
    for q_id in query_ids:
        if q_id in qrels:
            qrels_list.append(qrels[q_id])
        else:
            qrels_list.append({})
    
    # Convert qrels to ground truth lists
    qrels_dict = {}
    for q_id in query_ids:
        if q_id in qrels:
            qrels_dict[q_id] = list(qrels[q_id].keys())
        else:
            qrels_dict[q_id] = []
    
    ground_truths = [qrels_dict.get(q_id, []) for q_id in query_ids]
    
    print(f"Loaded {len(corpus_texts):,} documents, {len(query_texts)} queries")
    
    return corpus_texts, query_texts, ground_truths, qrels_list, corpus_ids, query_ids


def run_suite_a(
    dataset_name: str,
    scales: List[int],
    cache_dir: Path,
    seed: int = 42,
    max_queries: int = 200,
    concurrent: int = 1
) -> Dict:
    """
    Run Test Suite A: Harder BEIR dataset.
    
    Args:
        dataset_name: BEIR dataset name (e.g., "nfcorpus", "trec-covid")
        scales: List of document counts to test (e.g., [100000, 200000, 300000])
        cache_dir: Directory for embedding cache
        seed: Random seed
        max_queries: Maximum number of queries
        concurrent: Concurrent queries
    
    Returns:
        Results dict
    """
    print(f"\n{'='*80}")
    print(f"Test Suite A: BEIR Dataset ({dataset_name})")
    print(f"{'='*80}\n")
    
    # Load full dataset
    print("Loading dataset...")
    corpus_texts, query_texts, ground_truths_full, qrels_list, corpus_ids, query_ids = load_beir_dataset(
        dataset_name, max_docs=None, max_queries=max_queries, seed=seed
    )
    
    # Limit queries to what we have ground truth for
    valid_queries = [i for i, gt in enumerate(ground_truths_full) if len(gt) > 0]
    if len(valid_queries) < max_queries:
        print(f"  Warning: Only {len(valid_queries)} queries have ground truth")
        query_texts = [query_texts[i] for i in valid_queries]
        ground_truths_full = [ground_truths_full[i] for i in valid_queries]
        qrels_list = [qrels_list[i] for i in valid_queries]
        query_ids = [query_ids[i] for i in valid_queries]
    
    # Build embeddings
    cache = EmbeddingCache(cache_dir, os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    
    print("\nBuilding embeddings...")
    corpus_embeddings = cache.encode(corpus_texts, f"{dataset_name}_corpus")
    query_embeddings = cache.encode(query_texts, f"{dataset_name}_queries")
    dim = corpus_embeddings.shape[1]
    
    # Results storage
    all_results = {
        "scales": [],
        "metrics": {
            "mcn": [],
            "faiss": [],
            "brute_force": []
        }
    }
    
    # Test each scale
    for scale in scales:
        if scale > len(corpus_embeddings):
            print(f"\nSkipping scale {scale:,} (dataset has only {len(corpus_embeddings):,} docs)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing scale: {scale:,} vectors")
        print(f"{'='*80}\n")
        
        # Subsample corpus
        np.random.seed(seed)
        indices = np.random.choice(len(corpus_embeddings), scale, replace=False)
        indices = sorted(indices)
        
        corpus_emb_subset = corpus_embeddings[indices]
        corpus_ids_subset = [corpus_ids[i] for i in indices]
        corpus_texts_subset = [corpus_texts[i] for i in indices]
        
        # Create metadata
        metadata = [
            {"original_idx": i, "id": corpus_ids_subset[i], "text": corpus_texts_subset[i]}
            for i in range(len(corpus_ids_subset))
        ]
        
        # Adjust ground truths to only include docs in subset
        corpus_ids_set = set(corpus_ids_subset)
        ground_truths = [
            [gt for gt in gt_list if gt in corpus_ids_set]
            for gt_list in ground_truths_full
        ]
        
        # Filter qrels
        qrels_subset = []
        for qrel in qrels_list:
            qrel_subset = {doc_id: score for doc_id, score in qrel.items() if doc_id in corpus_ids_set}
            qrels_subset.append(qrel_subset)
        
        # Evaluate MCN
        mcn_config = {
            "hot_buffer_size": 50,
            "beam_size": 200,
            "target_cluster_size": 15,
        }
        
        mcn_results = evaluate_system(
            "mcn",
            corpus_emb_subset,
            query_embeddings,
            ground_truths,
            metadata,
            qrels_subset,
            config=mcn_config,
            concurrent=concurrent
        )
        
        # Evaluate FAISS
        faiss_results = evaluate_system(
            "faiss",
            corpus_emb_subset,
            query_embeddings,
            ground_truths,
            metadata,
            qrels_subset,
            concurrent=concurrent
        )
        
        # Evaluate Brute-force (limited to 20K for sanity check)
        brute_results = evaluate_system(
            "brute_force",
            corpus_emb_subset,
            query_embeddings,
            ground_truths,
            metadata,
            qrels_subset,
            concurrent=concurrent,
            max_brute_force=20000
        )
        
        # Store results
        all_results["scales"].append(scale)
        all_results["metrics"]["mcn"].append({
            "recall@10": mcn_results["metrics"]["recall@10"],
            "recall@100": mcn_results["metrics"]["recall@100"],
            "mrr@10": mcn_results["metrics"]["mrr@10"],
            "ndcg@10": mcn_results["metrics"].get("ndcg@10", 0.0),
            "latency_p50": mcn_results["latency_stats"]["p50"],
            "latency_p95": mcn_results["latency_stats"]["p95"],
            "latency_p99": mcn_results["latency_stats"]["p99"],
            "build_time": mcn_results["build_time"],
            "memory_mb": mcn_results["memory_mb"],
            "storage_mb": mcn_results["storage_mb"],
            "compression_ratio": mcn_results["compression_ratio"],
            "qps": mcn_results["qps"],
        })
        all_results["metrics"]["faiss"].append({
            "recall@10": faiss_results["metrics"]["recall@10"],
            "recall@100": faiss_results["metrics"]["recall@100"],
            "mrr@10": faiss_results["metrics"]["mrr@10"],
            "ndcg@10": faiss_results["metrics"].get("ndcg@10", 0.0),
            "latency_p50": faiss_results["latency_stats"]["p50"],
            "latency_p95": faiss_results["latency_stats"]["p95"],
            "latency_p99": faiss_results["latency_stats"]["p99"],
            "build_time": faiss_results["build_time"],
            "memory_mb": faiss_results["memory_mb"],
            "storage_mb": faiss_results["storage_mb"],
            "compression_ratio": faiss_results["compression_ratio"],
            "qps": faiss_results["qps"],
        })
        all_results["metrics"]["brute_force"].append({
            "recall@10": brute_results["metrics"]["recall@10"],
            "recall@100": brute_results["metrics"]["recall@100"],
            "mrr@10": brute_results["metrics"]["mrr@10"],
            "ndcg@10": brute_results["metrics"].get("ndcg@10", 0.0),
            "latency_p50": brute_results["latency_stats"]["p50"],
            "latency_p95": brute_results["latency_stats"]["p95"],
            "latency_p99": brute_results["latency_stats"]["p99"],
            "build_time": brute_results["build_time"],
            "memory_mb": brute_results["memory_mb"],
            "storage_mb": brute_results["storage_mb"],
            "compression_ratio": brute_results["compression_ratio"],
            "qps": brute_results["qps"],
        })
    
    return {
        "suite_name": f"BEIR-{dataset_name}",
        "dataset": dataset_name,
        "scales": all_results["scales"],
        "metrics": all_results["metrics"],
        "n_queries": len(query_texts),
        "seed": seed,
    }

