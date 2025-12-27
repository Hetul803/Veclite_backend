#!/usr/bin/env python3
"""
Simplified BEIR evaluation that avoids multiprocessing issues.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import psutil
from mcn import MCNLayer


def l2_normalize(vecs):
    """L2 normalize vectors."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs / norms


def brute_force_search(query, vectors, k=10):
    """Brute-force cosine similarity."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    vectors_norm = l2_normalize(vectors)
    scores = (query_norm @ vectors_norm.T).flatten()
    top_indices = np.argsort(-scores)[:k]
    return top_indices, scores[top_indices]


def evaluate_dataset(dataset_name, max_docs=5000, max_queries=200, output_file=None):
    """Evaluate MCN on a BEIR dataset."""
    print(f"\n{'='*80}")
    print(f"BEIR Evaluation: {dataset_name}")
    print(f"Max docs: {max_docs}, Max queries: {max_queries}")
    print(f"{'='*80}\n")
    
    # Load dataset
    if not BEIR_AVAILABLE:
        print("ERROR: BEIR not available")
        return
    
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = f"./beir_data/{dataset_name}"
    
    if not os.path.exists(data_path) or not os.path.exists(f"{data_path}/corpus.jsonl"):
        print(f"Downloading {dataset_name}...")
        util.download_and_unzip(url, data_path)
        # Fix nested directory
        if os.path.exists(f"{data_path}/{dataset_name}"):
            import shutil
            for f in os.listdir(f"{data_path}/{dataset_name}"):
                shutil.move(f"{data_path}/{dataset_name}/{f}", f"{data_path}/{f}")
    
    print("Loading dataset...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # Convert to lists
    corpus_ids = sorted(corpus.keys())
    corpus_texts = [corpus[doc_id]["text"] for doc_id in corpus_ids]
    
    query_ids = sorted(queries.keys())
    query_texts = [queries[q_id] for q_id in query_ids]
    
    # Limit size
    if max_docs:
        corpus_ids = corpus_ids[:max_docs]
        corpus_texts = corpus_texts[:max_docs]
    
    if max_queries:
        query_ids = query_ids[:max_queries]
        query_texts = query_texts[:max_queries]
        qrels = {q_id: qrels[q_id] for q_id in query_ids if q_id in qrels}
    
    print(f"Loaded {len(corpus_texts)} documents, {len(query_texts)} queries")
    
    # Build embeddings
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("ERROR: sentence-transformers not available")
        return
    
    print("\nBuilding embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    dim = model.get_sentence_embedding_dimension()
    
    print("  Corpus embeddings...")
    corpus_embeddings = model.encode(corpus_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    corpus_embeddings = corpus_embeddings.astype("float32")
    corpus_embeddings = l2_normalize(corpus_embeddings)
    
    print("  Query embeddings...")
    query_embeddings = model.encode(query_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    query_embeddings = query_embeddings.astype("float32")
    query_embeddings = l2_normalize(query_embeddings)
    
    print(f"Embeddings: {corpus_embeddings.shape}, {query_embeddings.shape}")
    
    # Build ground truth (brute force for first 50 queries)
    print("\nComputing ground truth (first 50 queries)...")
    ground_truth = {}
    for i, query_vec in enumerate(query_embeddings[:50]):
        gt_indices, _ = brute_force_search(query_vec, corpus_embeddings, k=10)
        ground_truth[query_ids[i]] = [corpus_ids[idx] for idx in gt_indices]
    
    # Evaluate MCN
    print("\n" + "="*80)
    print("Evaluating MCN")
    print("="*80)
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    mcn = MCNLayer(dim=dim, hot_buffer_size=50, use_background_thread=False)
    
    # Ingest
    print("Ingesting vectors...")
    ingest_start = time.time()
    metadata = [{"original_idx": i, "id": corpus_ids[i]} for i in range(len(corpus_embeddings))]
    
    batch_size = 1000
    for i in range(0, len(corpus_embeddings), batch_size):
        batch_vecs = corpus_embeddings[i:i+batch_size]
        batch_meta = metadata[i:i+batch_size]
        mcn.add(batch_vecs, batch_meta)
        if (i + batch_size) % 5000 == 0:
            print(f"  Ingested {i + batch_size:,}/{len(corpus_embeddings):,}...")
    
    ingest_time = time.time() - ingest_start
    
    # Finalize
    print("Finalizing index...")
    finalize_start = time.time()
    mcn.finalize_index(expected_count=len(corpus_embeddings), timeout_s=600.0)
    finalize_time = time.time() - finalize_start
    
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    
    n_clusters = mcn.get_cold_index_size()
    compression_ratio = len(corpus_embeddings) / max(1, n_clusters)
    
    print(f"Index built: {n_clusters:,} clusters ({compression_ratio:.2f}:1 compression)")
    print(f"Build time: {finalize_time:.2f}s")
    print(f"Memory: {mem_used:.1f} MB")
    
    # Search
    print("\nRunning queries...")
    latencies = []
    recall_10_scores = []
    recall_100_scores = []
    
    for i, query_vec in enumerate(query_embeddings):
        search_start = time.time()
        results, scores = mcn.search(query_vec, k=10)
        search_time = (time.time() - search_start) * 1000
        latencies.append(search_time)
        
        # Compute recall (only for queries with ground truth)
        if query_ids[i] in ground_truth:
            pred_ids = [r.get("id") for r in results[:10] if r.get("id")]
            pred_set = set(pred_ids)
            gt_set = set(ground_truth[query_ids[i]][:10])
            recall_10 = len(pred_set & gt_set) / max(1, len(gt_set))
            recall_10_scores.append(recall_10)
            
            # Recall@100
            results_100, _ = mcn.search(query_vec, k=100)
            pred_ids_100 = [r.get("id") for r in results_100[:100] if r.get("id")]
            pred_set_100 = set(pred_ids_100)
            gt_set_100 = set(ground_truth[query_ids[i]][:100] if len(ground_truth[query_ids[i]]) >= 100 else ground_truth[query_ids[i]])
            recall_100 = len(pred_set_100 & gt_set_100) / max(1, len(gt_set_100))
            recall_100_scores.append(recall_100)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(query_embeddings)} queries...")
    
    # Statistics
    avg_recall_10 = np.mean(recall_10_scores) if recall_10_scores else 0.0
    avg_recall_100 = np.mean(recall_100_scores) if recall_100_scores else 0.0
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    avg_latency = np.mean(latencies)
    
    # Storage estimate
    child_store_size = len(corpus_embeddings) * dim * 4
    super_vectors_size = n_clusters * dim * 4
    csr_size = (n_clusters + 1) * 4 + len(corpus_embeddings) * 4
    total_storage_mb = (child_store_size + super_vectors_size + csr_size) / 1024 / 1024
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {dataset_name} ({len(corpus_embeddings):,} vectors)")
    print(f"{'='*80}")
    print(f"Recall@10: {avg_recall_10:.4f} (from {len(recall_10_scores)} queries)")
    print(f"Recall@100: {avg_recall_100:.4f} (from {len(recall_100_scores)} queries)")
    print(f"Latency:")
    print(f"  p50: {p50_latency:.2f}ms")
    print(f"  p95: {p95_latency:.2f}ms")
    print(f"  avg: {avg_latency:.2f}ms")
    print(f"Build:")
    print(f"  Ingest: {ingest_time:.2f}s")
    print(f"  Finalize: {finalize_time:.2f}s")
    print(f"  Total: {ingest_time + finalize_time:.2f}s")
    print(f"Resources:")
    print(f"  Memory: {mem_used:.1f} MB")
    print(f"  Storage: {total_storage_mb:.2f} MB")
    print(f"  Compression: {compression_ratio:.2f}:1")
    print(f"  Clusters: {n_clusters:,}")
    print(f"{'='*80}\n")
    
    # Write report
    if output_file:
        with open(output_file, "w") as f:
            f.write(f"# BEIR Evaluation: {dataset_name}\n\n")
            f.write(f"**Dataset**: {dataset_name}\n")
            f.write(f"**Corpus Size**: {len(corpus_embeddings):,} vectors\n")
            f.write(f"**Queries**: {len(query_embeddings)}\n")
            f.write(f"**Embedding Model**: all-MiniLM-L6-v2 ({dim}D)\n\n")
            f.write("## Results\n\n")
            f.write(f"- **Recall@10**: {avg_recall_10:.4f}\n")
            f.write(f"- **Recall@100**: {avg_recall_100:.4f}\n")
            f.write(f"- **Latency p50**: {p50_latency:.2f}ms\n")
            f.write(f"- **Latency p95**: {p95_latency:.2f}ms\n")
            f.write(f"- **Build Time**: {ingest_time + finalize_time:.2f}s\n")
            f.write(f"- **Memory**: {mem_used:.1f} MB\n")
            f.write(f"- **Storage**: {total_storage_mb:.2f} MB\n")
            f.write(f"- **Compression**: {compression_ratio:.2f}:1\n")
            f.write(f"- **Clusters**: {n_clusters:,}\n")
        print(f"Results written to {output_file}")
    
    return {
        "dataset": dataset_name,
        "n_vectors": len(corpus_embeddings),
        "n_queries": len(query_embeddings),
        "recall_10": avg_recall_10,
        "recall_100": avg_recall_100,
        "latency_p50": p50_latency,
        "latency_p95": p95_latency,
        "latency_avg": avg_latency,
        "build_time": ingest_time + finalize_time,
        "memory_mb": mem_used,
        "storage_mb": total_storage_mb,
        "compression_ratio": compression_ratio,
        "n_clusters": n_clusters
    }


def main():
    parser = argparse.ArgumentParser(description="BEIR evaluation for MCN")
    parser.add_argument("--dataset", type=str, default="scifact", help="BEIR dataset name")
    parser.add_argument("--max_docs", type=int, default=5000, help="Max documents")
    parser.add_argument("--max_queries", type=int, default=200, help="Max queries")
    parser.add_argument("--output", type=str, help="Output markdown file")
    
    args = parser.parse_args()
    
    evaluate_dataset(args.dataset, args.max_docs, args.max_queries, args.output)


if __name__ == "__main__":
    main()

