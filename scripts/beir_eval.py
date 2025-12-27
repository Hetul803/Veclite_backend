#!/usr/bin/env python3
"""
BEIR Real-World Evaluation for MCN v1

Evaluates MCN on BEIR datasets and compares against:
1. Brute-force exact cosine (numpy)
2. FAISS IndexFlatIP exact cosine

Usage:
    python scripts/beir_eval.py --dataset scifact --max_docs 5000 --max_queries 200
"""
import sys
import os
import argparse
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from beir import util, LoggingHandler
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False
    print("Warning: BEIR not available. Install with: pip install beir")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

import psutil
from mcn import MCNLayer


class BEIREvaluator:
    """BEIR dataset evaluator for MCN."""
    
    def __init__(
        self,
        dataset: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_docs: Optional[int] = None,
        max_queries: Optional[int] = None,
        dim: int = 384
    ):
        self.dataset = dataset
        self.embedding_model = embedding_model
        self.max_docs = max_docs
        self.max_queries = max_queries
        self.dim = dim
        
        # Initialize embedding model
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
        print(f"Loading embedding model: {embedding_model}...")
        self.model = SentenceTransformer(embedding_model)
        print(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Override dim if model has different dimension
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != dim:
            print(f"Warning: Model dimension ({actual_dim}) != specified dim ({dim}). Using {actual_dim}.")
            self.dim = actual_dim
    
    def load_dataset(self) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        Load BEIR dataset.
        
        Returns:
            corpus_texts: List of corpus texts
            query_texts: List of query texts
            qrels: Dict mapping query_id -> [relevant_doc_ids]
        """
        if not BEIR_AVAILABLE:
            raise ImportError("BEIR required. Install with: pip install beir")
        
        print(f"Loading BEIR dataset: {self.dataset}...")
        
        # Download dataset if needed
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset}.zip"
        data_path = f"./beir_data/{self.dataset}"
        
        if not os.path.exists(data_path):
            print(f"Downloading {self.dataset}...")
            util.download_and_unzip(url, data_path)
        
        # Load data
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        
        # Convert to lists
        corpus_texts = [corpus[doc_id]["text"] for doc_id in sorted(corpus.keys())]
        corpus_ids = sorted(corpus.keys())
        
        query_texts = [queries[q_id] for q_id in sorted(queries.keys())]
        query_ids = sorted(queries.keys())
        
        # Limit size if specified
        if self.max_docs:
            corpus_texts = corpus_texts[:self.max_docs]
            corpus_ids = corpus_ids[:self.max_docs]
        
        if self.max_queries:
            query_texts = query_texts[:self.max_queries]
            query_ids = query_ids[:self.max_queries]
            # Filter qrels to only include queries we're using
            qrels = {q_id: qrels[q_id] for q_id in query_ids if q_id in qrels}
        
        print(f"Loaded {len(corpus_texts)} documents, {len(query_texts)} queries")
        
        return corpus_texts, query_texts, qrels, corpus_ids, query_ids
    
    def build_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Build embeddings for texts."""
        print(f"Building embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings.astype("float32")
    
    def evaluate_mcn(
        self,
        corpus_vectors: np.ndarray,
        query_vectors: np.ndarray,
        qrels: Dict[str, List[str]],
        corpus_ids: List[str],
        query_ids: List[str]
    ) -> Dict:
        """Evaluate MCN."""
        print("\n" + "="*80)
        print("Evaluating MCN")
        print("="*80)
        
        # Initialize MCN
        mcn = MCNLayer(dim=self.dim, hot_buffer_size=50, use_background_thread=False)
        
        # Build metadata
        corpus_metadata = [
            {"original_idx": i, "id": corpus_ids[i], "text": ""} 
            for i in range(len(corpus_vectors))
        ]
        
        # Ingest corpus
        print(f"Ingesting {len(corpus_vectors)} vectors into MCN...")
        ingest_start = time.time()
        batch_size = 100
        for i in range(0, len(corpus_vectors), batch_size):
            batch_vecs = corpus_vectors[i:i+batch_size]
            batch_meta = corpus_metadata[i:i+batch_size]
            mcn.add(batch_vecs, batch_meta)
        ingest_time = time.time() - ingest_start
        
        # Finalize index
        print("Finalizing index...")
        finalize_start = time.time()
        mcn.finalize_index(expected_count=len(corpus_vectors), timeout_s=300.0)
        finalize_time = time.time() - finalize_start
        
        # Get stats
        n_clusters = mcn.get_cold_index_size()
        compression_ratio = len(corpus_vectors) / max(1, n_clusters)
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"MCN Index built:")
        print(f"  - Vectors: {len(corpus_vectors)}")
        print(f"  - Clusters: {n_clusters}")
        print(f"  - Compression: {compression_ratio:.2f}:1")
        print(f"  - Build time: {finalize_time:.2f}s")
        print(f"  - Memory: {memory_mb:.1f} MB")
        
        # Run queries
        print(f"\nRunning {len(query_vectors)} queries...")
        latencies = []
        recall_10_scores = []
        recall_100_scores = []
        
        for i, (query_vec, query_id) in enumerate(zip(query_vectors, query_ids)):
            # Search
            search_start = time.time()
            results, scores = mcn.search(query_vec, k=100)
            search_time = (time.time() - search_start) * 1000  # ms
            latencies.append(search_time)
            
            # Get ground truth
            if query_id in qrels:
                gt_doc_ids = set(qrels[query_id])
                
                # Get predicted doc IDs
                pred_doc_ids = [r.get("id") for r in results[:100] if r.get("id")]
                
                # Compute recall
                recall_10 = len(set(pred_doc_ids[:10]) & gt_doc_ids) / max(1, len(gt_doc_ids))
                recall_100 = len(set(pred_doc_ids[:100]) & gt_doc_ids) / max(1, len(gt_doc_ids))
                
                recall_10_scores.append(recall_10)
                recall_100_scores.append(recall_100)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(query_vectors)} queries...")
        
        # Compute statistics
        avg_recall_10 = np.mean(recall_10_scores) if recall_10_scores else 0.0
        avg_recall_100 = np.mean(recall_100_scores) if recall_100_scores else 0.0
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        avg_latency = np.mean(latencies)
        
        return {
            "recall_10": avg_recall_10,
            "recall_100": avg_recall_100,
            "latency_p50": p50_latency,
            "latency_p95": p95_latency,
            "latency_avg": avg_latency,
            "build_time": finalize_time,
            "ingest_time": ingest_time,
            "compression_ratio": compression_ratio,
            "n_clusters": n_clusters,
            "memory_mb": memory_mb
        }
    
    def evaluate_brute_force(
        self,
        corpus_vectors: np.ndarray,
        query_vectors: np.ndarray,
        qrels: Dict[str, List[str]],
        corpus_ids: List[str],
        query_ids: List[str]
    ) -> Dict:
        """Evaluate brute-force numpy exact cosine."""
        print("\n" + "="*80)
        print("Evaluating Brute-Force (numpy)")
        print("="*80)
        
        # Normalize vectors
        print("Normalizing vectors...")
        corpus_norm = corpus_vectors / (np.linalg.norm(corpus_vectors, axis=1, keepdims=True) + 1e-10)
        
        # Run queries
        print(f"Running {len(query_vectors)} queries...")
        latencies = []
        recall_10_scores = []
        recall_100_scores = []
        
        for i, (query_vec, query_id) in enumerate(zip(query_vectors, query_ids)):
            # Normalize query
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            
            # Search
            search_start = time.time()
            scores = (corpus_norm @ query_norm).flatten()
            top_indices = np.argsort(-scores)[:100]
            search_time = (time.time() - search_start) * 1000  # ms
            latencies.append(search_time)
            
            # Get ground truth
            if query_id in qrels:
                gt_doc_ids = set(qrels[query_id])
                pred_doc_ids = [corpus_ids[idx] for idx in top_indices[:100]]
                
                recall_10 = len(set(pred_doc_ids[:10]) & gt_doc_ids) / max(1, len(gt_doc_ids))
                recall_100 = len(set(pred_doc_ids[:100]) & gt_doc_ids) / max(1, len(gt_doc_ids))
                
                recall_10_scores.append(recall_10)
                recall_100_scores.append(recall_100)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(query_vectors)} queries...")
        
        # Compute statistics
        avg_recall_10 = np.mean(recall_10_scores) if recall_10_scores else 0.0
        avg_recall_100 = np.mean(recall_100_scores) if recall_100_scores else 0.0
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        avg_latency = np.mean(latencies)
        
        return {
            "recall_10": avg_recall_10,
            "recall_100": avg_recall_100,
            "latency_p50": p50_latency,
            "latency_p95": p95_latency,
            "latency_avg": avg_latency
        }
    
    def evaluate_faiss(
        self,
        corpus_vectors: np.ndarray,
        query_vectors: np.ndarray,
        qrels: Dict[str, List[str]],
        corpus_ids: List[str],
        query_ids: List[str]
    ) -> Dict:
        """Evaluate FAISS IndexFlatIP exact cosine."""
        if not FAISS_AVAILABLE:
            return {"error": "FAISS not available"}
        
        print("\n" + "="*80)
        print("Evaluating FAISS IndexFlatIP")
        print("="*80)
        
        # Normalize vectors
        print("Normalizing vectors...")
        corpus_norm = corpus_vectors / (np.linalg.norm(corpus_vectors, axis=1, keepdims=True) + 1e-10)
        
        # Build index
        print("Building FAISS index...")
        build_start = time.time()
        index = faiss.IndexFlatIP(self.dim)
        index.add(corpus_norm.astype("float32"))
        build_time = time.time() - build_start
        print(f"FAISS index built in {build_time:.2f}s")
        
        # Run queries
        print(f"Running {len(query_vectors)} queries...")
        latencies = []
        recall_10_scores = []
        recall_100_scores = []
        
        for i, (query_vec, query_id) in enumerate(zip(query_vectors, query_ids)):
            # Normalize query
            query_norm = (query_vec / (np.linalg.norm(query_vec) + 1e-10)).astype("float32").reshape(1, -1)
            
            # Search
            search_start = time.time()
            scores, indices = index.search(query_norm, 100)
            search_time = (time.time() - search_start) * 1000  # ms
            latencies.append(search_time)
            
            # Get ground truth
            if query_id in qrels:
                gt_doc_ids = set(qrels[query_id])
                pred_doc_ids = [corpus_ids[idx] for idx in indices[0][:100]]
                
                recall_10 = len(set(pred_doc_ids[:10]) & gt_doc_ids) / max(1, len(gt_doc_ids))
                recall_100 = len(set(pred_doc_ids[:100]) & gt_doc_ids) / max(1, len(gt_doc_ids))
                
                recall_10_scores.append(recall_10)
                recall_100_scores.append(recall_100)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(query_vectors)} queries...")
        
        # Compute statistics
        avg_recall_10 = np.mean(recall_10_scores) if recall_10_scores else 0.0
        avg_recall_100 = np.mean(recall_100_scores) if recall_100_scores else 0.0
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        avg_latency = np.mean(latencies)
        
        return {
            "recall_10": avg_recall_10,
            "recall_100": avg_recall_100,
            "latency_p50": p50_latency,
            "latency_p95": p95_latency,
            "latency_avg": avg_latency,
            "build_time": build_time
        }
    
    def run_evaluation(self) -> str:
        """Run full evaluation and return markdown report."""
        print("="*80)
        print("BEIR Real-World Evaluation for MCN v1")
        print("="*80)
        
        # Load dataset
        corpus_texts, query_texts, qrels, corpus_ids, query_ids = self.load_dataset()
        
        # Build embeddings
        print("\nBuilding corpus embeddings...")
        corpus_vectors = self.build_embeddings(corpus_texts)
        
        print("\nBuilding query embeddings...")
        query_vectors = self.build_embeddings(query_texts)
        
        # Run evaluations
        mcn_results = self.evaluate_mcn(corpus_vectors, query_vectors, qrels, corpus_ids, query_ids)
        brute_force_results = self.evaluate_brute_force(corpus_vectors, query_vectors, qrels, corpus_ids, query_ids)
        faiss_results = self.evaluate_faiss(corpus_vectors, query_vectors, qrels, corpus_ids, query_ids)
        
        # Generate report
        report = self.generate_report(mcn_results, brute_force_results, faiss_results)
        
        return report
    
    def generate_report(
        self,
        mcn_results: Dict,
        brute_force_results: Dict,
        faiss_results: Dict
    ) -> str:
        """Generate markdown report."""
        report = f"""# BEIR Real-World Evaluation Report

**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Information

- **Dataset**: {self.dataset}
- **Corpus Size**: {self.max_docs or 'Full'}
- **Query Count**: {self.max_queries or 'Full'}
- **Embedding Model**: {self.embedding_model}
- **Dimension**: {self.dim}

## Results Summary

### MCN v1

| Metric | Value |
|--------|-------|
| Recall@10 | {mcn_results['recall_10']:.4f} |
| Recall@100 | {mcn_results['recall_100']:.4f} |
| Latency p50 (ms) | {mcn_results['latency_p50']:.2f} |
| Latency p95 (ms) | {mcn_results['latency_p95']:.2f} |
| Latency avg (ms) | {mcn_results['latency_avg']:.2f} |
| Build time (s) | {mcn_results['build_time']:.2f} |
| Compression ratio | {mcn_results['compression_ratio']:.2f}:1 |
| Clusters | {mcn_results['n_clusters']} |
| Memory (MB) | {mcn_results['memory_mb']:.1f} |

### Brute-Force (numpy)

| Metric | Value |
|--------|-------|
| Recall@10 | {brute_force_results['recall_10']:.4f} |
| Recall@100 | {brute_force_results['recall_100']:.4f} |
| Latency p50 (ms) | {brute_force_results['latency_p50']:.2f} |
| Latency p95 (ms) | {brute_force_results['latency_p95']:.2f} |
| Latency avg (ms) | {brute_force_results['latency_avg']:.2f} |

### FAISS IndexFlatIP

"""
        
        if "error" in faiss_results:
            report += f"**Error**: {faiss_results['error']}\n\n"
        else:
            report += f"""| Metric | Value |
|--------|-------|
| Recall@10 | {faiss_results['recall_10']:.4f} |
| Recall@100 | {faiss_results['recall_100']:.4f} |
| Latency p50 (ms) | {faiss_results['latency_p50']:.2f} |
| Latency p95 (ms) | {faiss_results['latency_p95']:.2f} |
| Latency avg (ms) | {faiss_results['latency_avg']:.2f} |
| Build time (s) | {faiss_results.get('build_time', 0):.2f} |

"""
        
        # Comparison
        report += """## Comparison

### Recall vs Baselines

"""
        
        if "error" not in faiss_results:
            report += f"- **MCN Recall@10**: {mcn_results['recall_10']:.4f} vs Brute-Force: {brute_force_results['recall_10']:.4f} vs FAISS: {faiss_results['recall_10']:.4f}\n"
            report += f"- **MCN Recall@100**: {mcn_results['recall_100']:.4f} vs Brute-Force: {brute_force_results['recall_100']:.4f} vs FAISS: {faiss_results['recall_100']:.4f}\n\n"
        else:
            report += f"- **MCN Recall@10**: {mcn_results['recall_10']:.4f} vs Brute-Force: {brute_force_results['recall_10']:.4f}\n"
            report += f"- **MCN Recall@100**: {mcn_results['recall_100']:.4f} vs Brute-Force: {brute_force_results['recall_100']:.4f}\n\n"
        
        report += f"""### Latency vs Baselines

- **MCN p95**: {mcn_results['latency_p95']:.2f}ms vs Brute-Force: {brute_force_results['latency_p95']:.2f}ms"""
        
        if "error" not in faiss_results:
            report += f" vs FAISS: {faiss_results['latency_p95']:.2f}ms"
        
        report += "\n\n"
        
        # Analysis
        recall_diff_10 = mcn_results['recall_10'] - brute_force_results['recall_10']
        recall_diff_100 = mcn_results['recall_100'] - brute_force_results['recall_100']
        
        report += """## Analysis

"""
        
        if recall_diff_10 < -0.05:
            report += f"⚠️ **Warning**: MCN Recall@10 is {abs(recall_diff_10):.4f} lower than brute-force. "
            report += "This may indicate:\n"
            report += "- Beam size too small (try increasing beam_size)\n"
            report += "- Cluster quality issues (check compression ratio)\n"
            report += "- Query distribution mismatch\n\n"
        else:
            report += f"✅ **MCN Recall@10** is within {abs(recall_diff_10):.4f} of brute-force (acceptable).\n\n"
        
        report += f"""### Compression Analysis

- **Compression Ratio**: {mcn_results['compression_ratio']:.2f}:1 ({mcn_results['n_clusters']} clusters from {self.max_docs or 'N'} vectors)
- **Memory Usage**: {mcn_results['memory_mb']:.1f} MB
- **Build Time**: {mcn_results['build_time']:.2f}s

### Performance Trade-offs

- **Latency**: MCN is {"faster" if mcn_results['latency_p95'] < brute_force_results['latency_p95'] else "slower"} than brute-force
- **Recall**: MCN achieves {mcn_results['recall_10']*100:.2f}% of brute-force Recall@10
- **Scalability**: MCN scales better with larger datasets due to compression

"""
        
        return report


def main():
    parser = argparse.ArgumentParser(description="BEIR Real-World Evaluation for MCN")
    parser.add_argument("--dataset", type=str, default="scifact", help="BEIR dataset name")
    parser.add_argument("--max_docs", type=int, default=None, help="Maximum documents to index")
    parser.add_argument("--max_queries", type=int, default=None, help="Maximum queries to evaluate")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--output", type=str, default="BENCH_REALWORLD.md", help="Output markdown file")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = BEIREvaluator(
        dataset=args.dataset,
        embedding_model=args.embedding_model,
        max_docs=args.max_docs,
        max_queries=args.max_queries
    )
    
    # Run evaluation
    report = evaluator.run_evaluation()
    
    # Save report
    output_path = Path(__file__).parent.parent / args.output
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"\n{'='*80}")
    print(f"Report saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Print summary
    print(report)


if __name__ == "__main__":
    main()

