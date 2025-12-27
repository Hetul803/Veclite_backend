#!/usr/bin/env python3
"""
Smoke test for real-world BEIR evaluation.

Tests on small subset (2k docs, 50 queries) in <2 minutes.
Asserts Recall@10 >= 0.85 and self-match invariant.
"""
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from beir import util
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
    print("Warning: sentence-transformers not available")

from mcn import MCNLayer


def test_scifact_smoke():
    """Smoke test on SciFact subset."""
    if not BEIR_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Skipping test: BEIR or sentence-transformers not available")
        return
    
    print("="*80)
    print("Real-World Smoke Test: SciFact Subset")
    print("="*80)
    
    # Load small subset
    dataset = "scifact"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = f"./beir_data/{dataset}"
    
    if not os.path.exists(data_path):
        print(f"Downloading {dataset}...")
        util.download_and_unzip(url, data_path)
    
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # Limit to small subset
    max_docs = 2000
    max_queries = 50
    
    corpus_texts = [corpus[doc_id]["text"] for doc_id in sorted(corpus.keys())[:max_docs]]
    corpus_ids = sorted(corpus.keys())[:max_docs]
    
    query_texts = [queries[q_id] for q_id in sorted(queries.keys())[:max_queries]]
    query_ids = sorted(queries.keys())[:max_queries]
    qrels = {q_id: qrels[q_id] for q_id in query_ids if q_id in qrels}
    
    print(f"Loaded {len(corpus_texts)} documents, {len(query_texts)} queries")
    
    # Build embeddings
    print("Building embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    
    corpus_vectors = model.encode(corpus_texts, batch_size=32, show_progress_bar=False).astype("float32")
    query_vectors = model.encode(query_texts, batch_size=32, show_progress_bar=False).astype("float32")
    
    print(f"Embeddings built: {dim} dimensions")
    
    # Initialize MCN
    print("Initializing MCN...")
    mcn = MCNLayer(dim=dim, hot_buffer_size=50, use_background_thread=False)
    
    # Ingest corpus
    print("Ingesting vectors...")
    corpus_metadata = [
        {"original_idx": i, "id": corpus_ids[i]} 
        for i in range(len(corpus_vectors))
    ]
    
    batch_size = 100
    for i in range(0, len(corpus_vectors), batch_size):
        batch_vecs = corpus_vectors[i:i+batch_size]
        batch_meta = corpus_metadata[i:i+batch_size]
        mcn.add(batch_vecs, batch_meta)
    
    # Finalize
    print("Finalizing index...")
    mcn.finalize_index(expected_count=len(corpus_vectors), timeout_s=120.0)
    
    print(f"Index built: {mcn.get_cold_index_size()} clusters")
    
    # Test self-match invariant
    print("\nTesting self-match invariant...")
    test_query_idx = 0
    test_query_vec = corpus_vectors[test_query_idx]
    test_query_id = corpus_ids[test_query_idx]
    
    results, scores = mcn.search(test_query_vec, k=10)
    result_ids = [r.get("id") for r in results]
    
    assert test_query_id in result_ids, f"Self-match invariant violated: query vector {test_query_id} not in results"
    assert result_ids[0] == test_query_id, f"Self-match invariant violated: query vector not ranked #1"
    print("✅ Self-match invariant passed")
    
    # Run queries and compute recall
    print(f"\nRunning {len(query_vectors)} queries...")
    recall_10_scores = []
    
    for i, (query_vec, query_id) in enumerate(zip(query_vectors, query_ids)):
        results, scores = mcn.search(query_vec, k=10)
        
        if query_id in qrels:
            gt_doc_ids = set(qrels[query_id])
            pred_doc_ids = [r.get("id") for r in results[:10] if r.get("id")]
            recall_10 = len(set(pred_doc_ids) & gt_doc_ids) / max(1, len(gt_doc_ids))
            recall_10_scores.append(recall_10)
    
    avg_recall_10 = np.mean(recall_10_scores) if recall_10_scores else 0.0
    
    print(f"\nResults:")
    print(f"  - Average Recall@10: {avg_recall_10:.4f}")
    print(f"  - Target: >= 0.85")
    
    assert avg_recall_10 >= 0.85, f"Recall@10 {avg_recall_10:.4f} < 0.85 target"
    print("✅ Recall@10 >= 0.85 passed")
    
    print("\n" + "="*80)
    print("✅ All smoke tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_scifact_smoke()

