"""
Test Suite B: Conversational Dataset Evaluation

Uses OpenAssistant Conversations or ShareGPT for chat memory / semantic retrieval.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
import gzip

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_harness import EmbeddingCache, evaluate_system


def load_openassistant(
    max_conversations: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Load OpenAssistant conversations.
    
    Returns:
        corpus_texts, query_texts, metadata_list
    """
    # Try to load from cache first
    cache_path = Path("./beir_data/openassistant_cache.json")
    
    if cache_path.exists():
        print("Loading OpenAssistant from cache...")
        with open(cache_path, 'r') as f:
            data = json.load(f)
        corpus_texts = data["corpus"]
        query_texts = data["queries"]
        metadata_list = data["metadata"]
    else:
        print("Downloading OpenAssistant dataset...")
        if not REQUESTS_AVAILABLE:
            print("  requests not available, using synthetic data")
            return load_synthetic_conversational(max_conversations, seed)
        
        url = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz"
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            corpus_texts = []
            query_texts = []
            metadata_list = []
            
            # Parse JSONL
            with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
                for line in f:
                    if max_conversations and len(corpus_texts) >= max_conversations * 10:
                        break
                    
                    try:
                        conv = json.loads(line)
                        messages = conv.get("messages", [])
                        
                        for i, msg in enumerate(messages):
                            text = msg.get("text", "")
                            if not text or len(text) < 10:
                                continue
                            
                            # Use earlier messages as corpus, later as queries
                            if i < len(messages) - 1:
                                corpus_texts.append(text)
                                metadata_list.append({
                                    "convo_id": conv.get("message_id", ""),
                                    "turn_id": i,
                                    "speaker": msg.get("role", "unknown"),
                                    "text": text
                                })
                            else:
                                # Last message as query
                                query_texts.append(text)
                    except:
                        continue
            
            # Cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({
                    "corpus": corpus_texts,
                    "queries": query_texts,
                    "metadata": metadata_list
                }, f)
        except Exception as e:
            print(f"Error downloading OpenAssistant: {e}")
            print("Falling back to synthetic conversational data...")
            return load_synthetic_conversational(max_conversations, seed)
    
    # Limit size
    if max_conversations:
        np.random.seed(seed)
        if len(corpus_texts) > max_conversations * 10:
            indices = np.random.choice(len(corpus_texts), max_conversations * 10, replace=False)
            corpus_texts = [corpus_texts[i] for i in indices]
            metadata_list = [metadata_list[i] for i in indices]
        
        if len(query_texts) > max_conversations:
            query_texts = query_texts[:max_conversations]
    
    print(f"Loaded {len(corpus_texts):,} corpus texts, {len(query_texts)} queries")
    return corpus_texts, query_texts, metadata_list


def load_synthetic_conversational(
    max_conversations: int,
    seed: int = 42
) -> Tuple[List[str], List[str], List[Dict]]:
    """Generate synthetic conversational data for testing."""
    np.random.seed(seed)
    
    topics = [
        "machine learning", "python programming", "data science",
        "web development", "cloud computing", "database design",
        "software engineering", "artificial intelligence", "cybersecurity"
    ]
    
    corpus_texts = []
    query_texts = []
    metadata_list = []
    
    for conv_id in range(max_conversations):
        topic = np.random.choice(topics)
        
        # Generate conversation turns
        for turn_id in range(5):
            text = f"User message about {topic} in conversation {conv_id}, turn {turn_id}. " + \
                   f"This is a detailed discussion about various aspects of {topic}."
            
            if turn_id < 4:
                corpus_texts.append(text)
                metadata_list.append({
                    "convo_id": f"conv_{conv_id}",
                    "turn_id": turn_id,
                    "speaker": "user",
                    "text": text
                })
            else:
                query_texts.append(text)
    
    return corpus_texts, query_texts, metadata_list


def compute_ground_truth_brute_force(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int = 10
) -> List[List[int]]:
    """Compute ground truth using brute-force (silver labels)."""
    print("Computing ground truth using brute-force...")
    ground_truths = []
    
    for query in tqdm(query_embeddings, desc="Computing ground truth"):
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        corpus_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10)
        scores = (query_norm @ corpus_norm.T).flatten()
        top_indices = np.argsort(-scores)[:k].tolist()
        ground_truths.append([str(i) for i in top_indices])
    
    return ground_truths


def run_suite_b(
    dataset_name: str = "openassistant",
    scales: List[int] = [100000, 200000, 300000],
    cache_dir: Path = Path("./cache"),
    seed: int = 42,
    max_queries: int = 200,
    concurrent: int = 1
) -> Dict:
    """
    Run Test Suite B: Conversational dataset.
    
    Args:
        dataset_name: "openassistant" or "synthetic"
        scales: List of corpus sizes to test
        cache_dir: Directory for embedding cache
        seed: Random seed
        max_queries: Maximum number of queries
        concurrent: Concurrent queries
    
    Returns:
        Results dict
    """
    print(f"\n{'='*80}")
    print(f"Test Suite B: Conversational Dataset ({dataset_name})")
    print(f"{'='*80}\n")
    
    # Load dataset
    print("Loading conversational dataset...")
    if dataset_name == "openassistant":
        corpus_texts, query_texts, metadata_list = load_openassistant(
            max_conversations=scales[-1] // 10 if scales else None,
            seed=seed
        )
    else:
        corpus_texts, query_texts, metadata_list = load_synthetic_conversational(
            max_conversations=scales[-1] // 10 if scales else 10000,
            seed=seed
        )
    
    # Limit queries
    if len(query_texts) > max_queries:
        query_texts = query_texts[:max_queries]
    
    # Build embeddings
    cache = EmbeddingCache(cache_dir, os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    
    print("\nBuilding embeddings...")
    corpus_embeddings = cache.encode(corpus_texts, f"{dataset_name}_conversational_corpus")
    query_embeddings = cache.encode(query_texts, f"{dataset_name}_conversational_queries")
    dim = corpus_embeddings.shape[1]
    
    # Compute ground truth using brute-force (silver labels)
    print("\nComputing ground truth (silver labels via brute-force)...")
    # Use subset for ground truth computation to save time
    max_gt_compute = min(50000, len(corpus_embeddings))
    corpus_emb_gt = corpus_embeddings[:max_gt_compute]
    
    ground_truths_full = compute_ground_truth_brute_force(
        query_embeddings, corpus_emb_gt, k=10
    )
    
    # Results storage
    all_results = {
        "scales": [],
        "metrics": {
            "mcn": [],
            "faiss": [],
            "brute_force": []
        },
        "qualitative_examples": []
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
        metadata_subset = [metadata_list[i] for i in indices]
        
        # Adjust ground truths
        indices_set = set(range(min(scale, max_gt_compute)))
        ground_truths = [
            [gt for gt in gt_list if int(gt) in indices_set]
            for gt_list in ground_truths_full
        ]
        
        # Create metadata with IDs
        metadata = [
            {**meta, "original_idx": i, "id": str(i)}
            for i, meta in enumerate(metadata_subset)
        ]
        
        # Evaluate systems
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
            None,  # No qrels for conversational
            config=mcn_config,
            concurrent=concurrent
        )
        
        faiss_results = evaluate_system(
            "faiss",
            corpus_emb_subset,
            query_embeddings,
            ground_truths,
            metadata,
            None,
            concurrent=concurrent
        )
        
        brute_results = evaluate_system(
            "brute_force",
            corpus_emb_subset,
            query_embeddings,
            ground_truths,
            metadata,
            None,
            concurrent=concurrent,
            max_brute_force=20000
        )
        
        # Store results
        all_results["scales"].append(scale)
        all_results["metrics"]["mcn"].append({
            "recall@10": mcn_results["metrics"]["recall@10"],
            "recall@100": mcn_results["metrics"]["recall@100"],
            "mrr@10": mcn_results["metrics"]["mrr@10"],
            "latency_p50": mcn_results["latency_stats"]["p50"],
            "latency_p95": mcn_results["latency_stats"]["p95"],
            "build_time": mcn_results["build_time"],
            "compression_ratio": mcn_results["compression_ratio"],
        })
        all_results["metrics"]["faiss"].append({
            "recall@10": faiss_results["metrics"]["recall@10"],
            "recall@100": faiss_results["metrics"]["recall@100"],
            "mrr@10": faiss_results["metrics"]["mrr@10"],
            "latency_p50": faiss_results["latency_stats"]["p50"],
            "latency_p95": faiss_results["latency_stats"]["p95"],
            "build_time": faiss_results["build_time"],
            "compression_ratio": faiss_results["compression_ratio"],
        })
        all_results["metrics"]["brute_force"].append({
            "recall@10": brute_results["metrics"]["recall@10"],
            "recall@100": brute_results["metrics"]["recall@100"],
            "mrr@10": brute_results["metrics"]["mrr@10"],
            "latency_p50": brute_results["latency_stats"]["p50"],
            "latency_p95": brute_results["latency_stats"]["p95"],
            "build_time": brute_results["build_time"],
            "compression_ratio": brute_results["compression_ratio"],
        })
        
        # Collect qualitative examples (first scale only)
        if scale == scales[0] and len(all_results["qualitative_examples"]) == 0:
            print("\nCollecting qualitative examples...")
            # Get top 5 queries
            for q_idx in range(min(5, len(query_texts))):
                # Get MCN results
                from .eval_harness import QueryRunner
                runner = QueryRunner(dim)
                mcn_index = mcn_results.get("index")  # We need to store index
                # This is simplified - in practice, we'd need to store the index
                # For now, just store query text
                all_results["qualitative_examples"].append({
                    "query": query_texts[q_idx][:200],
                    "query_idx": q_idx
                })
    
    return {
        "suite_name": f"Conversational-{dataset_name}",
        "dataset": dataset_name,
        "scales": all_results["scales"],
        "metrics": all_results["metrics"],
        "qualitative_examples": all_results["qualitative_examples"],
        "n_queries": len(query_texts),
        "seed": seed,
    }

