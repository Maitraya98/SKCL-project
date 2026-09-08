"""
build_graph.py

Implements Sections 3.3 (Semantic Knowledge Graph) and 3.4 (Multi-granularity
Knowledge Transfer, Top-K selection) of the paper.

- Embeds each class description with all-MiniLM-L6-v2 (the exact model
  named in the paper), producing 384-dim vectors.
- L2-normalizes embeddings (Eq. 1) and computes the cosine similarity
  matrix S (Eq. 2).
- For each node i, selects the Top-K most similar OTHER nodes (Eq. 3),
  across the combined fine+coarse graph (both levels are nodes in the
  same graph, per Fig. 5/6 -- this is what enables cross-granularity
  transfer, e.g. a fine class linking to a coarse node or a fine class
  in a different coarse group).

Usage:
    python3 build_graph.py --descriptions descriptions_cifar100.json \
        --dataset cifar100 --data_root $WORK/data --top_k 2 \
        --out graph_cifar100.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "datasets"))
from cifar_hierarchy import get_hierarchy  # noqa: E402


def build_similarity_matrix(descriptions: dict, node_order: list, model_name: str = "all-MiniLM-L6-v2"):
    """
    Returns (embeddings, S) where embeddings is (n, 384) L2-normalized,
    and S is the (n, n) cosine similarity matrix (Eq. 1-2).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [descriptions[name] for name in node_order]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # Eq. 1: L2-normalize so all embeddings lie on the unit sphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / norms

    # Eq. 2: cosine similarity == dot product of L2-normalized vectors
    S = embeddings_normed @ embeddings_normed.T
    return embeddings_normed, S


def top_k_neighbors(S: np.ndarray, k: int) -> list:
    """
    Eq. 3: for each row i, the Top-K most similar OTHER nodes (j != i).
    Returns a list of length n, each entry a list of k neighbor indices.
    """
    n = S.shape[0]
    neighbors = []
    for i in range(n):
        row = S[i].copy()
        row[i] = -np.inf  # exclude self
        top_k_idx = np.argsort(-row)[:k]
        neighbors.append(top_k_idx.tolist())
    return neighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptions", required=True, help="JSON file from generate_descriptions.py")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--top_k", type=int, default=2, help="K in the paper's Top-K selection (K=2 for CIFAR)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    with open(args.descriptions) as f:
        descriptions = json.load(f)

    fine_names, coarse_names, fine_to_coarse = get_hierarchy(args.dataset, args.data_root)
    # Node order: fine classes first (indices 0..n_fine-1), then coarse (n_fine..n_fine+n_coarse-1)
    node_order = list(fine_names) + list(coarse_names)
    n_fine = len(fine_names)

    missing = [c for c in node_order if c not in descriptions]
    if missing:
        raise ValueError(f"Missing descriptions for: {missing}")

    embeddings, S = build_similarity_matrix(descriptions, node_order, args.sbert_model)
    neighbors = top_k_neighbors(S, args.top_k)

    print(f"Built {S.shape[0]}x{S.shape[0]} similarity matrix ({n_fine} fine + {len(coarse_names)} coarse nodes)")
    print(f"Example: {node_order[0]!r}'s top-{args.top_k} neighbors: "
          f"{[node_order[j] for j in neighbors[0]]} "
          f"(sims: {[round(float(S[0, j]), 3) for j in neighbors[0]]})")

    out = {
        "node_order": node_order,
        "n_fine": n_fine,
        "n_coarse": len(coarse_names),
        "fine_to_coarse": fine_to_coarse,
        "similarity_matrix": S.tolist(),
        "top_k": args.top_k,
        "neighbors": neighbors,  # neighbors[i] = list of top_k neighbor indices for node i
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Saved graph to {args.out}")


if __name__ == "__main__":
    main()
