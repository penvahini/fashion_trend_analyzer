# src/cluster_registry.py
"""
Persistent lineage registry for incremental/online clustering.

Replaces the old approach (independently re-run KMeans every run, then
post-hoc guess which cluster in run N matches which cluster in run N-1 by
centroid similarity) with the more correct approach: cluster identity is
stable *by construction*. A new run's images are matched against existing
lineage centroids first; only images that don't match anything become
candidates for a brand-new lineage. This means trend_tracker.py no longer
needs to guess at matches across runs -- a lineage_id means the same trend
in every run it appears in.

Centroids live in raw CLIP embedding space (512-dim, unit-normalized
ViT-B/32 vectors) since that space is consistent across all runs --
PCA-reduced embeddings are NOT comparable across runs (PCA is refit per run).
"""
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "cluster_registry.json"

MATCH_THRESHOLD = 0.75  # min cosine similarity to assign an image to an existing lineage

def load_registry() -> list:
    if not REGISTRY_PATH.exists():
        return []
    raw = json.loads(REGISTRY_PATH.read_text())
    for entry in raw:
        entry["centroid"] = np.array(entry["centroid"], dtype="float32")
    return raw

def save_registry(registry: list):
    serializable = [
        {**entry, "centroid": entry["centroid"].tolist()} for entry in registry
    ]
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(serializable, indent=2))

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of A and every row of B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T

def assign_to_existing_lineages(Z: np.ndarray, registry: list, threshold: float = MATCH_THRESHOLD):
    """
    Matches each row of Z against the closest registry centroid.
    Returns (assigned_lineage_ids, unmatched_mask) where assigned_lineage_ids[i]
    is the matched lineage_id (or None) for image i.
    """
    n = Z.shape[0]
    assigned = [None] * n
    unmatched_mask = np.ones(n, dtype=bool)

    if not registry:
        return assigned, unmatched_mask

    centroids = np.stack([entry["centroid"] for entry in registry])
    sims = cosine_sim_matrix(Z, centroids)  # (n_images, n_lineages)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    for i in range(n):
        if best_sim[i] >= threshold:
            assigned[i] = registry[best_idx[i]]["lineage_id"]
            unmatched_mask[i] = False

    return assigned, unmatched_mask

def update_matched_centroids(registry: list, Z: np.ndarray, assigned_lineage_ids: list, run_id: str):
    """Running weighted mean update for every lineage that got new matches this run."""
    by_id = {entry["lineage_id"]: entry for entry in registry}
    sums, counts = {}, {}
    for vec, lineage_id in zip(Z, assigned_lineage_ids):
        if lineage_id is None:
            continue
        sums[lineage_id] = sums.get(lineage_id, 0) + vec
        counts[lineage_id] = counts.get(lineage_id, 0) + 1

    for lineage_id, count in counts.items():
        entry = by_id[lineage_id]
        old_n, old_centroid = entry["n_images"], entry["centroid"]
        entry["centroid"] = (old_centroid * old_n + sums[lineage_id]) / (old_n + count)
        entry["n_images"] = old_n + count
        entry["last_seen_run"] = run_id

def add_new_lineages(registry: list, new_centroids: np.ndarray, new_counts: list, run_id: str) -> list:
    """Appends brand-new lineages to the registry, returns their assigned lineage_ids."""
    next_id = max([e["lineage_id"] for e in registry], default=-1) + 1
    new_ids = []
    for centroid, count in zip(new_centroids, new_counts):
        lineage_id = next_id
        registry.append({
            "lineage_id": lineage_id,
            "centroid": centroid,
            "n_images": int(count),
            "first_seen_run": run_id,
            "last_seen_run": run_id,
        })
        new_ids.append(lineage_id)
        next_id += 1
    return new_ids
