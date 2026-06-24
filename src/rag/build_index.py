# src/rag/build_index.py
"""
Builds a FAISS index over every cluster's keywords + LLM summary across all
local runs, so the agent/MCP layer can do semantic search over trend data
("what's trending in outerwear this spring") instead of only exact
cluster-id lookups.
"""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
INDEX_DIR = PROJECT_ROOT / "data" / "rag_index"
EMBED_MODEL = "text-embedding-3-small"

load_dotenv(PROJECT_ROOT / ".env")

def build_documents():
    docs = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        labels_path = run_dir / "cluster_labels.json"
        clusters_path = run_dir / "clusters.csv"
        if not labels_path.exists() or not clusters_path.exists():
            continue

        sizes = pd.read_csv(clusters_path)["cluster"].value_counts().to_dict()
        labels = json.loads(labels_path.read_text())
        for entry in labels:
            cluster_id = int(entry["cluster"])
            keywords = entry.get("keywords", [])
            summary = entry.get("summary", "")
            text = f"Keywords: {', '.join(keywords)}. Summary: {summary}"
            docs.append({
                "run_id": run_id,
                "cluster_id": cluster_id,
                "size": int(sizes.get(cluster_id, 0)),
                "keywords": keywords,
                "summary": summary,
                "text": text,
            })
    return docs

def embed_texts(client, texts, batch_size=64):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype="float32")

def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    docs = build_documents()
    if not docs:
        raise RuntimeError(f"No labeled clusters found under {RUNS_DIR}. Run label_clusters.py first.")

    client = OpenAI()
    print(f"[rag] embedding {len(docs)} cluster documents with {EMBED_MODEL}")
    vectors = embed_texts(client, [d["text"] for d in docs])
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine sim via normalized inner product
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    (INDEX_DIR / "metadata.json").write_text(json.dumps(docs, indent=2))
    print(f"[rag] wrote index + metadata to {INDEX_DIR} ({len(docs)} documents, dim={vectors.shape[1]})")

if __name__ == "__main__":
    main()
