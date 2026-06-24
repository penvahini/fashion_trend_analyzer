# src/rag/retriever.py
"""
Loads the FAISS index built by build_index.py and does semantic search
over cluster keywords/summaries for a natural-language query.
"""
from pathlib import Path
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "rag_index"
EMBED_MODEL = "text-embedding-3-small"

load_dotenv(PROJECT_ROOT / ".env")

class TrendRetriever:
    def __init__(self):
        index_path = INDEX_DIR / "index.faiss"
        metadata_path = INDEX_DIR / "metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No RAG index found at {INDEX_DIR}. Run `python src/rag/build_index.py` first."
            )
        self.index = faiss.read_index(str(index_path))
        self.metadata = json.loads(metadata_path.read_text())
        self.client = OpenAI()

    def search(self, query: str, top_k: int = 5):
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[query])
        vec = np.array([resp.data[0].embedding], dtype="float32")
        faiss.normalize_L2(vec)

        scores, idxs = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            doc = dict(self.metadata[idx])
            doc["score"] = round(float(score), 4)
            results.append(doc)
        return results

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "minimalist neutral tones"
    retriever = TrendRetriever()
    for r in retriever.search(query):
        print(f"[{r['score']}] run={r['run_id']} cluster={r['cluster_id']} size={r['size']} keywords={r['keywords']}")
