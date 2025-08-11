# RAG demo — starter (FAISS + OpenAI; replace with your provider config)
import os
from typing import List, Tuple
import faiss
import numpy as np

def embed(texts: List[str]) -> np.ndarray:
    # TODO: replace with real embedding model (e.g., OpenAI, Hugging Face)
    # For starter/demo: random vectors with fixed seed (deterministic)
    rng = np.random.default_rng(0)
    return rng.normal(size=(len(texts), 384)).astype("float32")

def build_index(chunks: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    vecs = embed(chunks)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, vecs

def search(index, query: str, chunks: List[str], k=3):
    q = embed([query])
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]

if __name__ == "__main__":
    docs = [
        "Transformers use self-attention to mix token information.",
        "RAG augments generation by retrieving relevant passages.",
        "LoRA is a parameter-efficient finetuning method."
    ]
    idx, _ = build_index(docs)
    print(search(idx, "How does RAG work?", docs, k=2))
