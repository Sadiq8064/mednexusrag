# rag.py

from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================================
# Device Setup
# ============================================================
device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

embedder = SentenceTransformer(
    "pritamdeka/S-BioBert-snli-multinli-stsb",
    device=device
)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=device
)

# ============================================================
# File Paths (POINT TO RAILWAY VOLUME)
# ============================================================
DATA_DIR = "/app/data"

PUB_INDEX_PATH = f"{DATA_DIR}/pubmed_index.faiss"
PUB_META_PATH = f"{DATA_DIR}/pubmed_meta.pkl"

MED_INDEX_PATH = f"{DATA_DIR}/medquad_index.faiss"
MED_META_PATH = f"{DATA_DIR}/medquad_meta.pkl"

# ============================================================
# Load FAISS + Metadata
# ============================================================
def load_index_and_meta(index_path: str, meta_path: str):
    print(f"Loading: {index_path} ...")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)

    texts = data["texts"]
    meta = data["meta"]

    print(f"Loaded {len(texts)} texts from {meta_path}")
    return index, texts, meta


pub_index, pub_texts, pub_meta = load_index_and_meta(PUB_INDEX_PATH, PUB_META_PATH)
med_index, med_texts, med_meta = load_index_and_meta(MED_INDEX_PATH, MED_META_PATH)

# ============================================================
# Reranking
# ============================================================
def rerank(query, candidate_idxs, texts):
    pairs = [(query, texts[i]) for i in candidate_idxs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidate_idxs, scores), key=lambda x: x[1], reverse=True)
    return ranked

# ============================================================
# Search Dataset
# ============================================================
def search_dataset(query, index, texts, meta, dataset_name, top_k=5):

    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    sims, idxs = index.search(q_emb, top_k)
    idxs = idxs[0]

    reranked = rerank(query, idxs, texts)
    best_idx, rerank_score = reranked[0]

    ans = meta[best_idx].get("long_answer", "").strip()
    fd = meta[best_idx].get("final_decision", "").strip()

    return {
        "source": dataset_name,
        "answer": ans,
        "final_decision": fd,
        "rerank_score": float(rerank_score)
    }

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="Medical RAG Vector Search API")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_api(req: Question):

    q = req.question

    pub = search_dataset(q, pub_index, pub_texts, pub_meta, "PubMedQA")
    med = search_dataset(q, med_index, med_texts, med_meta, "MedQuAD")

    best = max([pub, med], key=lambda x: x["rerank_score"])

    return {
        "question": q,
        "source": best["source"],
        "answer": best["answer"],
        "final_decision": best["final_decision"],
        "rerank_score": best["rerank_score"]
    }
