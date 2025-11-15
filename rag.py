# rag.py
import os
import sys
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# ====================
# Config & env
# ====================
DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

# Google Drive direct-download URLs must be provided as Railway variables
PUBMED_INDEX_URL = os.getenv("PUBMED_INDEX_URL")
PUBMED_META_URL  = os.getenv("PUBMED_META_URL")
MED_INDEX_URL    = os.getenv("MED_INDEX_URL")
MED_META_URL     = os.getenv("MED_META_URL")

# Optional API key protection
API_KEY = os.getenv("API_KEY")  # set this in Railway variables if you want protection

# Local file paths (in persistent volume)
PUB_INDEX_PATH = os.path.join(DATA_DIR, "pubmed_index.faiss")
PUB_META_PATH  = os.path.join(DATA_DIR, "pubmed_meta.pkl")
MED_INDEX_PATH = os.path.join(DATA_DIR, "medquad_index.faiss")
MED_META_PATH  = os.path.join(DATA_DIR, "medquad_meta.pkl")

# ====================
# Helper: download once
# ====================
def download_file(url: str, dest_path: str, chunk_size: int = 32768):
    if not url:
        raise ValueError(f"Missing URL for {dest_path}. Set the appropriate environment variable.")
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[skip] already exists: {dest_path}")
        return
    print(f"[download] from {url} -> {dest_path}")
    # stream download
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        print(f"[done] {dest_path}")
    except Exception as e:
        # remove partial file if exists
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        raise RuntimeError(f"Failed to download {url}: {e}")

# ====================
# Download files if missing (only runs at startup)
# ====================
try:
    download_file(PUBMED_INDEX_URL, PUB_INDEX_PATH)
    download_file(PUBMED_META_URL,  PUB_META_PATH)
    download_file(MED_INDEX_URL,    MED_INDEX_PATH)
    download_file(MED_META_URL,     MED_META_PATH)
except Exception as e:
    # Fail fast - logs will show error in Railway
    print(f"[startup error] {e}", file=sys.stderr)
    raise

# ====================
# Device & models
# ====================
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embedder = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

# ====================
# Load FAISS + metadata
# ====================
def load_index_and_meta(index_path, meta_path, name):
    print(f"Loading: {index_path} ...")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    texts, meta = data["texts"], data["meta"]
    print(f"Loaded {len(texts)} texts from {meta_path} for {name}")
    return index, texts, meta

pub_index, pub_texts, pub_meta = load_index_and_meta(PUB_INDEX_PATH, PUB_META_PATH, "PubMedQA")
med_index, med_texts, med_meta = load_index_and_meta(MED_INDEX_PATH, MED_META_PATH, "MedQuAD")

# ====================
# Rerank & search
# ====================
def rerank_results(query, candidate_idxs, texts):
    pairs = [(query, texts[i]) for i in candidate_idxs]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(candidate_idxs, scores), key=lambda x: x[1], reverse=True)
    return reranked

def search_dataset(query, index, texts, meta, dataset_name, top_k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    sims, idxs = index.search(q_emb, top_k)
    idxs = idxs[0]

    reranked = rerank_results(query, idxs, texts)
    best_idx, rerank_score = reranked[0]

    best_meta = meta[best_idx]
    long_answer = best_meta.get("long_answer", "").strip()
    final_decision = best_meta.get("final_decision", "").strip()

    return {
        "dataset": dataset_name,
        "rerank_score": float(rerank_score),
        "long_answer": long_answer,
        "final_decision": final_decision
    }

# ====================
# FastAPI app
# ====================
app = FastAPI(title="Medical Vector Search RAG (FAISS only)")

class QueryRequest(BaseModel):
    question: str

def check_api_key(request: Request):
    if API_KEY:
        key = request.headers.get("x-api-key")
        if not key or key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/ask")
def ask(req: QueryRequest, request: Request):
    # API key protection (optional)
    check_api_key(request)

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    pub_result = search_dataset(q, pub_index, pub_texts, pub_meta, "PubMedQA")
    med_result = search_dataset(q, med_index, med_texts, med_meta, "MedQuAD")

    best = max([pub_result, med_result], key=lambda x: x["rerank_score"])
    answer = best["long_answer"] or "No answer found in dataset."

    return {
        "question": q,
        "answer": answer,
        "source": best["dataset"],
        "rerank_score": best["rerank_score"],
        "final_decision": best["final_decision"]
    }
