# rag.py
import os
import sys
import pickle
import numpy as np
import faiss
import torch
import gdown

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================
# Environment + Paths
# ==========================
DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

PUBMED_INDEX_URL = os.getenv("PUBMED_INDEX_URL")
PUBMED_META_URL  = os.getenv("PUBMED_META_URL")
MED_INDEX_URL    = os.getenv("MED_INDEX_URL")
MED_META_URL     = os.getenv("MED_META_URL")
API_KEY = os.getenv("API_KEY")

PUB_INDEX_PATH = os.path.join(DATA_DIR, "pubmed_index.faiss")
PUB_META_PATH  = os.path.join(DATA_DIR, "pubmed_meta.pkl")
MED_INDEX_PATH = os.path.join(DATA_DIR, "medquad_index.faiss")
MED_META_PATH  = os.path.join(DATA_DIR, "medquad_meta.pkl")

# ==========================
# Google Drive downloader
# ==========================
def download_file(url, dest_path):
    """Download from GDrive ID-based URL."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[skip] already exists: {dest_path}")
        return

    if not url:
        raise RuntimeError(f"Missing URL for {dest_path}")

    print(f"[gdown] downloading: {url}")

    # Extract Google Drive file ID from URL
    try:
        file_id = url.strip().split("/d/")[1].split("/")[0]
    except:
        raise RuntimeError(f"Invalid Google Drive URL: {url}")

    try:
        gdown.download(id=file_id, output=dest_path, quiet=False)
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise RuntimeError(f"gdown failed: {e}")


# ==========================
# Download dataset files once
# ==========================
try:
    download_file(PUBMED_INDEX_URL, PUB_INDEX_PATH)
    download_file(PUBMED_META_URL,  PUB_META_PATH)
    download_file(MED_INDEX_URL,    MED_INDEX_PATH)
    download_file(MED_META_URL,     MED_META_PATH)
except Exception as e:
    print(f"[startup error] {e}", file=sys.stderr)
    raise

# ==========================
# Device & Models
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

embedder = SentenceTransformer(
    "pritamdeka/S-BioBert-snli-multinli-stsb",
    device=device
)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=device
)

# ==========================
# Load FAISS index + metadata
# ==========================
def load_index_and_meta(index_path, meta_path, name):
    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        data = pickle.load(f)

    texts, meta = data["texts"], data["meta"]
    print(f"Loaded {len(texts)} entries for {name}")

    return index, texts, meta


pub_index, pub_texts, pub_meta = load_index_and_meta(PUB_INDEX_PATH, PUB_META_PATH, "PubMedQA")
med_index, med_texts, med_meta = load_index_and_meta(MED_INDEX_PATH, MED_META_PATH, "MedQuAD")


# ==========================
# Improved RAG Search
# ==========================
def improved_rag_search(query, top_k_faiss=30, top_k_rerank=5):
    """Higher-quality RAG: large FAISS search + cross-encoder rerank + answer fusion."""

    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # ---------- FAISS SEARCH ----------
    def faiss_search(index, texts, meta):
        sims, idxs = index.search(q_emb, top_k_faiss)
        idxs = idxs[0]
        return [(i, texts[i], meta[i]) for i in idxs]

    pub_candidates = faiss_search(pub_index, pub_texts, pub_meta)
    med_candidates = faiss_search(med_index, med_texts, med_meta)

    all_candidates = pub_candidates + med_candidates

    # ---------- CROSS ENCODER RERANK ----------
    pairs = [(query, text) for (_, text, _) in all_candidates]
    scores = cross_encoder.predict(pairs)

    results = []
    for (idx, text, meta), score in zip(all_candidates, scores):
        results.append({
            "text": text,
            "score": float(score),
            "long_answer": meta.get("long_answer", "").strip(),
            "final_decision": meta.get("final_decision", "").strip(),
            "source": "PubMedQA" if meta in pub_meta else "MedQuAD"
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top_results = results[:top_k_rerank]

    # ---------- ANSWER FUSION ----------
    fused_answer = " ".join(
        [r["long_answer"] for r in top_results if len(r["long_answer"]) > 5]
    ).strip()

    if not fused_answer:
        fused_answer = top_results[0]["text"]  # fallback

    # Safety check
    if top_results[0]["score"] < 0.2:
        fused_answer = "No confident medical answer found in the dataset."

    return {
        "question": query,
        "best_score": top_results[0]["score"],
        "source": top_results[0]["source"],
        "answer": fused_answer,
        "top_passages_used": len(top_results)
    }


# ==========================
# FastAPI App
# ==========================
app = FastAPI(title="MedNexus Medical RAG (Improved)")

def check_api_key(request: Request):
    if API_KEY:
        if request.headers.get("x-api-key") != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")


# ---- Health route ----
@app.get("/health")
def health():
    return {"status": "ok", "device": device}


# ---- GET route (browser testing) ----
@app.get("/ask")
def ask_get(q: str, request: Request):
    """Allows testing RAG directly in browser via GET /ask?q=your-question"""
    check_api_key(request)
    return improved_rag_search(q.strip())


# ---- POST route (API usage) ----
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_post(payload: Query, request: Request):
    check_api_key(request)
    return improved_rag_search(payload.question.strip())
