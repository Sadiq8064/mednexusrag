# rag.py
import os
import sys
import pickle
import numpy as np
import faiss
import torch
import gdown
import re

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================================
# Environment + Paths
# ============================================================
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

# ============================================================
# Google Drive downloader
# ============================================================
def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[skip] already exists: {dest_path}")
        return

    if not url:
        raise RuntimeError(f"Missing URL for: {dest_path}")

    print(f"[gdown] downloading GDrive file: {url}")

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

# ============================================================
# Download data files once
# ============================================================
try:
    download_file(PUBMED_INDEX_URL, PUB_INDEX_PATH)
    download_file(PUBMED_META_URL,  PUB_META_PATH)
    download_file(MED_INDEX_URL,    MED_INDEX_PATH)
    download_file(MED_META_URL,     MED_META_PATH)
except Exception as e:
    print(f"[startup error] {e}", file=sys.stderr)
    raise

# ============================================================
# Device + Models (Safe Torch Version)
# ============================================================
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

# ============================================================
# Load FAISS + Metadata
# ============================================================
def load_index_and_meta(index_path, meta_path, name):
    print(f"Loading index: {index_path}")
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        data = pickle.load(f)

    texts, meta = data["texts"], data["meta"]
    print(f"{name}: Loaded {len(texts)} entries")
    return index, texts, meta

pub_index, pub_texts, pub_meta = load_index_and_meta(PUB_INDEX_PATH, PUB_META_PATH, "PubMedQA")
med_index, med_texts, med_meta = load_index_and_meta(MED_INDEX_PATH, MED_META_PATH, "MedQuAD")

# ============================================================
# RAG v2 Utilities
# ============================================================
def softmax(xs):
    e = np.exp(np.array(xs) - np.max(xs))
    return (e / e.sum()).tolist()

def sentence_tokenize(text):
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def dedupe_sentences(sentences, threshold=0.88):
    if not sentences:
        return []
    s_embs = embedder.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    keep = []
    used = np.zeros(len(sentences), dtype=bool)

    for i, emb in enumerate(s_embs):
        if used[i]:
            continue
        keep.append(sentences[i])
        sims = np.dot(s_embs, emb)
        used = np.logical_or(used, sims >= threshold)

    return keep

# ============================================================
# Improved RAG v2 Pipeline
# ============================================================
def improved_rag_search_v2(query, top_k_faiss=50, top_k_rerank=12, fuse_sentences=7):
    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # ---- FAISS Search ----
    def faiss_search(index, texts, meta):
        sims, idxs = index.search(q_emb, top_k_faiss)
        idxs = idxs[0]
        return [(i, texts[i], meta[i]) for i in idxs]

    pub_candidates = faiss_search(pub_index, pub_texts, pub_meta)
    med_candidates = faiss_search(med_index, med_texts, med_meta)
    all_candidates = pub_candidates + med_candidates

    # ---- Cross-Encoder Rerank ----
    ce_pairs = [(query, text) for (_, text, _) in all_candidates]
    ce_scores = cross_encoder.predict(ce_pairs)

    # ---- Cosine Scores ----
    texts = [t for (_, t, _) in all_candidates]
    p_embs = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    cos_sims = (np.dot(p_embs, q_emb[0]) + 1.0) / 2.0

    # ---- Combine CE + Cosine ----
    ce_arr = np.array(ce_scores)
    if ce_arr.max() != ce_arr.min():
        ce_norm = (ce_arr - ce_arr.min()) / (ce_arr.max() - ce_arr.min())
    else:
        ce_norm = np.ones_like(ce_arr)

    w_ce = 0.7
    w_cos = 0.3
    combined = (w_ce * ce_norm) + (w_cos * np.array(cos_sims))

    # ---- Select Top rerank ----
    order = np.argsort(-combined)[:top_k_rerank]
    selected = []
    for idx in order:
        i, text, meta = all_candidates[idx]
        selected.append({
            "text": text,
            "long_answer": meta.get("long_answer", "").strip(),
            "final_decision": meta.get("final_decision", "").strip(),
            "source": "PubMedQA" if meta in pub_meta else "MedQuAD",
            "ce_score": float(ce_scores[idx]),
            "cos_sim": float(cos_sims[idx]),
            "combined_score": float(combined[idx])
        })

    # ---- Build Sentences to Fuse ----
    all_sentences = []
    for r in selected:
        raw = r["long_answer"] or r["text"]
        sents = sentence_tokenize(raw)
        all_sentences.extend(sents)

    all_sentences = all_sentences[: fuse_sentences * 4]
    deduped = dedupe_sentences(all_sentences)
    fused = " ".join(deduped[:fuse_sentences]).strip()

    if not fused:
        fused = selected[0]["long_answer"] or selected[0]["text"]

    # ---- Confidence ----
    conf = softmax([r["combined_score"] for r in selected])[0]

    if conf < 0.08:
        fused = "No confident medical answer found in the dataset."

    return {
        "question": query,
        "answer": fused,
        "confidence": conf,
        "source": selected[0]["source"],
        "top_passages": selected[:5]
    }

# ============================================================
# FastAPI Server
# ============================================================
app = FastAPI(title="MedNexus Medical RAG v2")

class Query(BaseModel):
    question: str

def check_api_key(request: Request):
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.get("/ask")
def ask_get(q: str, request: Request):
    check_api_key(request)
    return improved_rag_search_v2(q.strip())

@app.post("/ask")
def ask_post(body: Query, request: Request):
    check_api_key(request)
    return improved_rag_search_v2(body.question.strip())
