# rag.py
import os
import sys
import gdown
import pickle
import faiss
import numpy as np
import torch
from fastapi import FastAPI, Request, HTTPException
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

PUBMED_INDEX_URL = os.getenv("PUBMED_INDEX_URL")
PUBMED_META_URL  = os.getenv("PUBMED_META_URL")
MED_INDEX_URL    = os.getenv("MED_INDEX_URL")
MED_META_URL     = os.getenv("MED_META_URL")

API_KEY = os.getenv("API_KEY")   # optional
CE_THRESHOLD = float(os.getenv("CE_THRESHOLD", "5.0"))  # NEW ★ confidence threshold

PUB_INDEX_PATH = os.path.join(DATA_DIR, "pubmed_index.faiss")
PUB_META_PATH  = os.path.join(DATA_DIR, "pubmed_meta.pkl")
MED_INDEX_PATH = os.path.join(DATA_DIR, "medquad_index.faiss")
MED_META_PATH  = os.path.join(DATA_DIR, "medquad_meta.pkl")

# ============================================================
# Helper: download Google Drive file once
# ============================================================
def download_from_drive(drive_url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] already exists: {dest}")
        return

    if not drive_url:
        raise RuntimeError(f"Missing GDrive URL for {dest}")

    print(f"[gdown] downloading: {drive_url}")

    try:
        file_id = drive_url.split("/d/")[1].split("/")[0]
    except:
        raise RuntimeError(f"❌ Invalid Google Drive URL: {drive_url}")

    gdown.download(id=file_id, output=dest, quiet=False)

# ============================================================
# Download files on startup
# ============================================================
try:
    download_from_drive(PUBMED_INDEX_URL, PUB_INDEX_PATH)
    download_from_drive(PUBMED_META_URL,  PUB_META_PATH)
    download_from_drive(MED_INDEX_URL,    MED_INDEX_PATH)
    download_from_drive(MED_META_URL,     MED_META_PATH)
except Exception as e:
    print(f"[startup error] {e}", file=sys.stderr)
    raise

# ============================================================
# Model setup
# ============================================================
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

embedder = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

# ============================================================
# Load FAISS + metadata
# ============================================================
def load_index_and_meta(index_path, meta_path, name):
    print(f"Loading {name} ...")
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        data = pickle.load(f)

    texts = data["texts"]
    meta = data["meta"]

    print(f"Loaded {len(texts)} entries for {name}")
    return index, texts, meta

pub_index, pub_texts, pub_meta = load_index_and_meta(PUB_INDEX_PATH, PUB_META_PATH, "PubMedQA")
med_index, med_texts, med_meta = load_index_and_meta(MED_INDEX_PATH, MED_META_PATH, "MedQuAD")

# ============================================================
# Core RAG: return ONE best answer with thresholding
# ============================================================
def get_best_answer(query, top_k=40):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    # ---- Search PubMed ----
    sims_pub, idxs_pub = pub_index.search(q_emb, top_k)
    pub_hits = [(sims_pub[0][i], idxs_pub[0][i], pub_texts[idxs_pub[0][i]], pub_meta[idxs_pub[0][i]], "PubMedQA")
                for i in range(top_k)]

    # ---- Search MedQuAD ----
    sims_med, idxs_med = med_index.search(q_emb, top_k)
    med_hits = [(sims_med[0][i], idxs_med[0][i], med_texts[idxs_med[0][i]], med_meta[idxs_med[0][i]], "MedQuAD")
                for i in range(top_k)]

    # Combine all
    all_hits = pub_hits + med_hits
    if not all_hits:
        return "Not enough evidence to answer confidently.", None

    # ---- Rank using Cross-Encoder ----
    ce_inputs = [(query, h[2]) for h in all_hits]
    ce_scores = cross_encoder.predict(ce_inputs)

    # Best index
    best_idx = int(np.argmax(ce_scores))
    best_ce = float(ce_scores[best_idx])
    _, _, _, best_meta, source = all_hits[best_idx]

    # --- Threshold check ---
    if best_ce < CE_THRESHOLD:
        return "Not enough evidence to answer confidently.", None

    # Final answer
    answer = best_meta.get("long_answer", "").strip()
    if not answer:
        answer = best_meta.get("answer", "").strip()
    if not answer:
        answer = "Not enough evidence to answer confidently."

    return answer, source

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="Simple Medical RAG with Threshold")

def verify_api(request: Request):
    if API_KEY:
        if request.headers.get("x-api-key") != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.get("/health")
def health():
    return {"status": "ok", "threshold": CE_THRESHOLD}

@app.get("/ask")
def ask(question: str, request: Request):
    verify_api(request)

    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer, source = get_best_answer(question)

    return {
        "question": question,
        "answer": answer,
        "source": source
    }
