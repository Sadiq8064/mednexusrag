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
import google.generativeai as genai

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

PUBMED_INDEX_URL = os.getenv("PUBMED_INDEX_URL")
PUBMED_META_URL  = os.getenv("PUBMED_META_URL")
MED_INDEX_URL    = os.getenv("MED_INDEX_URL")
MED_META_URL     = os.getenv("MED_META_URL")

API_KEY = os.getenv("API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
CE_THRESHOLD = float(os.getenv("CE_THRESHOLD", "5.0"))

genai.configure(api_key=GEMINI_KEY)

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

PUB_INDEX_PATH = f"{DATA_DIR}/pubmed_index.faiss"
PUB_META_PATH  = f"{DATA_DIR}/pubmed_meta.pkl"
MED_INDEX_PATH = f"{DATA_DIR}/medquad_index.faiss"
MED_META_PATH  = f"{DATA_DIR}/medquad_meta.pkl"

# ============================================================
# Helper: download GDrive
# ============================================================
def download_from_drive(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest}")
        return
    file_id = url.split("/d/")[1].split("/")[0]
    gdown.download(id=file_id, output=dest, quiet=False)

for url, path in [
    (PUBMED_INDEX_URL, PUB_INDEX_PATH),
    (PUBMED_META_URL,  PUB_META_PATH),
    (MED_INDEX_URL,    MED_INDEX_PATH),
    (MED_META_URL,     MED_META_PATH),
]:
    try:
        download_from_drive(url, path)
    except Exception as e:
        print(f"[startup error] {e}")
        raise

# ============================================================
# Model setup
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

# ============================================================
# Load FAISS + meta
# ============================================================
def load_index_meta(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["meta"]

pub_index, pub_texts, pub_meta = load_index_meta(PUB_INDEX_PATH, PUB_META_PATH)
med_index, med_texts, med_meta = load_index_meta(MED_INDEX_PATH, MED_META_PATH)

# ============================================================
# Core RAG
# ============================================================
def rag_search(query, top_k=40):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    # PubMed
    sims_pub, idx_pub = pub_index.search(q_emb, top_k)
    pub_hits = [(idx_pub[0][i], sims_pub[0][i], pub_texts[idx_pub[0][i]], pub_meta[idx_pub[0][i]], "PubMedQA")
                for i in range(top_k)]

    # MedQuAD
    sims_med, idx_med = med_index.search(q_emb, top_k)
    med_hits = [(idx_med[0][i], sims_med[0][i], med_texts[idx_med[0][i]], med_meta[idx_med[0][i]], "MedQuAD")
                for i in range(top_k)]

    all_hits = pub_hits + med_hits

    ce_inputs = [(query, h[2]) for h in all_hits]
    ce_scores = cross_encoder.predict(ce_inputs)

    best_idx = int(np.argmax(ce_scores))
    best_score = float(ce_scores[best_idx])
    _, _, _, best_meta, source = all_hits[best_idx]

    if best_score < CE_THRESHOLD:
        return None, None  # weak → use Gemini

    answer = best_meta.get("long_answer") or best_meta.get("answer") or ""
    return answer.strip(), source


# ============================================================
# Gemini unified call
# ============================================================
def gemini_medical_classifier_and_context(question):
    prompt = f"""
You are a strict medical classifier + context generator.

1. First determine if the user question is truly MEDICAL. 
2. If it is NOT medical:
    - Respond ONLY with JSON:
      {{"is_medical": false}}
3. If it IS medical:
    - Generate ONE PARAGRAPH (120–200 words) written like a medical textbook.
    - The paragraph MUST contain the correct answer naturally inside it.
    - It should not look like AI content.
    - Do NOT give bullet points.
    - Do NOT say "As an AI".
    - Respond ONLY as JSON in this format:

      {{
        "is_medical": true,
        "context": "<paragraph>"
      }}

User question: "{question}"
"""

    response = gemini_model.generate_content(prompt).text
    return response


# ============================================================
# FastAPI
# ============================================================
app = FastAPI()

def verify_key(request: Request):
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(401, "Invalid API key")

@app.get("/ask")
def ask(question: str, request: Request):
    verify_key(request)

    question = question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    # -------------------------------
    # 1. Try RAG (high confidence)
    # -------------------------------
    rag_answer, rag_source = rag_search(question)

    if rag_answer:  # RAG succeeded
        return {
            "question": question,
            "answer": rag_answer,
            "source": rag_source,
            "is_medical": True
        }

    # -------------------------------
    # 2. RAG weak → use Gemini (single call)
    # -------------------------------
    gem = gemini_medical_classifier_and_context(question)

    try:
        gem_json = eval(gem)  # Gemini returns JSON text
    except:
        return {
            "question": question,
            "answer": "Failed to parse Gemini output.",
            "source": None,
            "is_medical": False
        }

    if gem_json.get("is_medical") is False:
        return {
            "question": question,
            "answer": "This question is not medical.",
            "source": None,
            "is_medical": False
        }

    # Medical → return context
    return {
        "question": question,
        "answer": gem_json.get("context", ""),
        "source": "MedBooks",
        "is_medical": True
    }
