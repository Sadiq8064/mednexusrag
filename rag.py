# rag.py
import os
import sys
import json
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
# Google Drive downloader
# ============================================================
def download_from_drive(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest}")
        return

    try:
        file_id = url.split("/d/")[1].split("/")[0]
    except Exception:
        raise RuntimeError(f"Invalid Google Drive URL: {url}")

    gdown.download(id=file_id, output=dest, quiet=False)


for url, path in [
    (PUBMED_INDEX_URL, PUB_INDEX_PATH),
    (PUBMED_META_URL,  PUB_META_PATH),
    (MED_INDEX_URL,    MED_INDEX_PATH),
    (MED_META_URL,     MED_META_PATH),
]:
    download_from_drive(url, path)


# ============================================================
# Models
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer(
    "pritamdeka/S-BioBert-snli-multinli-stsb",
    device=device
)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=device
)


# ============================================================
# Load FAISS + metadata
# ============================================================
def load_index_meta(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["meta"]


pub_index, pub_texts, pub_meta = load_index_meta(PUB_INDEX_PATH, PUB_META_PATH)
med_index, med_texts, med_meta = load_index_meta(MED_INDEX_PATH, MED_META_PATH)


# ============================================================
# RAG Search
# ============================================================
def rag_search(query, top_k=40):
    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    sims_pub, idx_pub = pub_index.search(q_emb, top_k)
    pub_hits = [
        (idx_pub[0][i], sims_pub[0][i], pub_texts[idx_pub[0][i]], pub_meta[idx_pub[0][i]], "PubMedQA")
        for i in range(top_k)
    ]

    sims_med, idx_med = med_index.search(q_emb, top_k)
    med_hits = [
        (idx_med[0][i], sims_med[0][i], med_texts[idx_med[0][i]], med_meta[idx_med[0][i]], "MedQuAD")
        for i in range(top_k)
    ]

    all_hits = pub_hits + med_hits
    ce_inputs = [(query, h[2]) for h in all_hits]
    ce_scores = cross_encoder.predict(ce_inputs)

    best_idx = int(np.argmax(ce_scores))
    best_score = float(ce_scores[best_idx])

    if best_score < CE_THRESHOLD:
        return None, None

    _, _, _, best_meta, source = all_hits[best_idx]
    answer = best_meta.get("long_answer") or best_meta.get("answer") or ""
    return answer.strip(), source


# ============================================================
# Gemini classifier + context generator
# ============================================================
def gemini_medical_classifier_and_context(question: str):
    system_prompt = f"""
You are a strict medical classifier and medical-textbook context generator.  
Always return ONLY JSON. NEVER include any explanation, reasoning, markdown, commentary, or extra text.

Your tasks:

1. Determine if the question is a true MEDICAL QUESTION based on:
   - diseases
   - anatomy
   - physiology
   - symptoms
   - diagnostics
   - treatment
   - pharmacology
   - pathology
   - anything clinically relevant

2. If it is NOT a medical question:
   {{
     "is_medical": false
   }}

3. If it IS a medical question:
   Generate a single paragraph (120–200 words) written in the style of a medical textbook.
   The paragraph must:
   - include the correct answer organically within the paragraph
   - contain supporting medical context around the answer
   - read like a natural excerpt from a clinical reference book
   - avoid bullet points
   - avoid AI-like wording
   - feel like real textbook content

Return strictly in this JSON format:
{{
  "is_medical": true,
  "context": "<textbook style paragraph>"
}}

User question: "{question}"
"""

    try:
        resp = gemini_model.generate_content(system_prompt)

        raw = resp.candidates[0].content.parts[0].text.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(cleaned)
        except:
            return {"is_medical": False, "error": "Failed to parse JSON"}

        return data

    except Exception as e:
        return {"is_medical": False, "error": str(e)}


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI()

def verify_key(request: Request):
    if API_KEY and request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/ask")
def ask(question: str, request: Request):
    verify_key(request)

    question = question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    rag_answer, rag_source = rag_search(question)

    if rag_answer:
        return {
            "question": question,
            "answer": rag_answer,
            "source": rag_source,
            "is_medical": True
        }

    gem = gemini_medical_classifier_and_context(question)

    if gem.get("is_medical") is False:
        return {
            "question": question,
            "answer": "This question is not medical.",
            "source": None,
            "is_medical": False
        }

    return {
        "question": question,
        "answer": gem.get("context", ""),
        "source": "MedBooks",
        "is_medical": True
    }