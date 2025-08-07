import os
import re
import pandas as pd
import numpy as np
import faiss
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Disable symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === CONFIGURATION ===
CSV_PATH = "questions.csv"
MODEL_NAME = "intfloat/multilingual-e5-base"
THRESHOLD = 0.7

# === NORMALIZATION FUNCTION ===
def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9ҳғўқҳӣі\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === LOAD DATA ===
print("📄 Loading CSV data...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["Questions", "Responsible (0-Operator, 1-AI Agent)"])
df["Questions_norm"] = df["Questions"].apply(normalize)

df_ai = df[df["Responsible (0-Operator, 1-AI Agent)"] == 1].reset_index(drop=True)
df_op = df[df["Responsible (0-Operator, 1-AI Agent)"] == 0].reset_index(drop=True)

# === LOAD MODEL ===
print("⏳ Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("🔍 Encoding questions...")
emb_ai = model.encode(df_ai["Questions_norm"].tolist())
emb_op = model.encode(df_op["Questions_norm"].tolist())

# === BUILD INDEXES ===
dim = emb_ai.shape[1]
index_ai = faiss.IndexFlatL2(dim)
index_ai.add(np.array(emb_ai))

index_op = faiss.IndexFlatL2(dim)
index_op.add(np.array(emb_op))

print("✅ Indexes ready!")

# === FASTAPI SETUP ===
app = FastAPI()

class MessageRequest(BaseModel):
    message: str

class ClassificationResponse(BaseModel):
    route: str
    matched_question: str | None
    similarity: float

@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: MessageRequest):
    message = request.message
    msg_norm = normalize(message)

    if len(msg_norm.strip()) < 3 or not re.search(r'[a-zA-Zа-яА-Я0-9]', msg_norm):
        return ClassificationResponse(route="AI", matched_question="(auto-routed due to unclear input)", similarity=0.0)

    emb_msg = model.encode([msg_norm])
    D_ai, I_ai = index_ai.search(np.array(emb_msg), k=1)
    sim_ai = 1 - D_ai[0][0] / 2

    D_op, I_op = index_op.search(np.array(emb_msg), k=1)
    sim_op = 1 - D_op[0][0] / 2

    if max(sim_ai, sim_op) < THRESHOLD:
        return ClassificationResponse(route="Operator", matched_question=None, similarity=max(sim_ai, sim_op))

    if sim_ai > sim_op:
        matched = df_ai.iloc[I_ai[0][0]]["Questions"]
        return ClassificationResponse(route="AI", matched_question=matched, similarity=sim_ai)
    else:
        matched = df_op.iloc[I_op[0][0]]["Questions"]
        return ClassificationResponse(route="Operator", matched_question=matched, similarity=sim_op)

@app.get("/")
async def root():
    return {"message": "Message router is running. POST to /classify"}
