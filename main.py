import os
import re
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ✅ Normalize text for comparison
def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9ҳғўқҳӣі\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ✅ Load updated dataset
print("📄 Loading updated CSV...")
df = pd.read_csv("questions.csv")
df = df.dropna(subset=["Questions", "Responsible (0-Operator, 1-AI Agent)"])
df["Questions_norm"] = df["Questions"].apply(normalize)

# ✅ Split into AI and Operator questions
df_ai = df[df["Responsible (0-Operator, 1-AI Agent)"] == 1].reset_index(drop=True)
df_op = df[df["Responsible (0-Operator, 1-AI Agent)"] == 0].reset_index(drop=True)

# ✅ Load embedding model
print("⏳ Loading multilingual model...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

# ✅ Encode both sets
print("🔍 Creating embeddings...")
emb_ai = model.encode(df_ai["Questions_norm"].tolist())
emb_op = model.encode(df_op["Questions_norm"].tolist())

# ✅ Create two FAISS indexes
dim = emb_ai.shape[1]
index_ai = faiss.IndexFlatL2(dim)
index_ai.add(np.array(emb_ai))

index_op = faiss.IndexFlatL2(dim)
index_op.add(np.array(emb_op))

print("✅ FAISS indexes for AI and Operator are ready!")

# ✅ Classifier
def classify_message(message, threshold=0.7):
    msg_norm = normalize(message)

    # Route gibberish/empty to AI
    if len(msg_norm.strip()) < 3 or not re.search(r'[a-zA-Zа-яА-Я0-9]', msg_norm):
        return "AI", "(auto-routed due to unclear input)", 0.0

    emb_msg = model.encode([msg_norm])

    D_ai, I_ai = index_ai.search(np.array(emb_msg), k=1)
    sim_ai = 1 - D_ai[0][0] / 2

    D_op, I_op = index_op.search(np.array(emb_msg), k=1)
    sim_op = 1 - D_op[0][0] / 2

    if max(sim_ai, sim_op) < threshold:
        return "Operator", None, max(sim_ai, sim_op)

    if sim_ai > sim_op:
        matched = df_ai.iloc[I_ai[0][0]]["Questions"]
        return "AI", matched, sim_ai
    else:
        matched = df_op.iloc[I_op[0][0]]["Questions"]
        return "Operator", matched, sim_op


# ✅ Run test loop
print("\n🔄 Classifier is ready. Type your messages (type 'exit' to quit):")
while True:
    msg = input("💬 User message: ")
    if msg.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break

    route, matched, sim = classify_message(msg)
    print(f"➡️ Route to: {route}")
    print(f"📈 Similarity: {sim:.2f}")
    if matched:
        print(f"🔁 Matched Question: {matched}")
