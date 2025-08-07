import os
from dotenv import load_dotenv
load_dotenv()

SKIP_INGEST = os.environ.get("SKIP_INGEST", "1") == "1"
print(f"🌐 Railway: {os.environ.get('RAILWAY_ENVIRONMENT') == 'true'} · SKIP_INGEST: {SKIP_INGEST}")

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from chromadb import PersistentClient
from ingest_supabase import ingest_supabase_docs
from zip_chroma import zip_chroma_store

# ✅ IMPORTANT: use the same path your ingestion uses on Railway
CHROMA_PATH = "/tmp/chroma_store"

client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("cocktailgpt")

if not SKIP_INGEST:
    print("🚀 Ingesting from Supabase...")
    ingest_supabase_docs(collection)
else:
    print("✅ SKIP_INGEST enabled, skipping ingestion.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- /ask route ----------
from openai import OpenAI
openai_client = OpenAI()

class AskPayload(BaseModel):
    question: str
    history: List[Dict[str, Any]] | None = None

def format_sources(metadatas: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for md in metadatas:
        # works whether values are dicts or strings
        if isinstance(md, dict):
            src = md.get("source") or md.get("path") or "Unknown"
            chunk = md.get("chunk")
        else:
            src = str(md)
            chunk = None
        if chunk is not None:
            lines.append(f"{src} (chunk {chunk})")
        else:
            lines.append(str(src))
    # dedupe while preserving order
    seen = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out[:10]

@app.post("/ask")
def ask(payload: AskPayload):
    q = payload.question.strip()
    if not q:
        return {"response": "Ask me something!"}

    # Retrieve context from Chroma
    results = collection.query(
        query_texts=[q],
        n_results=5,
        include=["documents", "metadatas"]
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    context = ""
    for i, d in enumerate(docs):
        context += f"[{i+1}] {d}\n"

    # Build chat history
    messages = [{"role": "system", "content": "Answer using the provided context. If unknown, say so."}]
    if payload.history:
        # pass through minimal prior turns
        for m in payload.history[-6:]:
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"})

    ai = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    answer = ai.choices[0].message.content.strip()

    sources = format_sources(metas)
    if sources:
        answer = f"{answer}\n\n📚 Sources:\n" + "\n".join(f"- {s}" for s in sources)

    return {"response": answer}
# ---------- end /ask ----------
