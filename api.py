import os
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# Env + startup log
SKIP_INGEST = os.environ.get("SKIP_INGEST", "1") == "1"
IS_RAILWAY = os.environ.get("RAILWAY_ENVIRONMENT") == "true"
print(f"🌐 Railway: {IS_RAILWAY} · SKIP_INGEST: {SKIP_INGEST}")

from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chromadb import PersistentClient
from ingest_supabase import ingest_supabase_docs
from zip_chroma import zip_chroma_store

# ---------- Chroma setup ----------
CHROMA_PATH = "/tmp/chroma_store"  # IMPORTANT: matches ingestion path on Railway
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("cocktailgpt")

if not SKIP_INGEST:
    print("🚀 Ingesting from Supabase...")
    ingest_supabase_docs(collection)
else:
    print("✅ SKIP_INGEST enabled, skipping ingestion.")

# ---------- FastAPI app ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return JSONResponse(
        content={
            "message": "CocktailGPT",
            "info": "Ephemeral vector mode with live Supabase citations",
        }
    )

@app.get("/health")
def health():
    try:
        count = collection.count()
        return {"status": "ok", "chroma_count": count}
    except Exception as e:
        return {"status": "fail", "error": str(e)}

@app.get("/debug/chroma-dir")
def check_chroma_dir():
    path = CHROMA_PATH
    if not os.path.exists(path):
        return {"exists": False}
    files = [os.path.join(path, f) for f in os.listdir(path)]
    return {"exists": True, "files": files}

@app.get("/debug/collections")
def debug_collections():
    """
    Lists all collections in the Chroma DB and their document counts.
    """
    try:
        cols = client.list_collections()
        result = []
        for col in cols:
            try:
                count = col.count()
            except Exception as e:
                count = f"Error: {str(e)}"
            result.append({"name": col.name, "count": count})
        return {"status": "ok", "collections": result}
    except Exception as e:
        return {"status": "fail", "error": str(e)}


# ---------- ZIP + Export ----------
@app.get("/zip-chroma")
def zip_route():
    zip_chroma_store()  # writes /tmp/chroma_store.zip
    return JSONResponse(
        content={"status": "ok", "stdout": "✅ Zipped to /tmp/chroma_store.zip", "stderr": ""}
    )

@app.get("/export-chroma")
def export_chroma():
    zip_path = "/tmp/chroma_store.zip"
    if os.path.exists(zip_path):
        return FileResponse(zip_path, filename="chroma_store.zip", media_type="application/zip")
    return JSONResponse(status_code=404, content={"error": "Vectorstore ZIP not found."})

@app.get("/export-chroma-part/{part_num}")
def export_chroma_chunk(part_num: int = Path(..., ge=1)):
    """
    Serve part of a chunked Chroma ZIP (e.g. /export-chroma-part/1 -> chroma_store_part1.zip)
    """
    file_path = f"/tmp/chroma_store_part{part_num}.zip"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            filename=f"chroma_store_part{part_num}.zip",
            media_type="application/zip",
        )
    return JSONResponse(status_code=404, content={"error": f"Part {part_num} not found."})

# ---------- /ask (RAG) ----------
from openai import OpenAI
openai_client = OpenAI()  # uses OPENAI_API_KEY env

class AskPayload(BaseModel):
    question: str
    history: List[Dict[str, Any]] | None = None  # optional chat history

def _format_sources(metadatas: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for md in metadatas:
        if isinstance(md, dict):
            src = md.get("source") or md.get("path") or "Unknown"
            chunk = md.get("chunk")
        else:
            src = str(md)
            chunk = None
        if chunk is not None:
            lines.append(f"{src} (chunk {chunk})")
        else:
            lines.append(src)
    # dedupe
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
        include=["documents", "metadatas"],
    )
    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []

    # Build context text
    context = ""
    for i, d in enumerate(docs):
        context += f"[{i+1}] {d}\n"

    # Build messages
    messages = [{"role": "system", "content": "Answer using the provided context. If unknown, say so."}]
    if payload.history:
        for m in payload.history[-6:]:
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"})

    # Call OpenAI
    ai = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    answer = ai.choices[0].message.content.strip()

    sources = _format_sources(metas)
    if sources:
        answer = f"{answer}\n\n📚 Sources:\n" + "\n".join(f"- {s}" for s in sources)

    return {"response": answer}
