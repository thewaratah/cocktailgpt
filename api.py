import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from ingest_supabase import ingest_supabase_docs
from zip_chroma import zip_chroma_store
from dotenv import load_dotenv

load_dotenv()

SKIP_INGEST = os.environ.get("SKIP_INGEST", "1") == "1"

client = PersistentClient(path="/tmp/chroma_store")
collection = client.get_or_create_collection("cocktailgpt")

if not SKIP_INGEST:
    ingest_supabase_docs(collection)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return JSONResponse(content={
        "message": "CocktailGPT",
        "info": "Ephemeral vector mode with live Supabase citations"
    })

@app.get("/health")
def health():
    try:
        count = collection.count()
        return {"status": "ok", "chroma_count": count}
    except Exception as e:
        return {"status": "fail", "error": str(e)}

@app.get("/debug/chroma-dir")
def check_chroma_dir():
    path = "/tmp/chroma_store"
    if not os.path.exists(path):
        return {"exists": False}
    files = [os.path.join(path, f) for f in os.listdir(path)]
    return {"exists": True, "files": files}

@app.get("/zip-chroma")
def zip_route():
    zip_chroma_store()
    return JSONResponse(content={"status": "ok", "stdout": "✅ Zipped to /tmp/chroma_store.zip", "stderr": ""})

@app.get("/export-chroma")
def export_chroma():
    zip_path = "/tmp/chroma_store.zip"
    if os.path.exists(zip_path):
        return FileResponse(zip_path, filename="chroma_store.zip", media_type="application/zip")
    return JSONResponse(status_code=404, content={"error": "Vectorstore ZIP not found."})
