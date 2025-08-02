import os
from dotenv import load_dotenv
load_dotenv()

SKIP_INGEST = os.environ.get("SKIP_INGEST", "1") == "1"
print(f"🌐 Railway: {os.environ.get('RAILWAY_ENVIRONMENT') == 'true'} · SKIP_INGEST: {SKIP_INGEST}")

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from ingest_supabase import ingest_supabase_docs
from zip_chroma import zip_chroma_store

client = PersistentClient(path="chroma_store")
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

from fastapi import Path

@app.get("/export-chroma-part/{part_num}")
def export_chroma_chunk(part_num: int = Path(..., ge=1)):
    """
    Serves part of the chunked Chroma ZIP (e.g. /export-chroma-part/1 → chroma_store_part1.zip)
    """
    file_path = f"/tmp/chroma_store_part{part_num}.zip"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            filename=f"chroma_store_part{part_num}.zip",
            media_type="application/zip"
        )
    return JSONResponse(
        status_code=404,
        content={"error": f"Part {part_num} not found."}
    )
