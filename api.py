import os
import zipfile
import shutil
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from chromadb import PersistentClient
from ingest_supabase import ingest_supabase_docs
from zip_chroma import zip_chroma_store

# ------------------------
# Env & constants
# ------------------------
load_dotenv()

SKIP_INGEST = os.environ.get("SKIP_INGEST", "1") == "1"
IS_RAILWAY = os.environ.get("RAILWAY_ENVIRONMENT") == "true"
CHROMA_DIR = "/tmp/chroma_store"
ZIP_PATH = "/tmp/chroma_store.zip"

print(f"🌐 Railway: {IS_RAILWAY} · SKIP_INGEST: {SKIP_INGEST}")

# ------------------------
# Chroma client
# ------------------------
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("cocktailgpt")

# Optionally ingest at boot (only if SKIP_INGEST == 0)
if not SKIP_INGEST:
    print("🚀 Ingesting from Supabase...")
    ingest_supabase_docs(collection)
else:
    print("✅ SKIP_INGEST enabled, skipping ingestion.")

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Helpers
# ------------------------
def _wipe_tmp_store():
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
        except Exception as e:
            print(f"⚠️ Failed to remove {CHROMA_DIR}: {e}")

def _reopen_collection():
    global client, collection
    client = PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("cocktailgpt")

# ------------------------
# Routes
# ------------------------
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
    if not os.path.exists(CHROMA_DIR):
        return {"exists": False}
    try:
        files = [os.path.join(CHROMA_DIR, f) for f in os.listdir(CHROMA_DIR)]
    except Exception as e:
        return {"exists": True, "error": str(e)}
    return {"exists": True, "files": files}

@app.get("/debug/collections")
def debug_collections():
    try:
        # Chroma v0.5+ client doesn’t list collections directly; we read current one.
        return {"status": "ok", "collections": [{"name": "cocktailgpt", "count": collection.count()}]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})

# ----- Export (zip -> download) -----
@app.get("/zip-chroma")
def zip_route():
    try:
        zip_chroma_store()  # writes /tmp/chroma_store.zip
        return JSONResponse(content={"status": "ok", "stdout": f"✅ Zipped to {ZIP_PATH}", "stderr": ""})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})

@app.get("/export-chroma")
def export_chroma():
    if os.path.exists(ZIP_PATH):
        return FileResponse(ZIP_PATH, filename="chroma_store.zip", media_type="application/zip")
    return JSONResponse(status_code=404, content={"error": "Vectorstore ZIP not found."})

# Optional: serve chunked parts if you ever create them
@app.get("/export-chroma-part/{part_num}")
def export_chroma_chunk(part_num: int = Path(..., ge=1)):
    part_path = f"/tmp/chroma_store_part{part_num}.zip"
    if os.path.exists(part_path):
        return FileResponse(part_path, filename=f"chroma_store_part{part_num}.zip", media_type="application/zip")
    return JSONResponse(status_code=404, content={"error": f"Part {part_num} not found."})

# ----- NEW: Upload & restore (no re-ingest needed) -----
@app.post("/upload-chroma")
async def upload_chroma(file: UploadFile = File(...)):
    """
    Upload a local chroma_store.zip to the server (saves to /tmp/chroma_store.zip).
    Use this when the zip is too large for GitHub.
    """
    try:
        with open(ZIP_PATH, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
        return {"status": "ok", "saved_to": ZIP_PATH}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})

@app.post("/force-restore")
def force_restore():
    """
    Wipe /tmp/chroma_store and re-extract /tmp/chroma_store.zip into it.
    """
    try:
        if not os.path.exists(ZIP_PATH):
            return JSONResponse(status_code=404, content={"error": "No /tmp/chroma_store.zip present."})

        _wipe_tmp_store()
        os.makedirs(CHROMA_DIR, exist_ok=True)

        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(CHROMA_DIR)

        _reopen_collection()
        return {"status": "ok", "restored": True, "count": collection.count()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})
