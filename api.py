import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from ingest_supabase import ingest_supabase_docs
from zip_chroma import zip_chroma_store
import shutil
import zipfile

SKIP_INGEST = os.environ.get("SKIP_INGEST", "1") == "1"
print(f"🌐 Railway: {os.environ.get('RAILWAY_ENVIRONMENT') == 'true'} · SKIP_INGEST: {SKIP_INGEST}")

CHROMA_DIR = "/tmp/chroma_store"
COMBINED_ZIP = "/tmp/chroma_store.zip"
UPLOADS_DIR = "/tmp/upload_parts"

# Ensure dirs exist
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Chroma client/collection
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("cocktailgpt")

# Optional ingestion (only when SKIP_INGEST=0)
if not SKIP_INGEST:
    print("🚀 Ingesting from Supabase...")
    ingest_supabase_docs(collection)
else:
    print("✅ SKIP_INGEST enabled, skipping ingestion.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "CocktailGPT", "info": "Ephemeral vector mode with live Supabase citations"}

@app.get("/health")
def health():
    try:
        return {"status": "ok", "chroma_count": collection.count()}
    except Exception as e:
        return {"status": "fail", "error": str(e)}

@app.get("/debug/chroma-dir")
def check_chroma_dir():
    if not os.path.exists(CHROMA_DIR):
        return {"exists": False}
    files = [os.path.join(CHROMA_DIR, f) for f in os.listdir(CHROMA_DIR)]
    return {"exists": True, "files": files}

@app.get("/debug/collections")
def debug_collections():
    try:
        names = [c.name for c in client.list_collections()]
        detail = [{"name": n, "count": client.get_collection(n).count()} for n in names]
        return {"status": "ok", "collections": detail}
    except Exception as e:
        return {"status": "fail", "error": str(e)}

# ------- ZIP/EXPORT (download from server) -------
@app.get("/zip-chroma")
def zip_route():
    zip_chroma_store()
    return {"status": "ok", "stdout": "✅ Zipped to /tmp/chroma_store.zip", "stderr": ""}

@app.get("/export-chroma")
def export_chroma():
    if os.path.exists(COMBINED_ZIP):
        return FileResponse(COMBINED_ZIP, filename="chroma_store.zip", media_type="application/zip")
    return JSONResponse(status_code=404, content={"error": "Vectorstore ZIP not found."})

# ------- UPLOAD (single zip) -------
@app.post("/upload-chroma")
async def upload_chroma(file: UploadFile = File(...)):
    os.makedirs("/tmp", exist_ok=True)
    with open(COMBINED_ZIP, "wb") as f:
        f.write(await file.read())
    size = os.path.getsize(COMBINED_ZIP)
    return {"status": "ok", "message": f"Uploaded {file.filename} → {COMBINED_ZIP}", "size": size}

# ------- UPLOAD (chunked parts) -------
@app.post("/upload-chroma-part/{part_num}")
async def upload_chroma_part(part_num: int = Path(..., ge=1), file: UploadFile = File(...)):
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    part_path = os.path.join(UPLOADS_DIR, f"chroma_store_part{part_num}.zip")
    with open(part_path, "wb") as f:
        f.write(await file.read())
    return {"status": "ok", "message": f"Saved part {part_num} to {part_path}"}

@app.post("/assemble-uploaded-zip")
def assemble_uploaded_zip():
    # concatenate parts in numeric order
    parts = sorted(
        [p for p in os.listdir(UPLOADS_DIR) if p.startswith("chroma_store_part") and p.endswith(".zip")],
        key=lambda x: int(x.replace("chroma_store_part", "").replace(".zip", ""))
    )
    if not parts:
        return JSONResponse(status_code=400, content={"error": "No parts found in /tmp/upload_parts"})
    with open(COMBINED_ZIP, "wb") as out:
        for p in parts:
            with open(os.path.join(UPLOADS_DIR, p), "rb") as part:
                shutil.copyfileobj(part, out)
    size = os.path.getsize(COMBINED_ZIP)
    return {"status": "ok", "message": f"Assembled {len(parts)} parts → {COMBINED_ZIP}", "size": size}

# ------- RESTORE from /tmp/chroma_store.zip -------
@app.post("/force-restore")
def force_restore():
    if not os.path.exists(COMBINED_ZIP):
        return JSONResponse(status_code=400, content={"error": "No /tmp/chroma_store.zip present."})

    # wipe existing dir and unzip fresh
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    with zipfile.ZipFile(COMBINED_ZIP, "r") as zf:
        zf.extractall(CHROMA_DIR)

    # reopen the collection
    global client, collection
    client = PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("cocktailgpt")
    count = collection.count()
    return {"status": "ok", "message": "Restored Chroma from zip.", "count": count}
