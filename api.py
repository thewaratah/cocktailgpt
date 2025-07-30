import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from ingest_supabase import ingest_supabase_docs
from query import ask, collection
import subprocess
import io
import zipfile

app = FastAPI()


@app.post("/ask")
async def ask_endpoint(request: Request):
    data = await request.json()
    question = data.get("question")
    tags = data.get("tags", None)
    if not question:
        return {"error": "Missing question"}
    answer = ask(question, tags=tags)
    return {"response": answer}


@app.get("/run-once")
def run_once():
    ingest_supabase_docs(collection)
    return {"status": "done", "count": collection.count()}


@app.get("/zip-chroma")
def zip_chroma():
    result = subprocess.run(
        ["python3", "zip_chroma.py"], capture_output=True, text=True
    )
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.get("/download-chroma")
def download_chroma():
    zip_path = "/tmp/chroma_store.zip"
    if not os.path.exists(zip_path):
        return {"error": "No chroma_store.zip found in /tmp"}
    return FileResponse(zip_path, filename="chroma_store.zip")


@app.get("/health")
def health():
    try:
        count = collection.count()
        return {"status": "ok", "chroma_count": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/export-chroma")
def export_chroma():
    chroma_dir = "/tmp/chroma_store"
    zip_stream = io.BytesIO()

    if not os.path.exists(chroma_dir):
        return {"status": "error", "message": "Chroma directory does not exist"}

    try:
        with zipfile.ZipFile(zip_stream, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(chroma_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, chroma_dir)
                    zf.write(full_path, arcname=arcname)

        zip_stream.seek(0)
        return StreamingResponse(zip_stream, media_type="application/zip", headers={
            "Content-Disposition": "attachment; filename=chroma_store_live.zip"
        })
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/debug/chroma-dir")
def debug_chroma_dir():
    path = "/tmp/chroma_store"
    if not os.path.exists(path):
        return {"exists": False}
    files = []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            files.append(os.path.join(root, f))
    return {"exists": True, "files": files}
