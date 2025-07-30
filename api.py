from fastapi import FastAPI
from pydantic import BaseModel
from query import ask
from ingest_supabase import ingest_supabase_docs
from query import collection

app = FastAPI()


# --- Run ingestion once ---
@app.get("/run-once")
def run_once():
    try:
        ingest_supabase_docs(collection)
        return {"status": "done", "count": collection.count()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- Ask endpoint (still live, optional) ---
class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_endpoint(data: AskRequest):
    try:
        answer = ask(data.question)
        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}

import subprocess

@app.get("/zip-chroma")
def zip_chroma():
    try:
        result = subprocess.run(
            ["python", "zip_chroma.py"],
            capture_output=True,
            text=True
        )
        return {
            "status": "ok",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

from fastapi.responses import FileResponse

@app.get("/download-chroma")
def download_chroma():
    return FileResponse("/tmp/chroma_store.zip", filename="chroma_store.zip")

@app.get("/health")
def health():
    try:
        count = collection.count()
        return {"status": "ok", "chroma_count": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


