import os
import io
import zipfile
import shutil
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from chromadb import PersistentClient
from openai import OpenAI

# Optional helpers you already have
from ingest_supabase import ingest_supabase_docs
from zip_chroma import zip_chroma_store
from utils import format_response_with_citations

# ---------- Env / Paths ----------
load_dotenv()

# Behaviour controls (you can override via Railway env vars if you want)
# Keep UK English always; metric units; ~2x detail
LOCALE = os.environ.get("LOCALE", "en-GB")
RESPONSE_DETAIL = os.environ.get("RESPONSE_DETAIL", "double")  # "default" | "double"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4-turbo")
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "900"))  # allow longer answers

SKIP_INGEST = os.environ.get("SKIP_INGEST", "1") == "1"

CHROMA_PATH = "/tmp/chroma_store"
UPLOAD_PARTS_DIR = "/tmp/upload_parts"
ZIP_PATH = "/tmp/chroma_store.zip"

os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(UPLOAD_PARTS_DIR, exist_ok=True)

print(f"✅ API starting | LOCALE={LOCALE} | DETAIL={RESPONSE_DETAIL} | MODEL={OPENAI_MODEL} | MAX_TOKENS={OPENAI_MAX_TOKENS} | SKIP_INGEST={SKIP_INGEST}")

# ---------- Chroma client (global) ----------
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("cocktailgpt")

# ---------- OpenAI client (global) ----------
openai_api_key = os.getenv("OPENAI_API_KEY")
oa = OpenAI(api_key=openai_api_key)

# ---------- Optional ingestion on boot ----------
if not SKIP_INGEST:
    print("🚀 Ingesting from Supabase...")
    try:
        ingest_supabase_docs(collection)
    except Exception as e:
        print(f"❌ Ingest failed: {e}")
else:
    print("⏩ SKIP_INGEST=1, skipping ingestion on boot.")

# ---------- FastAPI ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten if you want to lock to Softr domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def reopen_collection() -> bool:
    """Reopen Chroma after replacing /tmp/chroma_store."""
    global client, collection
    try:
        client = PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection("cocktailgpt")
        return True
    except Exception as e:
        print(f"❌ reopen_collection failed: {e}")
        return False


def results_to_sources(results: Dict[str, Any]) -> List[str]:
    """Convert Chroma query results to a flat list of 'filename (chunk N)' strings."""
    metas = results.get("metadatas") or []
    if not metas:
        return []
    seen = set()
    out: List[str] = []
    for m in metas[0]:
        source = None
        chunk = None
        if isinstance(m, dict):
            source = m.get("source") or m.get("path") or "Unknown"
            chunk = m.get("chunk")
        elif isinstance(m, str):
            source = m
        label = f"{source}" if chunk is None else f"{source} (chunk {chunk})"
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def build_context_from_results(results: Dict[str, Any]) -> str:
    """Build a quoted context block from the top documents returned by Chroma."""
    docs = results.get("documents") or [[]]
    metas = results.get("metadatas") or [[]]
    rows = []
    for i, (doc, meta) in enumerate(zip(docs[0], metas[0]), start=1):
        src = meta.get("source") if isinstance(meta, dict) else "Unknown"
        ch = meta.get("chunk") if isinstance(meta, dict) else None
        head = f"[{i}] {src}" + (f" (chunk {ch})" if ch is not None else "")
        body = doc.strip()
        rows.append(f"{head}\n{body}")
    return "\n\n".join(rows)


def make_system_prompt() -> str:
    """System prompt that enforces UK English and ~2x detail."""
    base = (
        "You are CocktailGPT, a precise assistant for cocktails, flavour, and food science.\n"
        f"- Use {LOCALE} spelling and grammar (UK English).\n"
        "- Use metric units (mL, grams, °C) by default; include imperial only if the source uses it explicitly.\n"
        "- Use ingredient and technique names faithfully as written in the sources.\n"
        "- If the answer is not present in the provided context, say you don't know.\n"
        "- Always stay faithful to the context; do not invent sources.\n"
    )
    if RESPONSE_DETAIL.lower() == "double":
        base += (
            "\nDetail & structure:\n"
            "- Provide approximately twice the normal detail.\n"
            "- When relevant, add two short sections at the end:\n"
            "  * Why it matters — one or two sentences that explain the practical significance.\n"
            "  * Practical tips — concise, actionable pointers a bartender or chef can use immediately.\n"
        )
    return base


# ---------- Routes ----------
@app.get("/")
def root():
    return JSONResponse(content={
        "message": "CocktailGPT",
        "info": "FastAPI backend with Chroma; UK English + detailed responses enabled"
    })


@app.get("/health")
def health():
    try:
        _ = client.list_collections()  # touch to avoid stale handle
        count = collection.count()
        return {"status": "ok", "chroma_count": count, "locale": LOCALE, "detail": RESPONSE_DETAIL}
    except Exception as e:
        return {"status": "fail", "error": str(e)}


@app.get("/debug/chroma-dir")
def check_chroma_dir():
    if not os.path.exists(CHROMA_PATH):
        return {"exists": False}
    files = [os.path.join(CHROMA_PATH, f) for f in os.listdir(CHROMA_PATH)]
    return {"exists": True, "files": files}


@app.get("/debug/collections")
def list_collections():
    try:
        cols = client.list_collections()
        return {
            "status": "ok",
            "collections": [{"name": c.name, "count": client.get_collection(c.name).count()} for c in cols]
        }
    except Exception as e:
        return {"status": "fail", "error": str(e)}


# ---------- RAG: Ask ----------
@app.post("/ask")
def ask(payload: Dict[str, Any]):
    """
    Body: {
      "question": str,
      "history": Optional[List[{"role":"user"|"assistant","content":str}]]
    }
    Returns: { "response": str, "sources": [str, ...] }
    """
    try:
        question: str = (payload or {}).get("question", "").strip()
        history = (payload or {}).get("history") or []
        if not question:
            return JSONResponse(status_code=400, content={"error": "Question is required."})

        # Query Chroma
        results = collection.query(
            query_texts=[question],
            n_results=5,
            include=["documents", "metadatas"]
        )

        # Build prompt with retrieved context
        context_block = build_context_from_results(results)
        sys_msg = make_system_prompt()

        msgs = [{"role": "system", "content": sys_msg}]
        # Optionally include a little recent history to keep style consistent
        for m in history[-8:]:
            r = m.get("role")
            c = m.get("content")
            if r in ("user", "assistant") and isinstance(c, str):
                msgs.append({"role": r, "content": c})
        msgs.append({"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"}))

        # Call OpenAI
        completion = oa.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.2,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        answer = completion.choices[0].message.content.strip()

        # Build structured sources list
        sources = results_to_sources(results)

        # Also create a '📚 Sources:' block for legacy clients (fallback)
        answer_with_block = format_response_with_citations(answer, results)

        return {
            "response": answer_with_block,   # includes '📚 Sources:' fallback
            "sources": sources               # preferred by the Streamlit UI
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


# ---------- Export / Maintenance ----------
@app.get("/zip-chroma")
def zip_route():
    try:
        zip_chroma_store()
        return JSONResponse(content={"status": "ok", "stdout": f"✅ Zipped to {ZIP_PATH}", "stderr": ""})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@app.get("/export-chroma")
def export_chroma():
    if os.path.exists(ZIP_PATH):
        return FileResponse(ZIP_PATH, filename="chroma_store.zip", media_type="application/zip")
    return JSONResponse(status_code=404, content={"error": "Vectorstore ZIP not found."})


@app.post("/upload-chroma")
async def upload_chroma(file: UploadFile = File(...)):
    try:
        with open(ZIP_PATH, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        size = os.path.getsize(ZIP_PATH)
        return {"status": "ok", "message": f"Uploaded to {ZIP_PATH}", "size": size}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@app.post("/upload-chroma-part/{part_num}")
async def upload_chroma_part(part_num: int = Path(..., ge=1), file: UploadFile = File(...)):
    try:
        part_path = os.path.join(UPLOAD_PARTS_DIR, f"chroma_store_part{part_num}.zip")
        with open(part_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        size = os.path.getsize(part_path)
        return {"status": "ok", "message": f"Saved part {part_num} to {part_path}", "size": size}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@app.post("/assemble-uploaded-zip")
def assemble_uploaded_zip():
    """Concatenate chroma_store_partN.zip (N ascending) -> /tmp/chroma_store.zip."""
    try:
        parts = [p for p in os.listdir(UPLOAD_PARTS_DIR) if p.startswith("chroma_store_part") and p.endswith(".zip")]
        if not parts:
            return JSONResponse(status_code=400, content={"error": "No parts found in /tmp/upload_parts"})

        def part_index(name: str) -> int:
            base = os.path.splitext(name)[0]
            return int(base.replace("chroma_store_part", ""))

        parts_sorted = sorted(parts, key=part_index)

        with open(ZIP_PATH, "wb") as out:
            for p in parts_sorted:
                with open(os.path.join(UPLOAD_PARTS_DIR, p), "rb") as src:
                    shutil.copyfileobj(src, out)

        size = os.path.getsize(ZIP_PATH)

        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                _ = zf.namelist()
        except Exception as ze:
            return JSONResponse(status_code=500, content={"status": "error", "error": f"ZIP verify failed: {ze}"})

        return {"status": "ok", "message": f"Assembled {len(parts_sorted)} parts → {ZIP_PATH}", "size": size}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@app.post("/force-restore")
def force_restore():
    """Delete /tmp/chroma_store, unzip /tmp/chroma_store.zip into it, reopen collection."""
    try:
        if not os.path.exists(ZIP_PATH):
            return JSONResponse(status_code=400, content={"error": f"No {ZIP_PATH} present."})

        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        os.makedirs(CHROMA_PATH, exist_ok=True)

        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(CHROMA_PATH)

        ok = reopen_collection()
        if not ok:
            return JSONResponse(status_code=500, content={"status": "error", "error": "Failed to reopen collection"})

        count = collection.count()
        return {"status": "ok", "message": "Restored Chroma from ZIP", "count": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@app.get("/export-chroma-part/{part_num}")
def export_chroma_chunk(part_num: int = Path(..., ge=1)):
    file_path = f"/tmp/chroma_store_part{part_num}.zip"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=f"chroma_store_part{part_num}.zip", media_type="application/zip")
    return JSONResponse(status_code=404, content={"error": f"Part {part_num} not found."})
