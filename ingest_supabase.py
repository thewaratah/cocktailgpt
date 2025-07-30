import os
from supabase import create_client
from chromadb import Client
from chromadb.config import Settings
from utils import extract_text_from_pdf, clean_text, chunk_text
import hashlib
import json
from tqdm import tqdm

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "cocktailgpt-pdfs")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ✅ Chroma v3 client
client = Client(Settings(
    anonymized_telemetry=False,
    persist_directory="/tmp/chroma_store"
))
collection = client.get_or_create_collection("cocktailgpt")

# Load ingestion state
ingested_path = "ingested_files.json"
if os.path.exists(ingested_path):
    with open(ingested_path) as f:
        ingested = json.load(f)
else:
    ingested = {}

def ingest_supabase_docs(collection):
    print(f"🌐 Railway: {os.environ.get('RAILWAY_ENVIRONMENT') == 'true'} · SKIP_INGEST: {os.environ.get('SKIP_INGEST') == '1'}")
    print("🔍 Fetching files from Supabase...")

    files = []
    res = supabase.storage.from_(SUPABASE_BUCKET).list("pdfs/")
    for file in res:
        if file["name"].endswith(".pdf") or file["name"].endswith(".csv"):
            files.append(f"pdfs/{file['name']}")

    print(f"📁 Files found: {files}")
    skipped = 0
    added = 0

    for filepath in tqdm(files):
        filename = filepath.split("/")[-1]

        if filename in ingested:
            skipped += 1
            continue

        try:
            response = supabase.storage.from_(SUPABASE_BUCKET).download(filepath)
            file_bytes = response

            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_bytes)
            elif filename.endswith(".csv"):
                text = file_bytes.decode("utf-8")
            else:
                continue

            clean = clean_text(text)
            chunks = chunk_text(clean)

            if not chunks:
                print(f"⚠️ No chunks from {filename}")
                continue

            valid_docs = []
            valid_metadatas = []
            valid_ids = []

            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.sha256((filename + str(i)).encode()).hexdigest()
                valid_docs.append(chunk)
                valid_metadatas.append({"source": filename, "chunk": i})
                valid_ids.append(chunk_id)

            if valid_docs:
                for i in range(0, len(valid_docs), 20):
                    batch_ids = valid_ids[i:i+20]
                    try:
                        collection.delete(ids=batch_ids)
                    except:
                        pass

                    collection.add(
                        documents=valid_docs[i:i+20],
                        metadatas=valid_metadatas[i:i+20],
                        ids=batch_ids
                    )

            ingested[filename] = True
            added += 1

        except Exception as e:
            print(f"❌ Failed on {filepath}: {e}")

    with open(ingested_path, "w") as f:
        json.dump(ingested, f)

    print(f"✅ Done. {added} files ingested, {skipped} skipped.")
