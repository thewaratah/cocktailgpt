import os
import io
import json
import requests
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv
from chromadb import EphemeralClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from supabase import create_client, Client

# --- Load env ---
load_dotenv()

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "cocktailgpt-pdfs")

# --- ChromaDB setup ---
os.environ["CHROMA_OPENAI_API_KEY"] = OPENAI_API_KEY
client = EphemeralClient()
embedding_function = OpenAIEmbeddingFunction()
collection = client.get_or_create_collection(
    name="cocktail_docs",
    embedding_function=embedding_function
)

# --- Supabase client ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helpers ---

def fetch_file_bytes(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}")
    return BytesIO(response.content)

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_csv(csv_bytes):
    df = pd.read_csv(csv_bytes)
    return df.to_string(index=False)

def clean_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def chunk_text(text, max_tokens=300):
    paras = text.split("\n")
    chunks, current = [], ""
    for para in paras:
        if len(current + para) < max_tokens * 4:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def list_all_files(bucket_name, path=""):
    """Recursively list all PDF/CSV files in Supabase Storage with pagination."""
    files = []
    limit = 100
    offset = 0

    while True:
        print(f"[DEBUG] Listing: {path} (offset={offset})")
        result = supabase.storage.from_(bucket_name).list(path, {"limit": limit, "offset": offset})
        if not result:
            break

        for item in result:
            if item["name"].startswith("."):
                continue
            full_path = f"{path}/{item['name']}" if path else item["name"]
            if "." not in item["name"]:
                files.extend(list_all_files(bucket_name, full_path))
            elif item["name"].endswith(".pdf") or item["name"].endswith(".csv"):
                files.append(full_path)

        if len(result) < limit:
            break
        offset += limit

    return files

# --- Main ingestion function ---

def ingest_supabase_docs():
    print("🔍 Fetching file list from Supabase (recursive)...")
    files = list_all_files(SUPABASE_BUCKET)
    ingested = 0

    for file_path in tqdm(files):
        filename = file_path.split("/")[-1]
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{file_path}"
        try:
            print(f"📄 Processing: {file_path}")
            file_bytes = fetch_file_bytes(url)

            if filename.endswith(".pdf"):
                raw = extract_text_from_pdf(file_bytes)
            elif filename.endswith(".csv"):
                raw = extract_text_from_csv(file_bytes)
            else:
                print(f"⚠️ Skipping unsupported file: {filename}")
                continue

            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned)
            doc_id = filename.replace(".pdf", "").replace(".csv", "").replace(" ", "_")

            valid_docs, valid_metadatas, valid_ids = [], [], []

            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) == 0 or len(chunk) > 16000:
                    print(f"⚠️ Skipping empty or oversized chunk: {filename} [chunk {i}]")
                    continue
                metadata = {
                    "source": filename,
                    "chunk": i,
                    "chunk_id": i,
                    "path": file_path
                }
                valid_docs.append(chunk)
                valid_metadatas.append(metadata)
                valid_ids.append(f"{doc_id}_{i}")

            if valid_docs:
                batch_size = 100
                for i in range(0, len(valid_docs), batch_size):
                    try:
                        collection.add(
                            documents=valid_docs[i:i+batch_size],
                            metadatas=valid_metadatas[i:i+batch_size],
                            ids=valid_ids[i:i+batch_size]
                        )
                    except Exception as e:
                        print(f"❌ Batch add failed for {filename} [chunks {i}–{i+batch_size}]: {e}")
                        raise

                ingested += 1

        except Exception as e:
            print(f"❌ Failed on {file_path}: {e}")

    print(f"✅ Ingestion complete. {ingested} files processed.")

# --- Run manually (local only) ---
if __name__ == "__main__":
    ingest_supabase_docs()
