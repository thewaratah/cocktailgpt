import os
import io
import json
import requests
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Load env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "cocktailgpt-pdfs")
STATE_FILE = "ingested_files.json"

# --- Supabase client ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Load ingestion history ---
try:
    with open(STATE_FILE, "r") as f:
        previously_ingested = set(json.load(f))
except:
    previously_ingested = set()

# --- Helpers ---
def fetch_file_bytes(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}")
    return BytesIO(response.content)

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def adaptive_chunk_dataframe(df, max_tokens=4000, min_rows=1, max_rows=10):
    def estimate_tokens(text):
        return len(text) // 4
    chunks, i = [], 0
    while i < len(df):
        for rows in range(max_rows, min_rows - 1, -1):
            sub_df = df.iloc[i:i+rows]
            chunk_text = sub_df.to_string(index=False)
            if estimate_tokens(chunk_text) <= max_tokens:
                chunks.append(chunk_text)
                i += rows
                break
        else:
            print(f"⚠️ Skipping single-row chunk at index {i} — exceeds max tokens")
            i += 1
    return chunks

def extract_text_from_csv(csv_bytes):
    df = pd.read_csv(csv_bytes)
    return adaptive_chunk_dataframe(df)

def clean_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def list_all_files(bucket_name, path=""):
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
def ingest_supabase_docs(collection):
    print("🔍 Fetching file list from Supabase (recursive)...")
    files = list_all_files(SUPABASE_BUCKET)
    ingested, skipped = 0, 0

    for file_path in tqdm(files):
        if file_path in previously_ingested:
            print(f"⏭️ Skipping already ingested: {file_path}")
            skipped += 1
            continue

        filename = file_path.split("/")[-1]
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{file_path}"

        try:
            print(f"📄 Processing: {file_path}")
            file_bytes = fetch_file_bytes(url)

            if filename.endswith(".pdf"):
                raw = extract_text_from_pdf(file_bytes)
                cleaned = clean_text(raw)
                chunks = [cleaned[i:i+16000] for i in range(0, len(cleaned), 16000)]

            elif filename.endswith(".csv"):
                chunks = extract_text_from_csv(file_bytes)

            else:
                print(f"⚠️ Skipping unsupported file: {filename}")
                continue

            doc_id = filename.replace(".pdf", "").replace(".csv", "").replace(" ", "_")
            valid_docs, valid_metadatas, valid_ids = [], [], []

            for i, chunk in enumerate(chunks):
                token_estimate = len(chunk.strip()) // 4
                print(f"[DEBUG] Chunk {i} → {len(chunk)} characters, est. {token_estimate} tokens")
                if token_estimate < 20 or token_estimate > 4000:
                    print(f"⚠️ Skipping undersized or oversized chunk: {filename} [chunk {i}] ({token_estimate} tokens)")
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
                        print(f"❌ Batch add failed for {filename} [chunks {i}-{i+batch_size}]: {e}")
                        raise

                previously_ingested.add(file_path)
                with open(STATE_FILE, "w") as f:
                    json.dump(list(previously_ingested), f, indent=2)

                ingested += 1

        except Exception as e:
            print(f"❌ Failed on {file_path}: {e}")

    print(f"✅ Ingestion complete. {ingested} new files processed, {skipped} skipped.")
