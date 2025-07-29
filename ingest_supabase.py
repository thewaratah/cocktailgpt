import os
import io
import json
import requests
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
from supabase import create_client, Client
from ebooklib import epub
from bs4 import BeautifulSoup
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb import PersistentClient


# --- Load environment ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "cocktailgpt-pdfs")
STATE_FILE = "ingested_files.json"

# --- Clients ---
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
client = PersistentClient(path="/tmp/chroma_store")
collection = client.get_or_create_collection("cocktail_docs", embedding_function=embedding_function)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- State tracking ---
try:
    with open(STATE_FILE, "r") as f:
        previously_ingested = set(json.load(f))
except:
    previously_ingested = set()

# --- Extraction helpers ---
def fetch_file_bytes(url): return BytesIO(requests.get(url).content)

def extract_text_from_pdf(pdf_bytes):
    return "\n".join([p.get_text() for p in fitz.open(stream=pdf_bytes, filetype="pdf")])

def extract_text_from_csv(csv_bytes):
    df = pd.read_csv(csv_bytes)
    return adaptive_chunk_dataframe(df)

def extract_text_from_epub(epub_bytes):
    book = epub.read_epub(epub_bytes)
    return "\n".join([
        BeautifulSoup(item.get_content(), "html.parser").get_text(separator="\n")
        for item in book.get_items() if item.get_type() == epub.EpubHtml
    ])

def clean_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def adaptive_chunk_dataframe(df, max_tokens=4000, min_rows=1, max_rows=10):
    def token_count(text): return len(text) // 4
    chunks, i = 0, 0
    output = []
    while i < len(df):
        for rows in range(max_rows, min_rows - 1, -1):
            sub_df = df.iloc[i:i+rows]
            text = sub_df.to_string(index=False)
            if token_count(text) <= max_tokens:
                output.append(text)
                i += rows
                break
        else:
            i += 1
    return output

def list_all_files(bucket_name, path=""):
    files = []
    offset = 0
    limit = 100
    while True:
        items = supabase.storage.from_(bucket_name).list(path, {"limit": limit, "offset": offset})
        if not items:
            break
        for item in items:
            if item["name"].startswith("."):
                continue
            full_path = f"{path}/{item['name']}"
            if item["name"].endswith((".pdf", ".csv", ".epub")):
                files.append(full_path)
        if len(items) < limit:
            break
        offset += limit
    return files

def ingest_supabase_docs(collection):
    print("🔍 Fetching files from Supabase...")
    files = list_all_files(SUPABASE_BUCKET, "pdfs")
    print("📁 Files found:", files)

    ingested, skipped = 0, 0

    for file_path in tqdm(files):
        if file_path in previously_ingested:
            skipped += 1
            continue

        filename = file_path.split("/")[-1]
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{file_path}"

        try:
            file_bytes = fetch_file_bytes(url)

            # --- Extract and chunk ---
            if filename.endswith(".pdf"):
                raw = extract_text_from_pdf(file_bytes)
                max_chunk_size = 8000  # chars ≈ 2000 tokens
                chunks = [clean_text(raw[i:i+max_chunk_size]) for i in range(0, len(raw), max_chunk_size)]
            elif filename.endswith(".csv"):
                chunks = extract_text_from_csv(file_bytes)
            elif filename.endswith(".epub"):
                raw = extract_text_from_epub(file_bytes)
                max_chunk_size = 8000
                chunks = [clean_text(raw[i:i+max_chunk_size]) for i in range(0, len(raw), max_chunk_size)]
            else:
                continue

            doc_id = filename.replace(" ", "_").rsplit(".", 1)[0]
            valid_docs, valid_metadatas, valid_ids = [], [], []

            for i, chunk in enumerate(chunks):
                tokens = len(chunk) // 4
                if tokens < 20 or tokens > 4000:
                    continue

                meta = {
                    "source": filename,
                    "chunk": i,
                    "chunk_id": i,
                    "path": file_path
                }

                cid = f"{doc_id}_{i}"
                valid_docs.append(chunk)
                valid_metadatas.append(meta)
                valid_ids.append(cid)

            if valid_docs:
                for i in range(0, len(valid_docs), 20):
                    collection.add(
                        documents=valid_docs[i:i+20],
                        metadatas=valid_metadatas[i:i+20],
                        ids=valid_ids[i:i+20]
                    )

                previously_ingested.add(file_path)
                with open(STATE_FILE, "w") as f:
                    json.dump(list(previously_ingested), f, indent=2)

                ingested += 1

        except Exception as e:
            print(f"❌ Failed on {file_path}: {e}")

    print(f"✅ Done. {ingested} files ingested, {skipped} skipped.")

# --- Run if called directly ---
if __name__ == "__main__":
    ingest_supabase_docs(collection)
