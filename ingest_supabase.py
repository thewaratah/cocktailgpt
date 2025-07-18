import os
import io
import requests
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "cocktailgpt-pdfs")

# Ensure Chroma sees this key
os.environ["CHROMA_OPENAI_API_KEY"] = OPENAI_API_KEY

# Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Connect to Chroma vector DB
client = PersistentClient(path="./embeddings")
embedding_function = OpenAIEmbeddingFunction()
collection = client.get_or_create_collection(
    name="cocktailgpt",
    embedding_function=embedding_function
)

# --- Helpers ---

def fetch_file_bytes(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}")
    return BytesIO(response.content)

def extract_text_from_pdf(pdf_bytes):
    text = ""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

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
    """Recursively list files in all folders."""
    files = []
    result = supabase.storage.from_(bucket_name).list(path)
    for item in result:
        if item["name"].endswith(".pdf") or item["name"].endswith(".csv"):
            full_path = f"{path}/{item['name']}" if path else item["name"]
            files.append(full_path)
        elif item.get("metadata") and item["metadata"].get("mimetype") == "application/vnd.folder":
            subfolder = f"{path}/{item['name']}" if path else item["name"]
            files.extend(list_all_files(bucket_name, subfolder))
    return files

# --- Main Ingestion Function ---

def ingest_supabase_docs():
    print("🔍 Fetching file list from Supabase (recursive)...")
    files = list_all_files(SUPABASE_BUCKET)

    if not files:
        print("⚠️ No matching files found.")
        return

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

            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) == 0 or len(chunk) > 16000:
                    print(f"⚠️ Skipping empty or oversized chunk: {filename} [chunk {i}]")
                    continue
                metadata = {
                    "source": filename,
                    "path": file_path,
                    "chunk_id": i
                }
                try:
                    collection.add(
                        documents=[chunk],
                        metadatas=[metadata],
                        ids=[f"{doc_id}_{i}"]
                    )
                except Exception as e:
                    print(f"❌ Failed to add chunk {i} from {filename}: {e}")
                    continue

        except Exception as e:
            print(f"❌ Failed on {file_path}: {e}")

    print("✅ Ingestion complete.")

# --- Run it ---

if __name__ == "__main__":
    ingest_supabase_docs()
