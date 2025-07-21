import os
import io
import json
import requests
import fitz
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from supabase import create_client, Client

# --- Setup ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "cocktailgpt-pdfs")

os.environ["CHROMA_OPENAI_API_KEY"] = OPENAI_API_KEY
client = PersistentClient(path="./embeddings")
embedding_function = OpenAIEmbeddingFunction()
collection = client.get_or_create_collection(
    name="cocktail_docs",
    embedding_function=embedding_function
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- State ---
STATE_FILE = "reattached_metadata.json"
try:
    with open(STATE_FILE, "r") as f:
        already_patched = set(json.load(f))
except:
    already_patched = set()

# --- Helpers ---
def fetch_file_bytes(url):
    res = requests.get(url)
    res.raise_for_status()
    return BytesIO(res.content)

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
    files = []
    limit = 100
    offset = 0
    while True:
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

# --- Main Patch Function ---
def reattach_metadata():
    print("🔧 Reattaching missing metadata...")
    files = list_all_files(SUPABASE_BUCKET)
    patched = 0
    skipped = 0

    for file_path in tqdm(files):
        if file_path in already_patched:
            skipped += 1
            continue

        filename = file_path.split("/")[-1]
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{file_path}"

        try:
            file_bytes = fetch_file_bytes(url)
            if filename.endswith(".pdf"):
                raw = extract_text_from_pdf(file_bytes)
            elif filename.endswith(".csv"):
                raw = extract_text_from_csv(file_bytes)
            else:
                continue

            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned)
            doc_id = filename.replace(".pdf", "").replace(".csv", "").replace(" ", "_")

            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) == 0 or len(chunk) > 16000:
                    continue
                ids.append(f"{doc_id}_{i}")
                metadatas.append({
                    "source": filename,
                    "chunk": i,
                    "chunk_id": i,
                    "path": file_path
                })

            if ids:
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    collection.add(
                        ids=ids[i:i+batch_size],
                        documents=chunks[i:i+batch_size],  # ✅ required to avoid error
                        metadatas=metadatas[i:i+batch_size]
                    )
                already_patched.add(file_path)
                with open(STATE_FILE, "w") as f:
                    json.dump(list(already_patched), f, indent=2)
                patched += 1

        except Exception as e:
            print(f"❌ Failed on {file_path}: {e}")

    print(f"✅ Done. {patched} files patched. {skipped} skipped.")

# --- Run ---
if __name__ == "__main__":
    reattach_metadata()
