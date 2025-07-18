import os
import requests
import fitz  # PyMuPDF
import openai
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "cocktailgpt-pdfs")

# Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Connect to Chroma vector DB
client = PersistentClient(path="./embeddings")
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key)
collection = client.get_or_create_collection(
    name="cocktail_docs",
    embedding_function=embedding_function
)

# --- Helpers ---

def fetch_pdf(url):
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

def clean_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def chunk_text(text, max_tokens=500):
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

# --- Main Ingestion Function ---

def ingest_supabase_pdfs():
    print("🔍 Fetching file list from Supabase...")

    files = [
        f for f in supabase.storage.from_(SUPABASE_BUCKET).list()
        if f["name"].endswith(".pdf")
    ]

    for f in tqdm(files):
        filename = f["name"]
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
        try:
            print(f"📄 Processing: {filename}")
            pdf_bytes = fetch_pdf(url)
            raw = extract_text_from_pdf(pdf_bytes)
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned)

            doc_id = filename.replace(".pdf", "").replace(" ", "_")
            for i, chunk in enumerate(chunks):
                metadata = {"source": filename, "chunk": i}
                collection.add(
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[f"{doc_id}_{i}"]
                )
        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")

    print("✅ Ingestion complete.")

# --- Trigger on script run ---

if __name__ == "__main__":
    ingest_supabase_pdfs()

