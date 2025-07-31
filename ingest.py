import os
import openai
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from utils import extract_text_from_pdf, clean_text, chunk_text
from tqdm import tqdm
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialise ChromaDB vector database
client = PersistentClient(path="/tmp/chroma_store")

# Define collection (creates if not exists)
collection = client.get_or_create_collection(
    name="cocktail_docs",
    embedding_function=OpenAIEmbeddingFunction(api_key=openai.api_key)
)

# Ingest all PDFs from the /pdfs folder
pdf_folder = "./pdfs"

for fname in os.listdir(pdf_folder):
    if not fname.endswith(".pdf"):
        continue

    doc_id = fname.replace(".pdf", "")
    print(f"📄 Processing {fname}...")

    full_path = os.path.join(pdf_folder, fname)
    raw_text = extract_text_from_pdf(full_path)
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)

    for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding chunks from {fname}")):
        metadata = {
            "source": fname,
            "chunk": i
        }
        try:
            collection.add(
                documents=[chunk],
                metadatas=[metadata],
                ids=[f"{doc_id}_{i}"]
            )
        except Exception as e:
            print(f"❌ Failed to add chunk {i} from {fname}: {e}")

print("✅ All files processed and embedded.")

