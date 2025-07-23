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
from chromadb import EphemeralClient
from openai import OpenAI

# --- Load environment ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "cocktailgpt-pdfs")
STATE_FILE = "ingested_files.json"

# --- Clients ---
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
client = EphemeralClient()
collection = client.get_or_create_collection("cocktail_docs", embedding_function=embedding_function)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- State tracking ---
try:
    with open(STATE_FILE, "r") as f:
        previously_ingested = set(json.load(f))
except:
    previously_ingested = set()

# --- Tag normalisation ---
TAG_SYNONYM_MAP = {
    "citrusy": "citrus", "green apple": "green-fruit", "herbal": "herbaceous",
    "savoury": "umami", "floral notes": "floral", "fruity": "fruit",
    "earthy": "earth", "mushroomy": "earth", "meaty": "umami"
}

def normalise_tags(tag_list):
    return list({TAG_SYNONYM_MAP.get(tag.lower().strip(), tag.lower().strip()) for tag in tag_list})

def generate_tags(chunk_text):
    prompt = (
        "You are a semantic tagging assistant for a cocktail R&D knowledge base. "
        "Given a chunk of cocktail text, return structured tags. "
        "Use JSON format with optional fields: technique, flavour, ingredient, category, process, skill_level, discipline. "
        "Only include fields that are relevant."
    )
    try:
        res = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": chunk_text[:1000]}],
            temperature=0.2
        )
        content = res.choices[0].message.content.strip()

        # Remove markdown code block markers
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        elif content.startswith("```"):
            content = content.lstrip("```").rstrip("```").strip()

        try:
            tags = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Tagging returned non-JSON response:\n{content}")
            return {}

        for key in tags:
            if isinstance(tags[key], list):
                tags[key] = normalise_tags(tags[key])
        return tags
    except Exception as e:
        print(f"⚠️ Tagging failed (no response): {e}")
        return {}

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
    chunks, i = [], 0
    while i < len(df):
        for rows in range(max_rows, min_rows - 1, -1):
            sub_df = df.iloc[i:i+rows]
            text = sub_df.to_string(index=False)
            if token_count(text) <= max_tokens:
                chunks.append(text)
                i += rows
                break
        else:
            i += 1
    return chunks

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

    try:
        with open("tags_by_chunk.json", "r") as tf:
            all_tags = json.load(tf)
    except:
        all_tags = {}

    for file_path in tqdm(files):
        if file_path in previously_ingested:
            skipped += 1
            continue

        filename = file_path.split("/")[-1]
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{file_path}"

        try:
            file_bytes = fetch_file_bytes(url)
            if filename.endswith(".pdf"):
                raw = extract_text_from_pdf(file_bytes)
                chunks = [clean_text(raw[i:i+16000]) for i in range(0, len(raw), 16000)]
            elif filename.endswith(".csv"):
                chunks = extract_text_from_csv(file_bytes)
            elif filename.endswith(".epub"):
                raw = extract_text_from_epub(file_bytes)
                chunks = [clean_text(raw[i:i+16000]) for i in range(0, len(raw), 16000)]
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
                tags = generate_tags(chunk)
                for key, val in tags.items():
                    if isinstance(val, list):
                        val = [v for v in val if v is not None]
                        if val:
                            meta[key] = ", ".join(str(v) for v in val)
                    elif val is not None:
                        meta[key] = str(val)

                cid = f"{doc_id}_{i}"
                valid_docs.append(chunk)
                valid_metadatas.append(meta)
                valid_ids.append(cid)
                all_tags[cid] = {k: v for k, v in meta.items() if k not in ["source", "chunk", "chunk_id", "path"]}

            if valid_docs:
                for i in range(0, len(valid_docs), 100):
                    collection.add(
                        documents=valid_docs[i:i+100],
                        metadatas=valid_metadatas[i:i+100],
                        ids=valid_ids[i:i+100]
                    )

                previously_ingested.add(file_path)
                with open(STATE_FILE, "w") as f:
                    json.dump(list(previously_ingested), f, indent=2)

                with open("tags_by_chunk.json", "w") as tf:
                    json.dump(all_tags, tf, indent=2)

                ingested += 1

        except Exception as e:
            print(f"❌ Failed on {file_path}: {e}")

    print(f"✅ Done. {ingested} files ingested, {skipped} skipped.")

# --- Run if called directly ---
if __name__ == "__main__":
    ingest_supabase_docs(collection)
