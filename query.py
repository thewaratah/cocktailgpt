import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from collections import defaultdict
import datetime

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
IS_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT", "false").lower() == "true"
SKIP_INGEST = os.getenv("SKIP_INGEST", "0") == "1"

print(f"🌐 Railway: {IS_RAILWAY} · SKIP_INGEST: {SKIP_INGEST}")

# --- OpenAI client ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- ChromaDB client ---
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)

if IS_RAILWAY:
    from chromadb import EphemeralClient
    client = EphemeralClient()
else:
    from chromadb import PersistentClient
    client = PersistentClient(path="./chroma_store")

collection = client.get_or_create_collection(
    name="cocktail_docs",
    embedding_function=embedding_function
)

# --- Optional ingestion on startup ---
if not SKIP_INGEST:
    from ingest_supabase import ingest_supabase_docs
    ingest_supabase_docs(collection)

# --- Logging ---
def log_interaction(question, answer, source_summary):
    timestamp = datetime.datetime.utcnow().isoformat()
    entry = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "sources": source_summary
    }
    with open("query_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

# --- SerpAPI fallback ---
def serp_api_search(query):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        results = data.get("organic_results", [])
        return [
            f"{r['title']} - {r['link']}\n{r.get('snippet', '')}"
            for r in results[:3]
        ]
    except Exception as e:
        return [f"(Web search failed: {e})"]

# --- Main assistant function ---
def ask(question, message_history=None):
    if not isinstance(question, str) or not question.strip():
        return "⚠️ Invalid question input."

    try:
        print("🧮 Chroma collection count:", collection.count())

        results = collection.query(
            query_texts=[question],
            n_results=5,
            include=["documents", "metadatas"]
        )
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]

        if isinstance(metadatas[0], list):  # defensive flatten
            metadatas = metadatas[0]

    except Exception as e:
        print(f"❌ Chroma query error: {e}")
        docs, metadatas = [], []

    chroma_context = "\n\n".join(docs) if docs else ""
    web_context = ""

    if not chroma_context.strip() or len(chroma_context.split()) < 150:
        web_results = serp_api_search(question)
        web_context = "\n\n[Web Results]\n" + "\n\n".join(web_results) if web_results else ""

    context = (chroma_context + web_context).strip() or "[No relevant documents or web results found.]"
    print("🧾 Context used (preview):", context[:500])
    print("📎 Metadatas used:", metadatas)

    # --- Format citations ---
    citations_by_file = defaultdict(list)
    for meta in metadatas:
        filename = meta.get("source")
        chunk_id = meta.get("chunk_id")
        if filename and chunk_id is not None:
            citations_by_file[filename].append(chunk_id)

    if citations_by_file:
        formatted = []
        for fname, chunks in citations_by_file.items():
            chunk_str = ", ".join(str(c) for c in sorted(set(chunks)))
            formatted.append(f"- {fname} (chunks {chunk_str})")
        citation_block = "\n\n📚 Sources:\n" + "\n".join(formatted)
    else:
        citation_block = "\n\n📚 Sources:\n[No chunk citations found.]"

    # --- GPT call ---
    messages = [{"role": "system", "content": (
        "You are a doctoral-level expert in beverage and flavour science, supporting bartenders, chefs, and creators. "
        "Use the retrieved context below — from internal documents or, if needed, web results — to answer clearly, scientifically, and practically.\n\n"
        f"Context:\n{context}"
    )}]

    if message_history and isinstance(message_history, list):
        messages.extend(message_history)

    messages.append({"role": "user", "content": question})

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.3
    )

    answer = response.choices[0].message.content.strip()
    source_summary = citation_block.replace("📚 Sources:", "").strip()
    log_interaction(question, answer, source_summary)

    return answer + citation_block

# --- CLI test runner ---
if __name__ == "__main__":
    while True:
        q = input("\nAsk CocktailGPT (or type 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n" + ask(q))
        print("—" * 80)
