import os
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import requests
from collections import defaultdict

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ChromaDB setup
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
client = PersistentClient(path="./embeddings")
collection = client.get_or_create_collection(
    name="cocktail_docs",  # must match your ingestion
    embedding_function=embedding_function
)

# 🔎 SerpAPI fallback search
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

# 🧠 Core assistant function
def ask(question, message_history=None):
    if not isinstance(question, str) or not question.strip():
        return "⚠️ Invalid question input."

    # --- Step 1: Query Chroma for Supabase PDF chunks ---
    try:
        results = collection.query(
            query_texts=[question],
            n_results=5,
            include=["documents", "metadatas"]
        )
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
    except Exception as e:
        docs, metadatas = [], []

    chroma_context = "\n\n".join(docs) if docs else ""
    web_context = ""

    # --- Step 2: Only call SerpAPI if no internal chunks ---
    if not chroma_context.strip():
        web_results = serp_api_search(question)
        web_context = "\n\n".join(web_results) if web_results else ""

    # --- Step 3: Combine into context block ---
    if chroma_context:
        context = chroma_context
    elif web_context:
        context = f"[Web Results]\n{web_context}"
    else:
        context = "[No relevant documents or web results found.]"

    # --- Step 4: Build Supabase-only citations ---
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
        citation_block = ""

    # --- Step 5: Build message thread for OpenAI ---
    messages = []

    system_prompt = (
        "You are a doctoral-level expert in beverage and flavour science, supporting bartenders, chefs, and creators. "
        "Use the retrieved context below — from internal documents or, if needed, web results — to answer clearly, scientifically, and practically.\n\n"
        f"Context:\n{context}"
    )
    messages.append({"role": "system", "content": system_prompt})

    if message_history and isinstance(message_history, list):
        messages.extend(message_history)

    messages.append({"role": "user", "content": question})

    # --- Step 6: Ask GPT ---
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.3
    )

    answer = response.choices[0].message.content.strip()

    return answer + citation_block
