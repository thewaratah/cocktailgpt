import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
import requests

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Setup OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Setup ChromaDB
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
client = PersistentClient(path="./embeddings")
collection = client.get_or_create_collection(
    name="cocktail_docs",
    embedding_function=embedding_function
)

def web_search_serpapi(query):
    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "num": 5
            },
            timeout=10
        )
        results = resp.json()
        snippets = []

        if "organic_results" in results:
            for result in results["organic_results"]:
                if "snippet" in result:
                    snippets.append(result["snippet"])
        return snippets
    except Exception as e:
        return [f"[Web search failed: {e}]"]

def ask(question):
    if not isinstance(question, str) or not question.strip():
        return "⚠️ Invalid question input."

    # --- Local context ---
    try:
        results = collection.query(query_texts=[question], n_results=5)
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
    except Exception as e:
        return f"❌ Local vector DB query error: {e}"

    print("🧠 DEBUG metadatas:", metadatas)

    context_blocks = []
    citations = []

    for i in range(len(docs)):
        context_blocks.append(docs[i])
        meta = metadatas[i]
        source = meta.get("source", "Unknown Source")
        chunk = meta.get("chunk_id", meta.get("chunk", "?"))
        citations.append(f"{source} (chunk {chunk})")

    local_context = "\n\n".join(context_blocks)
    citation_list = "\n".join(f"- {c}" for c in citations)

    # --- Web context ---
    web_snippets = web_search_serpapi(question)
    web_context = "\n\n".join(web_snippets)

    # --- Combined Prompt ---
    prompt = f"""
You are a doctoral-level expert in beverage and flavour science, supporting bartenders, chefs, and creators.
You will answer questions using both:
- Local expert materials (such as scientific and culinary PDF extracts), and
- Live information from web search snippets when available.

Your answer should be clearly structured, scientifically accurate, and practically useful. You MUST:
- Use UK English spelling
- Use metric units (°C, grams, mL, etc)
- Avoid imperial or US-style terminology
- Indicate when information comes from the web, if used

[Local context]:
{local_context}

[Web snippets]:
{web_context}

Question:
{question}

Answer:
"""

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    answer = response.choices[0].message.content.strip()

    return f"{answer}\n\n📚 Sources used:\n{citation_list}"

# CLI use only
if __name__ == "__main__":
    while True:
        q = input("\nAsk CocktailGPT (or type 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n" + ask(q))
        print("—" * 80)
