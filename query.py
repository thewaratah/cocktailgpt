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

    # Step 1: Local context from Chroma
    try:
        results = collection.query(query_texts=[question], n_results=5)
    except Exception as e:
        return f"❌ Vector DB query failed: {e}"

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    print("🧠 DEBUG metadatas:", metadatas)

    context_blocks = []
    citations = []

    for i in range(len(docs)):
        doc = docs[i]
        meta = metadatas[i]
        context_blocks.append(doc)
        source = meta.get("source", "Unknown Source")
        chunk = meta.get("chunk_id", meta.get("chunk", "?"))
        citations.append(f"{source} (chunk {chunk})")

    # Step 2: Web search via SerpAPI
    search_results = serp_api_search(question)
    if search_results:
        context_blocks.append("\n\n--- Web Results ---\n" + "\n".join(search_results))

    # Step 3: Compile final prompt
    context = "\n\n".join(context_blocks)
    citation_list = "\n".join(f"- {c}" for c in citations) if citations else "None found."

    prompt = f"""
You are a doctoral-level expert in beverage and flavour science. Use the context below — drawn from documents and optionally web content — to answer the question clearly, practically, and with scientific reasoning.

Context:
{context}

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
