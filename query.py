import os
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import requests

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ChromaDB setup
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
client = PersistentClient(path="./embeddings")
collection = client.get_or_create_collection(
    name="cocktail_docs",
    embedding_function=embedding_function
)

# 🔎 SerpAPI helper
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

# 🧠 Ask function
def ask(question):
    if not isinstance(question, str) or not question.strip():
        return "⚠️ Invalid question input."

    try:
        results = collection.query(query_texts=[question], n_results=5)
    except Exception as e:
        return f"❌ Query error: {e}"

    docs = results['documents'][0]
    context = "\n\n".join(docs)

    prompt = f"""
You are a doctoral-level expert in beverage and flavour science, supporting bartenders, chefs, and creators. Use the context below — drawn from technical documents and training materials — to answer the 
question clearly, accurately, and with practical application. When useful, explain the science or give step-by-step recommendations.

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

    return response.choices[0].message.content.strip()

# CLI test
if __name__ == "__main__":
    while True:
        q = input("\nAsk CocktailGPT (or type 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n" + ask(q))
        print("—" * 80)
