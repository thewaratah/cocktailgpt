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

# 🧠 Main assistant function
def ask(question, message_history=None):
    if not isinstance(question, str) or not question.strip():
        return "⚠️ Invalid question input."

    try:
        results = collection.query(
            query_texts=[question],
            n_results=5,
            include=["documents", "metadatas"]
        )
    except Exception as e:
        return f"❌ Query error: {e}"

    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    context = "\n\n".join(docs)

    # Build Supabase-only citation list
    supabase_citations = []
    seen = set()
    for meta in metadatas:
        source = meta.get("source") or meta.get("filename")
        if source and source not in seen:
            supabase_citations.append(f"- {source}")
            seen.add(source)

    # Construct chat history for OpenAI
    messages = []

    # System prompt with context
    system_prompt = (
        "You are a doctoral-level expert in beverage and flavour science, supporting bartenders, chefs, and creators. "
        "Use only the retrieved context below to answer. Be practical, scientific, and accurate.\n\n"
        f"Context:\n{context}"
    )
    messages.append({"role": "system", "content": system_prompt})

    # Prior chat history
    if message_history and isinstance(message_history, list):
        messages.extend(message_history)

    # Final question
    messages.append({"role": "user", "content": question})

    # Get response
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.3
    )

    answer = response.choices[0].message.content.strip()

    if supabase_citations:
        citation_block = "\n\n📚 Sources:\n" + "\n".join(supabase_citations)
        return f"{answer}{citation_block}"
    else:
        return answer
