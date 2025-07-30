import os
import datetime
import json
from openai import OpenAIError
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SKIP_INGEST = os.environ.get("SKIP_INGEST", "1") == "1"
RAILWAY_ENVIRONMENT = os.environ.get("RAILWAY_ENVIRONMENT", "false") == "true"

# New Chroma client setup
settings = Settings(
    chroma_api_impl="chromadb.api.local.LocalAPI",
    persist_directory="/tmp/chroma_store",
    anonymized_telemetry=False
)

client = Client(settings)
collection = client.get_or_create_collection("cocktailgpt")

def ask(question: str, tags: dict[str, str] = None):
    import openai
    openai.api_key = OPENAI_API_KEY

    try:
        filters = {"where": tags} if tags else {}
        results = collection.query(
            query_texts=[question],
            n_results=5,
            **filters
        )
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
    except Exception as e:
        print(f"❌ Chroma query error: {e}")
        docs = []
        metadatas = []

    context_blocks = []
    for doc, meta in zip(docs, metadatas):
        context_blocks.append(f"{doc}\nSOURCE: {meta.get('source')} (chunk {meta.get('chunk')})")

    context = "\n\n---\n\n".join(context_blocks)
    prompt = f"""You are a flavour, fermentation, and food science assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()

    if metadatas:
        citation_block = "\n\n📚 Sources:\n"
        sources = [f"{m.get('source')} (chunk {m.get('chunk')})" for m in metadatas]
        citation_block += ", ".join(sources)
        answer += citation_block

    try:
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "question": question,
            "tags_used": tags,
            "chunks_used": metadatas,
            "response": answer
        }
        with open("query_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as log_error:
        print(f"⚠️ Logging failed: {log_error}")

    return answer
