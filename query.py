import os
import datetime
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient
from utils import format_response_with_citations

# Load .env and initialize OpenAI + Chroma
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection("cocktailgpt")

def ask(question):
    results = collection.query(
        query_texts=[question],
        n_results=5,
        include=["documents", "metadatas"]
    )

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Answer with citations from the context. If unknown, say so."},
            {"role": "user", "content": f"Context:\n{str(results)}\n\nQuestion: {question}"}
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content
    content = format_response_with_citations(content, results)

    with open("query_log.jsonl", "a") as log:
        log.write(f'{datetime.datetime.utcnow().isoformat()} {question}\n')

    return content

def ask_loop():
    print(f"🌐 Railway: {os.environ.get('RAILWAY_ENVIRONMENT') == 'true'} · SKIP_INGEST: {os.environ.get('SKIP_INGEST')}")
    print(f"🧮 Chroma collection count: {collection.count()}")
    while True:
        user_input = input("Ask CocktailGPT (or type 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        print(ask(user_input))

if __name__ == "__main__":
    ask_loop()
