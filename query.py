import os
import datetime
from dotenv import load_dotenv
import openai
from chromadb import PersistentClient
from utils import format_response_with_citations

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = PersistentClient(path="/tmp/chroma_store")
collection = client.get_or_create_collection("cocktailgpt")

def ask(question):
    results = collection.query(
        query_texts=[question],
        n_results=5,
        include=["documents", "metadatas"]
    )

    answer = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Answer with citations from the context. If unknown, say so."},
            {"role": "user", "content": f"Context:\n{str(results)}\n\nQuestion: {question}"}
        ],
        temperature=0.2
    )

    response = answer["choices"][0]["message"]["content"]
    response = format_response_with_citations(response, results)
    
    with open("query_log.jsonl", "a") as log:
        log.write(f'{datetime.datetime.utcnow().isoformat()} {question}\n')
    
    return response
