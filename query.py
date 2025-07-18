import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialise vector DB
embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
client = PersistentClient(path="./embeddings")

collection = client.get_or_create_collection(
    name="cocktail_docs",
    embedding_function=embedding_function
)

def ask(question):
    # Query vector DB for relevant chunks
    results = collection.query(query_texts=[question], n_results=5)
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]

    context_blocks = []
    citations = []

    for i in range(len(docs)):
        context_blocks.append(docs[i])
        meta = metadatas[i]
        citations.append(f"{meta['source']} (chunk {meta['chunk']})")

    context = "\n\n".join(context_blocks)
    citation_list = "\n".join(f"- {c}" for c in citations)

    # GPT-4 Turbo prompt
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

    answer = response.choices[0].message.content.strip()

    return f"{answer}\n\n📚 Sources used:\n{citation_list}"

# Interactive CLI — only runs when called directly
if __name__ == "__main__":
    while True:
        q = input("\nAsk CocktailGPT (or type 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n" + ask(q))
        print("—" * 80)
