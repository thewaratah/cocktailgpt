import os
import json
from openai import OpenAI
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from tqdm import tqdm

# Load your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup OpenAI client and embedding function
openai_client = OpenAI(api_key=openai_api_key)
embedding_function = OpenAIEmbeddingFunction(api_key=openai_api_key)

# Connect to your ChromaDB collection
client = PersistentClient(path="/tmp/chroma_store")
collection = client.get_or_create_collection(
    name="cocktail_docs",
    embedding_function=embedding_function
)

# Synonym map for tag normalisation
TAG_SYNONYM_MAP = {
    "citrusy": "citrus",
    "green apple": "green-fruit",
    "herbal": "herbaceous",
    "savoury": "umami",
    "floral notes": "floral",
    "fruity": "fruit",
    "earthy": "earth",
    "mushroomy": "earth",
    "meaty": "umami"
}

def normalise_tags(tag_list):
    normalised = []
    for tag in tag_list:
        base = tag.lower().strip()
        normalised.append(TAG_SYNONYM_MAP.get(base, base))
    return list(set(normalised))

def generate_tags_for_chunk(chunk_text):
    system_prompt = (
        "You are a semantic tagging assistant for a cocktail R&D knowledge base. "
        "Given a chunk of cocktail text, return structured tags. "
        "Use JSON format with optional fields: technique, flavour, ingredient, category, process, skill_level, discipline. "
        "Only include fields that are relevant."
    )
    user_prompt = f"Chunk:\n{chunk_text[:1000]}"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        tags = json.loads(response.choices[0].message.content.strip())
        for key in tags:
            if isinstance(tags[key], list):
                tags[key] = normalise_tags(tags[key])
        return tags
    except Exception as e:
        print(f"⚠️ Tagging failed: {e}")
        return {}

# Retrieve all documents and metadata from the collection
print("🔍 Retrieving all documents...")
results = collection.get(include=["documents", "metadatas"])
docs = results["documents"]
metas = results["metadatas"]
ids = results.get("ids", [f"chunk_{i}" for i in range(len(docs))])

print(f"🔁 Retagging {len(docs)} chunks...")

# Loop through each chunk and apply updated tags
for i in tqdm(range(len(docs))):
    doc = docs[i]
    meta = metas[i]
    chunk_id = ids[i]

    new_tags = generate_tags_for_chunk(doc)
    updated_metadata = {**meta, **new_tags}

    try:
        collection.update(
            ids=[chunk_id],
            metadatas=[updated_metadata]
        )
    except Exception as e:
        print(f"❌ Failed to update chunk {chunk_id}: {e}")

print("✅ Retagging complete.")

