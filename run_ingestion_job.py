from ingest_supabase import ingest_supabase_docs
from query import collection

if __name__ == "__main__":
    print("🚀 Starting ingestion...")
    ingest_supabase_docs(collection)
    print(f"✅ Ingestion complete. Total chunks: {collection.count()}")

