import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "cocktailgpt-pdfs")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🔍 Listing contents of 'cocktailgpt-pdfs/pdfs'")
files = supabase.storage.from_(SUPABASE_BUCKET).list("pdfs")

for f in files:
    print("📄", f["name"])

