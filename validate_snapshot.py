import os
import json
from dotenv import load_dotenv
from supabase import create_client

# Load .env
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
bucket = os.environ.get("SUPABASE_BUCKET")
client = create_client(url, key)

# Load local ingested state
try:
    with open("ingested_files.json", "r") as f:
        local_ingested = set(json.load(f).keys())
except FileNotFoundError:
    local_ingested = set()

# List Supabase files
response = client.storage.from_(bucket).list("pdfs", {"limit": 9999})
remote_files = set(f["name"] for f in response)

# Detect untracked files
missing = remote_files - local_ingested

if missing:
    print("❌ These Supabase files have NOT been ingested:")
    for f in sorted(missing):
        print(f" - {f}")
else:
    print("✅ All Supabase files have been ingested.")

