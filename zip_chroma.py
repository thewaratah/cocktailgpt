import zipfile
import os

def zip_chroma_store():
    source_dir = "/tmp/chroma_store"
    zip_path = "/tmp/chroma_store.zip"

    if not os.path.exists(source_dir):
        print("❌ Source Chroma directory not found.")
        return

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, source_dir)
                zf.write(full_path, arcname)

    print(f"✅ Zipped to {zip_path}")

if __name__ == "__main__":
    zip_chroma_store()
