import zipfile
import os

SOURCE = "/tmp/chroma_store"
ZIP_TARGET = "/tmp/chroma_store.zip"

def zip_dir(source_dir, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, start=source_dir)
                zf.write(filepath, arcname)

if __name__ == "__main__":
    if os.path.exists(SOURCE):
        zip_dir(SOURCE, ZIP_TARGET)
        print(f"✅ Zipped to {ZIP_TARGET}")
    else:
        print("❌ No vectorstore found.")

