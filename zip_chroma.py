import os
import shutil

def zip_chroma_store():
    chroma_dir = "/tmp/chroma_store"
    zip_base = "/tmp/chroma_store_part"
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(f"{chroma_dir} not found")

    part_size = 100 * 1024 * 1024  # 100MB
    temp_zip = "/tmp/chroma_store_full.zip"

    # Step 1: Zip the entire chroma_store dir
    shutil.make_archive(temp_zip.replace(".zip", ""), 'zip', chroma_dir)

    # Step 2: Split the zip into ~100MB parts
    with open(temp_zip, "rb") as f:
        i = 1
        while chunk := f.read(part_size):
            with open(f"{zip_base}{i}.zip", "wb") as part:
                part.write(chunk)
            i += 1

    os.remove(temp_zip)
    print(f"✅ Created {i - 1} chunk(s) at {zip_base}*.zip")
