import os
import shutil
import zipfile
from tqdm import tqdm
import requests

def download_with_progress(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def download_and_extract(url, dest_folder):
    filename = url.split('/')[-1]
    zip_path = os.path.join(dest_folder, filename)

    # Download file with tqdm
    if not os.path.exists(zip_path):
        print(f"⬇️  Downloading {filename}...")
        download_with_progress(url, zip_path)
    else:
        print(f"✅ {filename} already exists, skipping download.")

    # Extract zip
    print(f"📦 Extracting {filename}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

    # Clean up zip
    os.remove(zip_path)
    print(f"🧹 Removed {filename}")

def main():
    data_dir = "data"

    # Remove and recreate data directory
    if os.path.exists(data_dir):
        print(f"⚠️  Removing existing '{data_dir}' directory...")
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    print(f"📁 Created new directory: {os.path.abspath(data_dir)}")

    # List of COCO URLs
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/zips/test2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ]

    for url in urls:
        download_and_extract(url, data_dir)

    print("✅ All COCO 2017 files downloaded and extracted successfully!")

if __name__ == "__main__":
    main()

