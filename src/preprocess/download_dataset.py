import os
import requests
from tqdm import tqdm

# 다운로드 URL 목록
DOWNLOAD_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# 다운로드 받을 경로
DOWNLOAD_DIR = "./data/COCO2017"

# 압축 해제 여부
UNZIP_AFTER_DOWNLOAD = True

# 디렉토리 생성
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 다운로드 함수
def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

# 다운로드 실행
for name, url in DOWNLOAD_URLS.items():
    filename = os.path.basename(url)
    dest_path = os.path.join(DOWNLOAD_DIR, filename)
    
    if not os.path.exists(dest_path):
        print(f"Downloading {name}...")
        download_file(url, dest_path)
    else:
        print(f"{filename} already exists, skipping.")

# 압축 해제
if UNZIP_AFTER_DOWNLOAD:
    import zipfile
    for name, url in DOWNLOAD_URLS.items():
        filename = os.path.basename(url)
        dest_path = os.path.join(DOWNLOAD_DIR, filename)
        extract_dir = DOWNLOAD_DIR
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

print("All done!")
