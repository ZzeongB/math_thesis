import os
import random
import json
from PIL import Image
from tqdm import tqdm

# 경로 설정
COCO_PATH = "./data/COCO2017"  # COCO 루트 경로
INSTANCES_PATH = os.path.join(COCO_PATH, "annotations/instances_train2017.json")
CAPTIONS_PATH = os.path.join(COCO_PATH, "annotations/captions_train2017.json")
IMAGES_PATH = os.path.join(COCO_PATH, "train2017")
OUTPUT_DIR = "./data/COCO_3000"

# 샘플링 개수
SAMPLE_SIZE = 3000  # 원하는 개수로 조정

# 결과 저장 경로 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# annotation 로드
with open(INSTANCES_PATH, 'r') as f:
    instances = json.load(f)

with open(CAPTIONS_PATH, 'r') as f:
    captions = json.load(f)

# image_id → caption 매핑 생성
caption_dict = {}
for ann in captions['annotations']:
    caption_dict.setdefault(ann['image_id'], []).append(ann['caption'])

# 전체 이미지 리스트
all_images = instances['images']
sampled_images = random.sample(all_images, SAMPLE_SIZE)

# 샘플링 정보 저장
sampled_metadata = []

for img_info in tqdm(sampled_images):
    file_name = img_info['file_name']
    image_id = img_info['id']
    width, height = img_info['width'], img_info['height']

    image_path = os.path.join(IMAGES_PATH, file_name)
    img = Image.open(image_path).convert('RGB')

    # object annotations 가져오기
    objects = [ann for ann in instances['annotations'] if ann['image_id'] == image_id]

    # caption 가져오기
    img_captions = caption_dict.get(image_id, [])

    # 이미지 복사
    img.save(os.path.join(OUTPUT_DIR, "images", file_name))

    # 메타데이터 저장
    sampled_metadata.append({
        "image_id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "objects": objects,
        "captions": img_captions
    })

# 최종 메타데이터 저장
with open(os.path.join(OUTPUT_DIR, "sampled_metadata.json"), 'w') as f:
    json.dump(sampled_metadata, f, indent=2)

print("샘플링 완료!")
