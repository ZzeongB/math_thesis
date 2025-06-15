import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch

# SAM 라이브러리 로드
from segment_anything import sam_model_registry, SamPredictor

# DINOv2 로드
import dinov2.models.vision_transformer as vits
from torchvision import transforms

# # 경로 설정
# COCO_METADATA = "./COCO_500/sampled_metadata.json"
# COCO_IMAGE_DIR = "./COCO_500/images"
# OUTPUT_CROP_DIR = "./object_crops_sam"
# OUTPUT_EMBEDDING_FILE = "./object_embeddings_dino.json"

# 기존 sam_dino.py 구조 그대로 사용
COCO_METADATA = "./data/COCO_3000/sampled_metadata.json"
COCO_IMAGE_DIR = "./data/COCO_3000/images"
OUTPUT_CROP_DIR = "./data/object_crops_3000_sam"
OUTPUT_EMBEDDING_FILE = "./data/embeddings/object_embeddings_dino_3000.json"

os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

# SAM 모델 준비
sam_checkpoint = "data/pretrained/sam_vit_b_01ec64.pth"  # 사전 다운로드 필요
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# # DINOv2 모델 준비
# dino_model = vits.vitb14()
# dino_model.eval().to(device)
import torch

dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitb14.eval().cuda()


dino_transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# COCO metadata 로드
with open(COCO_METADATA, 'r') as f:
    metadata = json.load(f)

# 전체 파이프라인 수행
object_embeddings = {}

for item in tqdm(metadata):
    image_id = item['image_id']
    file_name = item['file_name']
    image_path = os.path.join(COCO_IMAGE_DIR, file_name)
    
    # 이미지 로드 (BGR → RGB)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    for idx, obj in enumerate(item['objects']):
        bbox = obj['bbox']  # [x, y, w, h]
        x, y, w, h = bbox
        input_box = np.array([x, y, x + w, y + h])

        masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
        mask = masks[0].astype(np.uint8)

        # Masked crop 추출
        masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        ys, xs = np.where(mask == 1)
        crop = masked[np.min(ys):np.max(ys)+1, np.min(xs):np.max(xs)+1]
        crop_pil = Image.fromarray(crop)

        # Crop 저장 (선택사항)
        crop_save_path = os.path.join(OUTPUT_CROP_DIR, f"{image_id}_{idx}.jpg")
        crop_pil.save(crop_save_path)

        # DINOv2 embedding 추출
        crop_tensor = dino_transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            features = dinov2_vitb14(crop_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy().squeeze()

        object_embeddings[f"{image_id}_{idx}.jpg"] = features.tolist()

# 결과 저장
with open(OUTPUT_EMBEDDING_FILE, 'w') as f:
    json.dump(object_embeddings, f)

print(f"DINOv2 object embeddings saved to {OUTPUT_EMBEDDING_FILE}")
