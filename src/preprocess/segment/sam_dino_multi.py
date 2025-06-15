import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch
import multiprocessing as mp
import glob

# SAM 라이브러리 로드
from segment_anything import sam_model_registry, SamPredictor

# DINOv2 로드
import dinov2.models.vision_transformer as vits
from torchvision import transforms

# 경로 설정
COCO_METADATA = "./data/COCO_3000/sampled_metadata.json"
COCO_IMAGE_DIR = "./data/COCO_3000/images"
OUTPUT_CROP_DIR = "./data/object_crops_3000_sam"
OUTPUT_EMBEDDING_FILE = "./data/embeddings/object_embeddings_dino_3000.json"

os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

def process_items(sub_metadata, device_id, total_len):
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    from segment_anything import sam_model_registry, SamPredictor
    from torchvision import transforms
    import json

    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)

    sam_checkpoint = "data/pretrained/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2_vitb14.eval().to(device)

    dino_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    result = {}

    for item in tqdm(sub_metadata, desc=f"Worker {device_id}", position=device_id):
        image_id = item['image_id']
        file_name = item['file_name']
        image_path = os.path.join(COCO_IMAGE_DIR, file_name)

        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        for idx, obj in enumerate(item['objects']):
            bbox = obj['bbox']
            x, y, w, h = bbox
            input_box = np.array([x, y, x + w, y + h])

            masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
            mask = masks[0].astype(np.uint8)

            ys, xs = np.where(mask == 1)
            if len(xs) == 0 or len(ys) == 0:
                continue  # skip empty mask

            crop = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
            crop = crop[np.min(ys):np.max(ys)+1, np.min(xs):np.max(xs)+1]
            crop_pil = Image.fromarray(crop)

            crop_save_path = os.path.join(OUTPUT_CROP_DIR, f"{image_id}_{idx}.jpg")
            crop_pil.save(crop_save_path)

            crop_tensor = dino_transform(crop_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                features = dinov2_vitb14(crop_tensor)
                features /= features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy().squeeze()

            result[f"{image_id}_{idx}.jpg"] = features.tolist()

    with open(f"partial_embedding_worker_{device_id}.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    with open(COCO_METADATA, 'r') as f:
        metadata = json.load(f)

    num_workers = 2
    chunk_size = len(metadata) // num_workers
    processes = []

    for i in range(num_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_workers - 1 else len(metadata)
        sub_metadata = metadata[start:end]

        p = mp.Process(target=process_items, args=(sub_metadata, i, len(sub_metadata)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final_result = {}
    for fpath in glob.glob("partial_embedding_worker_*.json"):
        with open(fpath, "r") as f:
            part = json.load(f)
            final_result.update(part)

    with open(OUTPUT_EMBEDDING_FILE, 'w') as f:
        json.dump(final_result, f)
