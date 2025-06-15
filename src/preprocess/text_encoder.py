import json
import clip
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# METADATA_PATH = "./COCO_500/sampled_metadata.json"
# OUTPUT_FILE = "./data/embeddings/text_embeddings.json"

METADATA_PATH = "data/COCO_3000/sampled_metadata.json"
OUTPUT_FILE = "./data/embeddings/text_embeddings_3000.json"

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

TEXT_EMBEDDINGS = {}

for item in tqdm(metadata):
    captions = item['captions']
    caption = captions[0]  # 첫 번째 캡션 사용 (다중 캡션 사용도 가능)

    text = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm()

    TEXT_EMBEDDINGS[item['image_id']] = text_features.cpu().numpy().tolist()[0]

# 저장
with open(OUTPUT_FILE, 'w') as f:
    json.dump(TEXT_EMBEDDINGS, f)

print("Text embeddings saved!")
