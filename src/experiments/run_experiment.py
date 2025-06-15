import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# 경로 설정
OBJECT_EMB_PATH = "./data/embeddings/object_embeddings_dino_3000.json"
TEXT_EMB_PATH = "./data/embeddings/text_embeddings_3000.json"
METADATA_PATH = "./data/COCO_3000/sampled_metadata.json"
REPORT_DIR = "./reports/"
os.makedirs(REPORT_DIR, exist_ok=True)

# 데이터 로드
with open(OBJECT_EMB_PATH, 'r') as f:
    object_embeddings = json.load(f)
with open(TEXT_EMB_PATH, 'r') as f:
    text_embeddings = json.load(f)
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

# filename ↔ image_id 매핑
filename_to_imageid = {}
for item in metadata:
    image_id = item['image_id']
    for i, obj in enumerate(item['objects']):
        filename = f"{image_id}_{i}.jpg"
        filename_to_imageid[filename] = image_id

# 먼저 COCO category 매핑 로드
with open("./data/COCO2017/annotations/instances_train2017.json", 'r') as f:
    instances = json.load(f)

category_map = {cat['id']: cat['name'] for cat in instances['categories']}

# image_id ↔ category 매핑 (1st object 기준, COCO category name 사용)
imageid_to_category = {}
for item in metadata:
    image_id = item['image_id']
    if len(item['objects']) > 0:
        category_id = item['objects'][0]['category_id']
        category_name = category_map.get(category_id, "unknown")
        imageid_to_category[str(image_id)] = category_name

# paired data 추출
object_vecs, text_vecs, categories = [], [], []

for filename, obj_emb in tqdm(object_embeddings.items()):
    image_id = filename_to_imageid.get(filename)
    if image_id is None:
        continue
    text_emb = text_embeddings.get(str(image_id)) or text_embeddings.get(image_id)
    if text_emb is None:
        continue

    object_vecs.append(np.array(obj_emb).squeeze())
    text_vecs.append(np.array(text_emb).squeeze())

    category = imageid_to_category.get(str(image_id), "unknown")
    categories.append(category)

object_vecs = np.array(object_vecs)
text_vecs = np.array(text_vecs)
categories = np.array(categories)

print(f"Total paired samples: {len(object_vecs)}")

### STEP 1: PCA 차원축소
pca = PCA(n_components=512)
object_vecs_reduced = pca.fit_transform(object_vecs)

# PCA cosine similarity
cosine_sims = 1 - np.array([cosine(o, t) for o, t in zip(object_vecs_reduced, text_vecs)])
print(f"PCA reduced cosine: {np.mean(cosine_sims):.4f}")

### STEP 2: Linear Ridge Regression
ridge = Ridge(alpha=1e-3)
ridge.fit(object_vecs_reduced, text_vecs)
object_vecs_mapped = ridge.predict(object_vecs_reduced)

ridge_cosine_sims = 1 - np.array([cosine(o, t) for o, t in zip(object_vecs_mapped, text_vecs)])
ridge_mse = np.mean(np.sum((object_vecs_mapped - text_vecs)**2, axis=1))

print(f"Ridge cosine: {np.mean(ridge_cosine_sims):.4f}")
print(f"Ridge MSE: {ridge_mse:.4f}")

### STEP 3: Kernel Ridge Regression
krr = KernelRidge(alpha=1e-3, kernel='rbf', gamma=0.1)
krr.fit(object_vecs_reduced, text_vecs)
object_vecs_kernel = krr.predict(object_vecs_reduced)

kernel_cosine_sims = 1 - np.array([cosine(o, t) for o, t in zip(object_vecs_kernel, text_vecs)])
kernel_mse = np.mean(np.sum((object_vecs_kernel - text_vecs)**2, axis=1))

print(f"Kernel Ridge cosine: {np.mean(kernel_cosine_sims):.4f}")
print(f"Kernel Ridge MSE: {kernel_mse:.4f}")

### STEP 4: Category-wise 분석
category_results = defaultdict(lambda: {'ridge': [], 'kernel': []})

for idx, cat in enumerate(categories):
    category_results[cat]['ridge'].append(ridge_cosine_sims[idx])
    category_results[cat]['kernel'].append(kernel_cosine_sims[idx])

category_summary = []
for cat, scores in category_results.items():
    ridge_mean = np.mean(scores['ridge'])
    kernel_mean = np.mean(scores['kernel'])
    category_summary.append((cat, ridge_mean, kernel_mean))

category_summary.sort(key=lambda x: x[1], reverse=True)

### STEP 5: 분포 tail 분석
plt.figure(figsize=(8, 5))
plt.hist(cosine_sims, bins=50, alpha=0.5, label="PCA reduced")
plt.hist(ridge_cosine_sims, bins=50, alpha=0.5, label="Ridge")
plt.hist(kernel_cosine_sims, bins=50, alpha=0.5, label="Kernel Ridge")
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.title("Cross-dimensional Alignment")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "alignment_hist.png"))
plt.close()

### STEP 6: Report 저장
with open(os.path.join(REPORT_DIR, "alignment_report.txt"), "w") as f:
    f.write(f"Total samples: {len(object_vecs)}\n")
    f.write(f"PCA reduced cosine: {np.mean(cosine_sims):.4f}\n")
    f.write(f"Ridge cosine: {np.mean(ridge_cosine_sims):.4f}\n")
    f.write(f"Ridge MSE: {ridge_mse:.4f}\n")
    f.write(f"Kernel Ridge cosine: {np.mean(kernel_cosine_sims):.4f}\n")
    f.write(f"Kernel Ridge MSE: {kernel_mse:.4f}\n\n")
    
    f.write("Category-wise Results:\n")
    for cat, ridge_mean, kernel_mean in category_summary:
        f.write(f"{cat}: Ridge {ridge_mean:.3f}, Kernel {kernel_mean:.3f}\n")

print(f"Extended report saved at {REPORT_DIR}")
