import os
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm

# ========= 配置部分 =========
label_dir = r"/cluster/scratch/pangyi/reto/data/patches_640_split/annotations/training_background"
num_classes = 8
suffixes = (".png", ".tif", ".tiff", ".jpg")
# ==========================

pixel_counter = Counter()
total_pixels = 0

label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(suffixes)]

for fname in tqdm(label_files, desc="Processing labels"):
    path = os.path.join(label_dir, fname)

    mask = np.array(Image.open(path))

    # 如果标签图不是单通道，报错提醒
    if mask.ndim != 2:
        raise ValueError(f"{fname} 不是单通道标签图，shape={mask.shape}")

    total_pixels += mask.size

    unique, counts = np.unique(mask, return_counts=True)
    for cls_id, cnt in zip(unique, counts):
        pixel_counter[int(cls_id)] += int(cnt)

print("\n每个类别的像素统计结果：")
print(f"总像素数: {total_pixels}\n")

class_names = [
    "background",
    "river",
    "forest",
    "lake",
    "wetland",
    "stream",
    "building",
    "road"
]

for cls_id in range(num_classes):
    cls_pixels = pixel_counter[cls_id]
    ratio = cls_pixels / total_pixels if total_pixels > 0 else 0
    print(f"{class_names[cls_id]}: {cls_pixels} pixels, ratio = {ratio:.6f} ({ratio*100:.2f}%)")


# 总像素数: 222822400

# background: 132887865 pixels, ratio = 0.614457 (61.45%)
# river: 3540009 pixels, ratio = 0.016369 (1.64%)
# forest: 65986514 pixels, ratio = 0.305113 (30.51%)
# lake: 918376 pixels, ratio = 0.004246 (0.42%)
# wetland: 228737 pixels, ratio = 0.001058 (0.11%)
# stream: 1005017 pixels, ratio = 0.004647 (0.46%)
# building: 1438252 pixels, ratio = 0.006650 (0.67%)
# road: 10264030 pixels, ratio = 0.047460 (4.75%)