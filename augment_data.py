import os
import json
import cv2
import random
import shutil
import albumentations as A

random.seed(42)

LABEL_ORDER = [
    "top_left_corner",
    "top_right_corner",
    "bottom_right_corner",
    "bottom_left_corner",
    "left_service_box_top",
    "right_service_box_top",
    "right_service_box_bottom",
    "left_service_box_bottom",
    "center_line_top",
    "center_line_bottom",
    "net_top_left",
    "net_top_right",
    "net_bottom_right",
    "net_bottom_left"
]

INPUT_JSON = "./json_files/copied_annotations.json"
INPUT_IMG_DIR = "./model/data/images"
OUTPUT_IMG_DIR = "./model/augmented_data/images"
OUTPUT_JSON = "./model/augmented_data/data_train.json"
OUTPUT_VAL_JSON = "./model/augmented_data/data_val.json"
VAL_SPLIT = 0.2
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Albumentations pipelines
augmentations = [
    A.Compose([A.VerticalFlip(p=1.0)], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
    A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
    A.Compose([A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(hue_shift_limit=20, p=1.0)], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
    A.Compose([A.HorizontalFlip(p=1.0), A.HueSaturationValue(hue_shift_limit=20, p=1.0)], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
]

# Load original annotations
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

augmented_samples = []

for task in data:
    image_path = task['data']['image']
    image_filename = os.path.basename(image_path)
    image_base = os.path.splitext(image_filename)[0]
    video_id = os.path.basename(task['data']['video_id'])
    new_image_id = f"{video_id}_{image_base}"

    annotations = task['annotations'][0]['result']
    coords_by_label = {}

    for kp in annotations:
        label = kp['value']['keypointlabels'][0]
        x_pct = kp['value']['x']
        y_pct = kp['value']['y']
        new_w = 1280
        new_h = 720
            
        # Convert from percent to absolute pixel coordinates
        x_abs = x_pct / 100.0 * new_w
        y_abs = y_pct / 100.0 * new_h

        coords_by_label[label] = [round(x_abs), round(y_abs)]

    keypoints = []
    for label in LABEL_ORDER:
        if label in coords_by_label:
            keypoints.append(coords_by_label[label])
        else:
            keypoints.append([0.0, 0.0])

    # Load image
    image_path_full = os.path.join(INPUT_IMG_DIR, f"{new_image_id}.jpg")
    image = cv2.imread(image_path_full)
    if image is None:
        print(f"❌ Couldn't load image {image_path_full}")
        continue

    h, w = image.shape[:2]

    # Save original
    shutil.copy2(image_path_full, os.path.join(OUTPUT_IMG_DIR, f"{new_image_id}.jpg"))
    augmented_samples.append({"id": new_image_id, "kps": [[round(x), round(y)] for x, y in keypoints]})

    # Apply one random augmentation
    aug = random.choice(augmentations)
    aug_result = aug(image=image, keypoints=keypoints)
    aug_img = aug_result['image']
    aug_kps = [[round(x), round(y)] for (x, y) in aug_result['keypoints']]

    aug_img_name = f"{new_image_id}_aug.jpg"
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, aug_img_name), aug_img)

    augmented_samples.append({"id": aug_img_name[:-4], "kps": aug_kps})

# Split train and validation
random.shuffle(augmented_samples)
split_idx = int(len(augmented_samples) * (1 - VAL_SPLIT))
train_samples = augmented_samples[:split_idx]
val_samples = augmented_samples[split_idx:]

# Save JSONs
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(train_samples, f, indent=2)

with open(OUTPUT_VAL_JSON, "w", encoding="utf-8") as f:
    json.dump(val_samples, f, indent=2)

print(f"✅ Augmented dataset saved to {OUTPUT_JSON} and {OUTPUT_VAL_JSON}")
print(f"➡ Training samples: {len(train_samples)}")
print(f"➡ Validation samples: {len(val_samples)}")
