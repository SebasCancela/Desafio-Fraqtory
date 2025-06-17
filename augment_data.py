import os
import json
import cv2
import random
import shutil
import albumentations as A

random.seed(42)

LABEL_ORDER = [
    "left_service_box_top",
    "right_service_box_top",
    "left_service_box_bottom",
    "right_service_box_bottom",
    "net_top_left",
    "net_top_right",
    "net_bottom_left",
    "net_bottom_right"
]

INPUT_JSON = "./json_files/copied_annotations.json"
INPUT_IMG_DIR = "./model/data/images"
OUTPUT_IMG_DIR = "./model/augmented_data/images"
OUTPUT_JSON = "./model/augmented_data/data_train.json"
OUTPUT_VAL_JSON = "./model/augmented_data/data_val.json"
OUTPUT_TEST_JSON = "./model/augmented_data/data_test.json"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

augmentations = [
    A.Compose([A.HorizontalFlip(p=1.0)], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
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
    new_image_id = f"{image_base}"

    annotations = task['annotations'][0]['result']
    coords_by_label = {}

    for kp in annotations:
        label = kp['value']['keypointlabels'][0]
        x_pct = kp['value']['x']
        y_pct = kp['value']['y']
        w = 2688
        h = 1520
            
        # Convert from percent to absolute pixel coordinates
        x_abs = x_pct / 100.0 * w
        y_abs = y_pct / 100.0 * h

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

# Split into 70% train, 15% val, 15% test
random.shuffle(augmented_samples)
n = len(augmented_samples)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_samples = augmented_samples[:train_end]
val_samples = augmented_samples[train_end:val_end]
test_samples = augmented_samples[val_end:]

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(train_samples, f, indent=2)

with open(OUTPUT_VAL_JSON, "w", encoding="utf-8") as f:
    json.dump(val_samples, f, indent=2)

with open(OUTPUT_TEST_JSON, "w", encoding="utf-8") as f:
    json.dump(test_samples, f, indent=2)

print(f"✅ Augmented dataset saved.")
print(f"➡ Train: {len(train_samples)} samples -> {OUTPUT_JSON}")
print(f"➡ Val:   {len(val_samples)} samples -> {OUTPUT_VAL_JSON}")
print(f"➡ Test:  {len(test_samples)} samples -> {OUTPUT_TEST_JSON}")
