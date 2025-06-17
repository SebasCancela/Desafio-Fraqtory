import json
import os
import random

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

LABEL_STUDIO_JSON = "json_files/copied_annotations.json"
OUTPUT_DIR = "./model/data"
TRAIN_JSON = os.path.join(OUTPUT_DIR, "data_train.json")
VAL_JSON = os.path.join(OUTPUT_DIR, "data_val.json")
TEST_JSON = os.path.join(OUTPUT_DIR, "data_test.json")

def convert_labelstudio(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for task in data:
        image_path = task['data']['image']
        image_filename = os.path.basename(image_path)
        image_base = os.path.splitext(image_filename)[0]

        new_image_id = f"{image_base}" 

        annotations = task['annotations'][0]['result']
        coords_by_label = {}

        for kp in annotations:
            label = kp['value']['keypointlabels'][0]
            x_pct = kp['value']['x']
            y_pct = kp['value']['y']
            orig_w = kp['original_width']
            orig_h = kp['original_height']

            x_abs = x_pct / 100.0 * orig_w
            y_abs = y_pct / 100.0 * orig_h

            coords_by_label[label] = [round(x_abs), round(y_abs)]

        keypoints = []
        for label in LABEL_ORDER:
            if label in coords_by_label:
                keypoints.append(coords_by_label[label])
            else:
                print(f"[WARNING] Missing label '{label}' in {new_image_id}. Filling with [0, 0].")
                keypoints.append([0.0, 0.0])

        samples.append({
            "id": new_image_id, 
            "kps": keypoints
        })

    return samples

def save_json_split(samples, output_dir):
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    os.makedirs(output_dir, exist_ok=True)

    with open(TRAIN_JSON, 'w') as f:
        json.dump(train_samples, f, indent=2)
    with open(VAL_JSON, 'w') as f:
        json.dump(val_samples, f, indent=2)
    with open(TEST_JSON, 'w') as f:
        json.dump(test_samples, f, indent=2)

    print(f"✅ Saved {len(train_samples)} training, {len(val_samples)} validation, and {len(test_samples)} test samples.")
    print(f"➡ {TRAIN_JSON}")
    print(f"➡ {VAL_JSON}")
    print(f"➡ {TEST_JSON}")

samples = convert_labelstudio(LABEL_STUDIO_JSON)
save_json_split(samples, OUTPUT_DIR)