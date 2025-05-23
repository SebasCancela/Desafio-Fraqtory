import json
import os
import random

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

LABEL_STUDIO_JSON = "json_files/copied_annotations.json"
OUTPUT_DIR = "./model/data"
TRAIN_JSON = os.path.join(OUTPUT_DIR, "data_train.json")
VAL_JSON = os.path.join(OUTPUT_DIR, "data_val.json")
VAL_SPLIT = 0.2

def convert_labelstudio(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
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
                print(f"[WARNING] Missing label '{label}' in {new_image_id}. Filling with [0, 0].")
                keypoints.append([0.0, 0.0])

        samples.append({
            "id": new_image_id, 
            "kps": keypoints
        })

    return samples

def save_json_split(samples, output_dir, val_ratio):
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    os.makedirs(output_dir, exist_ok=True)
    with open(TRAIN_JSON, 'w') as f:
        json.dump(train_samples, f, indent=2)
    with open(VAL_JSON, 'w') as f:
        json.dump(val_samples, f, indent=2)

    print(f"✅ Saved {len(train_samples)} training and {len(val_samples)} validation samples.")
    print(f"➡ {TRAIN_JSON}")
    print(f"➡ {VAL_JSON}")

samples = convert_labelstudio(LABEL_STUDIO_JSON)
save_json_split(samples, OUTPUT_DIR, VAL_SPLIT)
