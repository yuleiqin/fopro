import numpy as np
import json


def extract_class_weight(pathlist, N_class):
    labels = [1. for _ in range(N_class)]
    with open(pathlist, "r") as f:
        lines = f.readlines()
        for line in lines:
            # 00052-of-00064.tfrecord@101027009 {"conf_score": 1.0, "label": 0, "sample_weight": 1.2033292383292387}
            img_path = line.strip().split(" {")[0]
            json_path = line.strip().replace(img_path, "")
            label_json = json.loads(json_path.strip())
            label_idx = int(label_json["label"])
            label_weight = float(label_json["sample_weight"])
            labels[label_idx] = label_weight
    return labels





