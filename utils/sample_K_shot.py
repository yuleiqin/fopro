import os
import json
import numpy as np
import random
from tqdm import tqdm



def sample_K_shots(few_shot_path, ref_path, K=1):
    """
    few_shot_path: the path to the few-shot examples
    ignore_path:   the path to ignore the few-shots
    K:             to split to K shots per index
    """
    mapping_label2meta = {}
    img_pathlist_to_ignore = {}
    with open(ref_path, "r") as fr:
        for line in fr.readlines():
            img_path = line.strip().split(" ")[0]
            meta_str = line.strip().replace(img_path + " ", "")
            meta_json = json.loads(meta_str)
            class_label = int(meta_json["label"])
            if not (class_label in img_pathlist_to_ignore):
                img_pathlist_to_ignore[class_label] = set()
            img_pathlist_to_ignore[class_label].add(img_path)
            mapping_label2meta[class_label] = meta_str

    img_pathlist_all = {}
    with open(few_shot_path, "r") as fr:
        for line in fr.readlines():
            img_path = line.strip().split(" ")[0]
            meta_str = line.strip().replace(img_path + " ", "")
            meta_json = json.loads(meta_str)
            class_label = int(meta_json["label"])
            if not (class_label in img_pathlist_all):
                img_pathlist_all[class_label] = set()
            img_pathlist_all[class_label].add(img_path)      

    for save_index in tqdm(range(1, 5)):
        # 随机抽样4组试试
        save_path = ref_path.replace(".txt", "_{}.txt".format(save_index))
        assert(save_path != few_shot_path) and (save_path != ref_path)
        with open(save_path, "w") as fw:
            for class_label in tqdm(img_pathlist_all):
                img_pathlist_idx = img_pathlist_all[int(class_label)]
                assert(len(img_pathlist_idx) == 16)
                meta_info = mapping_label2meta[int(class_label)]
                img_pathlist_idx_sample = random.sample(list(img_pathlist_idx), K)
                for img_path in img_pathlist_idx_sample:
                    fw.write(img_path + " " + meta_info + "\n")
    return



# few_shot_path = "./SCC_baseline/imglists/imgnet_webvision1k/fewshot_16_shot.txt"
# ref_path = "./SCC_baseline/imglists/imgnet_webvision1k/fewshot_4_shot.txt"
# K = 4
# ref_path = "./SCC_baseline/imglists/imgnet_webvision1k/fewshot_2_shot.txt"
# K = 2
# ref_path = "./SCC_baseline/imglists/imgnet_webvision1k/fewshot_8_shot.txt"
# K = 8
# ref_path = "./SCC_baseline/imglists/imgnet_webvision1k/fewshot_1_shot.txt"
# K = 1
# sample_K_shots(few_shot_path, ref_path, K)

