import os
import numpy as np


def sample_10_shots_K(few_shot_path):
    with open(few_shot_path, "r") as f:
        lines = f.readlines()
    one_shot_path = few_shot_path.replace("16_shot.txt", "1_shot.txt")
    assert(os.path.exists(one_shot_path))
    with open(one_shot_path, "r") as f:
        lines_one = f.readlines()
    visited = set()
    for line in lines_one:
        visited.add(line.strip())
    lines = [line.strip() for line in lines if not (line.strip() in visited)]
    lines_by_dict = {}
    for line in lines:
        info = line.split("@")
        class_name = info[1]
        if not class_name in lines_by_dict:
            lines_by_dict[class_name] = []
        lines_by_dict[class_name].append(line)
    for idx in range(15):
        save_fewshot_path_idx = os.path.join(os.path.dirname(few_shot_path), "fewshot_1_shot_{}.txt".format(idx + 1))
        assert(not (os.path.exists(save_fewshot_path_idx)))
        with open(save_fewshot_path_idx, "w") as f:
            for class_name in lines_by_dict.keys():
                f.write(lines_by_dict[class_name][idx] + "\n")
    return


