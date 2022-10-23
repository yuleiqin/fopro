import json
import os
import csv

"""this script is to analyze the corrected image label
json files by the pretrained model"""

json_path = ""
webvision = True

with open(json_path, "r") as f:
    json_file = json.load(f)

samples = json_file['samples']
root_dirs = json_file["roots"]
index2roots = json_file["index2root"]
samples_full = []
for sample, root_dir in zip(samples, root_dirs):
    ## 增加根目录
    if webvision:
        tf_record, offset = sample
        tf_record_full = os.path.join(index2roots[str(root_dir)], tf_record)
        samples_full.append([tf_record_full, offset])
    else:
        img_name, label = sample
        img_name_full = os.path.join(index2roots[str(root_dir)], img_name)
        samples_full.append([img_name_full, label])
samples = samples_full
targets = json_file['targets']
domains = json_file['domains']

imgs_by_class = {}
for sample, target, domain in zip(samples, targets, domains):
    if not (target in imgs_by_class):
        imgs_by_class[target] = [[],[]]
    if int(domain) > 0:
        imgs_by_class[target][1].append(sample)
    else:
        imgs_by_class[target][0].append(sample)

# print(imgs_by_class.keys(), len(imgs_by_class.keys()))
csv_path = os.path.join(os.path.dirname(json_path), "corrected_json_analysis.csv")
with open(csv_path, "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["class", "class_name", "number of web images", "number of labeled images"])
    for class_i in imgs_by_class:
        imgs_list = imgs_by_class[class_i]
        if len(imgs_list[0]) == 0:
            print(class_i)
            img_name = "NULL"
        else:
            img_name = imgs_list[0][0][0]
        csv_writer.writerow([class_i, img_name, len(imgs_list[0]), len(imgs_list[1])])



