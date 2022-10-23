import os
import random
import shutil
import io
import struct
import cv2
import numpy as np
import warnings
import json
from tqdm import tqdm
import csv
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
from multiprocessing import Pool, Queue
import multiprocessing as mp
from functools import partial
import sys
sys.path.append('../')
from DataLoader.example_pb2 import Example


"""this script is to clean image pathlist by defined basic rules
"""


def split_by_num_process(items_to_split, N_threads):
    """splits the items into N subsequences evenly"""
    num_each_split = len(items_to_split) // N_threads
    # remain_split = len(items_to_split) % N_threads
    nums_split = [[] for _ in range(N_threads)]
    for idx, item_to_split in enumerate(items_to_split):
        if num_each_split != 0:
            idx_split = idx // num_each_split
            if idx_split >= N_threads:
                idx_split = idx % N_threads
        else:
            idx_split = idx % N_threads
        nums_split[idx_split].append(item_to_split)
    nums_split = [num_split for num_split in nums_split if len(num_split) > 0]
    return nums_split


def filter_basic_rule(img_path="", img_obj=None):
    """use basic rules to filter images
    1) the size of the images must be over 64 x 64
    2) avoid pure images such as whole-white or whole-black images
    """
    if img_obj is None:
        try:
            img_obj = Image.open(img_path).convert("L")
        except:
            return -1
        if img_obj is None:
            return -1
    h = img_obj.height
    w = img_obj.width
    ratio = h / (w + 1.0)
    # remove images with extreme size and aspect ratio
    if min(h, w) < 32 or ratio > 5 or ratio < 0.2:
        return -2
    # remove pure color image
    # down-sample with nearest neighborling
    resize_size = min(64, h, w)
    img_obj = img_obj.resize((resize_size, resize_size))
    img = np.asarray(img_obj)
    img = img.flatten()
    # count the number of pixels from 0-255
    bin_count = np.bincount(img)
    if np.amax(bin_count) >= 0.8 * len(img):
        # most likely to be pure image
        return 0
    return 1


def filter_mp(idx, path_img_sublists, root_dir):
    """filter images by basic rules (multi-processing) parallelization"""
    img_keep = []
    img_filtered = []
    path_img_sublist = path_img_sublists[idx]
    for line_i in tqdm(path_img_sublist):
        path_i = line_i.strip().split(": ")[0]
        path_i = os.path.join(root_dir, path_i)
        flag_i = filter_basic_rule(path_i)
        if flag_i == 1:
            img_keep.append(line_i)
        else:
            img_filtered.append(line_i)      
    return img_keep, img_filtered


def clean_webFG496(root_dir, N_threads=64):
    """clean the image pathlist to remove broken, pure, small images
    image_path_list: pathlist of images with class_names in imageNet 1k/21k, endswith ".pathlist"
    return: 
    clean_path_list: path to the filtered image pathlist
    1) deleted image pathlist files list for broken, pure, small images
    2) csv file for deleted images
    """
    # root_dir = "/youtu_pedestrian_detection/yuleiqin/datasets/finegrained/WebFG496/web-bird"
    train_list_path = os.path.join(root_dir, "train-list.txt")
    assert(os.path.exists(train_list_path))
    with open(train_list_path, 'r') as f:
        lines_all = f.readlines()
    count_img_num = len(lines_all)
    # prepare number of splits for multi-thread processing
    path_img_sublists = split_by_num_process(lines_all, N_threads)
    path_img_sublists = [sublist for sublist in path_img_sublists if len(sublist) > 0]
    N_threads = min(N_threads, len(path_img_sublists))
    print("Use %d threads to process the entire %d image files"%(N_threads, count_img_num))
    train_list_keep_path = os.path.join(root_dir, "train-list-keep.txt")
    train_list_filter_path = os.path.join(root_dir, "train-list-filter.txt")
    # assert(not (os.path.exists(train_list_keep_path)))
    # assert(not (os.path.exists(train_list_filter_path)))
    # start processing
    processes = []
    img_keeps = []
    img_filters = []
    try:
        mp.set_start_method('spawn', force=True)
        print("Context MP spawned")
        print("Start multiprocessing")
        pool = mp.Pool(processes=N_threads)
        for i in range(N_threads):
            processes.append(pool.apply_async(filter_mp,\
                args=(i, path_img_sublists, root_dir)))
        pool.close()
        pool.join()
        for process in processes:
            img_keep_i, img_filter_i = process.get()
            img_keeps += img_keep_i
            img_filters += img_filter_i
        print("End multiprocessing")
    except RuntimeError:
        print("Context MP failed")
        print("Start processing one-by-one")
        img_keeps, img_filters = filter_mp(0, [lines_all], root_dir)
    print("Finish")
    # combine all image pathlist from mp
    with open(train_list_keep_path, "w") as f:
        for line in img_keeps:
            f.write(line)
    with open(train_list_filter_path, "w") as f:
        for line in img_filters:
            f.write(line)
    return


def get_tfrecord_image(record_file, offset):
    """read images from tfrecord"""
    def parser(feature_list):
        """get the image file and perform transformation
        feature_list: the dictionary to load features (images)
        """
        for key, feature in feature_list: 
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                image = Image.open(io.BytesIO(image_raw))
                image = image.convert('RGB')
                return image
        return
    with open(record_file, 'rb') as ifs:
        ifs.seek(offset)
        byte_len_crc = ifs.read(12)
        proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
        pb_data = ifs.read(proto_len)
        if len(pb_data) < proto_len:
            print("read pb_data err, proto_len:%s pb_data len:%s" % (proto_len, len(pb_data)))
            return
    example = Example()
    example.ParseFromString(pb_data)
    # keep key value in order
    feature = sorted(example.features.feature.items())
    image = parser(feature)
    return image


def filter_mp_webvision(idx, path_img_sublists, root_dir):
    """filter images by basic rules (multi-processing) parallelization"""
    img_keep = []
    img_filtered = []
    path_img_sublist = path_img_sublists[idx]
    num_all_lines = len(path_img_sublist)
    num_filter_lines = 0
    for line_i in tqdm(path_img_sublist):
        info_i = line_i.strip().split(" {")[0]
        path_i, offset_i = info_i.strip().split("@")
        path_i = os.path.join(root_dir, path_i)
        offset_i = int(offset_i)
        image = get_tfrecord_image(path_i, offset_i)
        flag_i = filter_basic_rule(img_path="", img_obj=image)
        if flag_i == 1:
            img_keep.append(line_i)
        else:
            num_filter_lines += 1
            img_filtered.append(line_i)
    print("filter/delete lines {}: {}/{}".format(line_i, num_filter_lines, num_all_lines))
    return img_keep, img_filtered


def clean_web_vision(root_dir, pathlist, N_threads=16):    
    train_list_keep_path = pathlist.replace(".txt", "_keep.txt")
    assert(train_list_keep_path != pathlist)
    train_list_filter_path = pathlist.replace(".txt", "_filter.txt")
    assert(train_list_filter_path != pathlist)
    train_list_path = pathlist
    assert(os.path.exists(pathlist))
    with open(train_list_path, 'r') as f:
        lines_all = f.readlines()
    count_img_num = len(lines_all)
    # prepare number of splits for multi-thread processing
    path_img_sublists = split_by_num_process(lines_all, N_threads)
    path_img_sublists = [sublist for sublist in path_img_sublists if len(sublist) > 0]
    N_threads = min(N_threads, len(path_img_sublists))
    print("total number of lines = {}".format(count_img_num))
    # start processing
    processes = []
    img_keeps = []
    img_filters = []
    try:
        mp.set_start_method('spawn', force=True)
        print("Context MP spawned")
        print("Start multiprocessing")
        pool = mp.Pool(processes=N_threads)
        for i in range(N_threads):
            processes.append(pool.apply_async(filter_mp_webvision,\
                args=(i, path_img_sublists, root_dir)))
        pool.close()
        pool.join()
        for process in processes:
            img_keep_i, img_filter_i = process.get()
            img_keeps += img_keep_i
            img_filters += img_filter_i
        print("End multiprocessing")
    except RuntimeError:
        print("Context MP failed")
        print("Start processing one-by-one")
        img_keeps, img_filters = filter_mp(0, [lines_all], root_dir)
    print("Finish")
    # combine all image pathlist from mp
    with open(train_list_keep_path, "w") as f:
        for line in img_keeps:
            f.write(line)
    with open(train_list_filter_path, "w") as f:
        for line in img_filters:
            f.write(line)
    return


def calc_statistics(img_pathlist_path):
    imgs_by_class = {}
    with open(img_pathlist_path, "r") as f:
        lines_all = f.readlines()
    for line in lines_all:
        info = line.strip().split(": ")
        img_class = int(info[1])

        # img_path = line.strip().split(" ")[0]
        # json_path = line.strip().replace(img_path + " ", "")
        # info = json.loads(json_path)
        # img_class = int(info["label"])

        if not (img_class in imgs_by_class):
            imgs_by_class[img_class] = []
        imgs_by_class[img_class].append(line)
    csv_path = os.path.join(os.path.dirname(img_pathlist_path),\
        "statistics_" + os.path.basename(img_pathlist_path).replace(".txt", ".csv"))
    # assert(not (os.path.exists(csv_path)))
    nums_classes = []
    with open(csv_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["image class id", "image number"])
        for class_idx in imgs_by_class.keys():
            nums_classes.append(len(imgs_by_class[class_idx]))
            csv_writer.writerow([class_idx, len(imgs_by_class[class_idx])])
    nums_classes = np.array(nums_classes)
    print("min {} max {} mean {} median {}".format(np.min(nums_classes),\
        np.max(nums_classes), np.mean(nums_classes), np.median(nums_classes)))
    return


def sample_badcases(img_pathlist_path, root_dir):
    # train/001.Black_footed_Albatross/001.Black_footed_Albatross_00001.jpg
    with open(img_pathlist_path, "r") as f:
        lines_all = f.readlines()
    lines_sample = random.sample(lines_all, min(len(lines_all),100))
    save_dir = os.path.join(os.path.dirname(img_pathlist_path), os.path.basename(img_pathlist_path).replace(".txt", "_imgs"))
    os.makedirs(save_dir, exist_ok=True)
    for line in tqdm(lines_sample):
        img_path = line.strip().split(": ")[0]
        img_path = os.path.join(root_dir, img_path)
        save_img_path = os.path.join(save_dir, os.path.basename(img_path))
        shutil.copyfile(img_path, save_img_path)
    return


def sample_badcases_webvision(img_pathlist_path, root_dir):
    with open(img_pathlist_path, "r") as f:
        lines_all = f.readlines()
    lines_sample = random.sample(lines_all, min(len(lines_all),100))
    save_dir = os.path.join(os.path.dirname(img_pathlist_path), os.path.basename(img_pathlist_path).replace(".txt", "_imgs"))
    os.makedirs(save_dir, exist_ok=True)
    for line_i in tqdm(lines_sample):
        info_i = line_i.strip().split(" {")[0]
        tf_i, offset_i = info_i.strip().split("@")
        path_i = os.path.join(root_dir, tf_i)
        offset_i = int(offset_i)
        image = get_tfrecord_image(path_i, offset_i)
        save_img_path = os.path.join(save_dir, "{}_offset_{}.jpg".format(tf_i, offset_i))
        image.save(save_img_path)
    return

