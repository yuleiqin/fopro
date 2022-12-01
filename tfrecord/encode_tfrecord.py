"""
https://github.com/tensorflow/models/blob/master/research/slim/datasets/build_imagenet_data.py
"""
import tensorflow as tf
from tensorflow.train import Example, Features, Feature, FloatList, BytesList, Int64List
import numpy as np
import threading
import argparse
import os
import sys
from io import BytesIO
from vlkit_bytes import array2bytes, bytes2array
from glob import glob
from utils import write_json


def data2example(anno, config):
    """reads all images with raw byte format and transforms into tfrecords"""
    filename, wdnet_id = anno.split(" ")
    img_path = config['img_root'] + filename 
    try:
        image_raw = open(img_path, mode='rb').read()        
    except:
        print("Warning: FileNotExists", filename)
        return None
    example = Example(
    features=Features(feature={
            'filename': Feature(bytes_list=BytesList(value=[filename.encode('utf-8'),])),
            'image': Feature(bytes_list=BytesList(value=[image_raw,])),
            'label_wdnet_id': Feature(bytes_list=BytesList(value=[array2bytes(wdnet_id),])),
        })
    )
    return example


def batch_process(thread_index, output, ranges, filenames, shards, config):
    """pack tf-record data in parallel"""
    os.makedirs(output, exist_ok=True)
    num_threads = len(ranges)
    assert not shards % num_threads, "%d %d" % (shards, num_threads)
    shards_per_batch = int(shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             shards_per_batch + 1).astype(int)
    samples_per_thres = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(shards_per_batch):
        shard = thread_index * shards_per_batch + s
        tfrecord_filename = os.path.join(output, "%.5d-of-%.5d.tfrecord"% (shard, shards))
        index_filename = os.path.join(output, "%.5d-of-%.5d.index"% (shard, shards))
        writer = tf.io.TFRecordWriter(tfrecord_filename)
        index = open(index_filename, 'w')
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        shard_offset = 0
        for i in files_in_shard:
            filename = filenames[i]
            example = data2example(filename, config)
            if type(example) == type(None):
                continue
            example_bytes = example.SerializeToString()
            writer.write(example_bytes)
            writer.flush()
            path_i, wdnet_id = filename.split(" ")
            index.write('{} {} {} {}\n'.format(path_i, wdnet_id, "%.5d-of-%.5d.tfrecord"% (shard, shards), shard_offset))
            shard_offset += (len(example_bytes) + 16)
            shard_counter += 1
            counter += 1
            if i % 100 == 0:
                print("thread%.3d-shard%.3d %d/%d" % (thread_index, shard, i, len(filenames)))
        writer.close()
        index.close()
    return


def run_tfrecord_encoding(output_path, train_list, N_threads=64):
    """this function encodes all tf-record files"""
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    config = dict()
    config['img_root'] = ''
    config['shards'] = 64
    config['threads'] = N_threads
    config['output'] = output_path
    assert config['shards'] % config['threads'] == 0

    filenames = [i.strip() for i in open(train_list, 'r').readlines()]
    spacing = np.linspace(0, len(filenames), config['threads'] + 1).astype(np.int)

    """
    batch process samples with multi-thread.
    each thread process several shards (each shard corresponds to a tfrecord file).
    """
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    threads = []
    for thread_index in range(len(ranges)):
        arg = (thread_index, config['output'], ranges, filenames, config['shards'], config)
        thread = threading.Thread(target=batch_process, args=arg)
        # thread.start()
        threads.append(thread)
    
    # start them all
    for thread in threads:
        thread.start()
    
    # wait for all to complete
    for thread in threads:
        thread.join()
    return


def combine_all_tfrecord_index(tfrecord_path, save_dir):
    """this file combines all index files in the tfrecord_path"""
    index_pathlist = glob(os.path.join(tfrecord_path, "*of*.index"))
    assert(len(index_pathlist) > 1)
    img_dict_info = {}
    index_pathlist_save_path = os.path.join(save_dir, "image_all_tfrecord.pathlist")
    json_save_path = os.path.join(save_dir, "image_all_tfrecord.json")
    with open(index_pathlist_save_path, "w") as f_write:
        for index_path in index_pathlist:
            with open(index_path, "r") as f_read:
                for line_i in f_read:
                    f_write.write(line_i)
                    info_i = line_i.strip().split(" ")
                    wdnet_id = info_i[1]
                    if not wdnet_id in img_dict_info:
                        img_dict_info[wdnet_id] = []
                    img_dict_info[wdnet_id].append(line_i)
    write_json(json_save_path, img_dict_info)
    return


if __name__ == "__main__":
    """this script is to pack all data into tf-record
    """
    print("WebVision Train")
    tfrecord_path = "../dataset/webvision1k/tfrecord_webvision_train"
    clean_path_list = "../dataset/webvision1k/webvision_train_raw.pathlist"
    save_dir = os.path.join(tfrecord_path, "index_pathlist")
    run_tfrecord_encoding(tfrecord_path, clean_path_list, 8)
    combine_all_tfrecord_index(tfrecord_path, save_dir)
    
    print("WebVision Val")
    tfrecord_path = "../dataset/webvision1k/tfrecord_webvision_val"
    clean_path_list = "../dataset/webvision1k/webvision_val_raw.pathlist"
    save_dir = os.path.join(tfrecord_path, "index_pathlist")
    run_tfrecord_encoding(tfrecord_path, clean_path_list, 8)
    combine_all_tfrecord_index(tfrecord_path, save_dir)

    print("ImgNet Train")
    tfrecord_path = "../dataset/webvision1k/tfrecord_imgnet_train"
    clean_path_list = "../dataset/webvision1k/imgnet_train_raw.pathlist"
    save_dir = os.path.join(tfrecord_path, "index_pathlist")
    run_tfrecord_encoding(tfrecord_path, clean_path_list, 8)
    combine_all_tfrecord_index(tfrecord_path, save_dir)

    print("ImgNet Val")
    tfrecord_path = "../dataset/webvision1k/tfrecord_imgnet_val"
    clean_path_list = "../dataset/webvision1k/imgnet_val_raw.pathlist"
    save_dir = os.path.join(tfrecord_path, "index_pathlist")
    run_tfrecord_encoding(tfrecord_path, clean_path_list, 8)
    combine_all_tfrecord_index(tfrecord_path, save_dir)

