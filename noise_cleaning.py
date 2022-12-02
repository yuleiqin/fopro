#!/usr/bin/env python
import argparse
import os
import random
import tensorboard_logger as tb_logger
import json
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import FoPro, init_weights
import DataLoader.webFG_dataset as webFG496
import DataLoader.webvision_dataset as webvision
from config_train import parser
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    args = parser.parse_args()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print("=> creating model '{}'".format(args.arch))
    model = FoPro(args)
    if not (args.pretrained):
        model.apply(init_weights)
    model = model.cuda(args.gpu)
    model.eval()
    # resume from a checkpoint
    assert(os.path.exists(args.resume) and os.path.isfile(args.resume)), "must load trained model ckpt for noise cleaning"
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module'):
            # remove prefix
            ## 如果是多卡训练存储ckpt时会加上前缀module
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]        
    model.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    args.distributed = False
    # Data loading code
    assert(os.path.exists(args.root_dir)), "please make sure the path to web data is valid {}".format(args.root_dir)
    assert(os.path.exists(args.root_dir_t)), "please make sure the path to fewshot target domain data is valid {}".format(args.root_dir_t)
    assert(os.path.isfile(args.pathlist_t)), "please make sure the pathlist path to fewshot target domain data is valid {}".format(args.pathlist_t)
    if args.webvision:
        ## load webvision dataset
        assert(os.path.isfile(args.pathlist_web)), "please make sure the pathlist path to webvision web data is valid"
        assert(os.path.exists(args.root_dir_test_web)), "please make sure the path to webvision web test data is valid"
        assert(os.path.isfile(args.pathlist_test_web)), "please make the pathlist path to webvision web test data is valid"
        assert(os.path.exists(args.root_dir_test_target)), "please make sure the path to webvision imgnet test data is valid"
        assert(os.path.isfile(args.pathlist_test_target)), "please make the pathlist path to webvision imgnet test data is valid"
        loader = webvision.webvision_dataloader(batch_size=args.batch_size, num_class=args.num_class, num_workers=args.workers,\
            root_dir=args.root_dir, pathlist=args.pathlist_web,\
                root_dir_test_web=args.root_dir_test_web,\
                    pathlist_test_web=args.pathlist_test_web,\
                        root_dir_test_target=args.root_dir_test_target,\
                            pathlist_test_target=args.pathlist_test_target, distributed=args.distributed, crop_size=0.8,\
                                root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                                    save_dir=args.exp_dir, dry_run=args.dry_run,\
                                        use_fewshot=args.use_fewshot, eval_only=True)
        train_loader, _, _, _ = loader.run()
    else:
        ## load webFG496 dataset
        loader = webFG496.webFG496_dataloader(batch_size=args.batch_size, num_class=args.num_class,\
            num_workers=args.workers, root_dir=args.root_dir, distributed=args.distributed, crop_size=0.8,\
                root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                    save_dir=args.exp_dir, dry_run=args.dry_run,\
                        use_fewshot=args.use_fewshot, eval_only=True)
        train_loader, _, _ = loader.run() 
    
    samples = []
    targets = []
    domains = []
    root_dirs = []
    root_dirs2index = {}
    index2root_dirs = {}
    count_index = 0

    print("=> performing noise cleaning on the training data")
    with torch.no_grad():   
        for batch in tqdm(train_loader):
            img_domain = batch[3]
            pathlist = batch[4]
            ## 旧inference方式手动计算定义所有参数
            # img = batch[0]
            # target = batch[1]
            # img_aug = batch[2]
            # img_index = batch[5]
            # img = img.cuda(args.gpu, non_blocking=True)
            # img_domain = img_domain.cuda(args.gpu, non_blocking=True).view(-1)
            # target = target.cuda(args.gpu, non_blocking=True)
            # output, _, target, q_compress = model(batch, args, is_eval=True)
            # logits = torch.mm(q_compress, model.prototypes.t())/args.temperature   
            # soft_label = (F.softmax(output, dim=1) + F.softmax(logits,dim=1))/2
            # img_target_domain = (img_domain > 0)
            # gt_score = soft_label[target>=0,target]
            # clean_idx = gt_score>(1/args.num_class)     
            # max_score, hard_label = soft_label.max(1)
            # correct_idx = max_score>args.pseudo_th
            # target[correct_idx] = hard_label[correct_idx]
            # clean_idx = clean_idx | correct_idx
            # clean_idx = clean_idx | img_target_domain

            ## 新inference方式直接推理得到所有结果
            _, _, soft_label, indxs, dists = model(batch, args, is_eval=False,\
                is_proto=True, is_clean=3,\
                    is_analysis=True, is_relation=True)
            target, cos_max_idx, cos_median_idx, cos_min_idx, clean_idx, arcface_idx = indxs
            # dist_target, dist_min, dist_mean, dist_median, dist_max = dists
            for clean, label, path, domain in zip(clean_idx, target.cpu(), pathlist, img_domain):
                if clean:
                    if args.webvision:
                        # tfrecord, offset = self.samples[index]
                        # path = tfrecord + "@" + str(offset)
                        info = path.split("@")
                        tfrecord = info[0]
                        offset = int(info[1])
                        img_root_dir = os.path.dirname(tfrecord)
                        tfrecord_name = os.path.basename(tfrecord)
                        samples.append([tfrecord_name, offset])
                    else:
                        img_root_dir = os.path.dirname(os.path.dirname(path))
                        img_name = path.replace(img_root_dir, "")
                        if img_name.startswith("/"):
                            img_name = img_name[1:]
                        samples.append([img_name, label.item()])
                    ## 记录文件名仅仅记录底层路径以省略文本
                    if not img_root_dir in root_dirs2index:
                        root_dirs2index[img_root_dir] = count_index
                        index2root_dirs[count_index] = img_root_dir
                        count_index += 1
                    img_cur_index = root_dirs2index[img_root_dir]
                    root_dirs.append(int(img_cur_index))
                    targets.append(label.item())
                    domains.append(domain.item())
    
    with open(args.annotation, "w") as f:
        json.dump({'samples':samples,\
            'targets':targets,\
                'domains':domains,\
                    "roots":root_dirs,\
                        "root2index":root_dirs2index,\
                            "index2root":index2root_dirs}, f)
    print("=> pseudo-label annotation saved to {}".format(args.annotation))
    return


if __name__ == '__main__':
    main()
