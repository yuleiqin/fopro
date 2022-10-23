from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageFile
import torch
import os
import io
import albumentations as alb
import struct
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import cv2
import sys
from copy import deepcopy as copy

sys.path.append('.')
from .example_pb2 import Example
from .fancy_pca import FancyPCA
import sys
sys.path.append("../")
from utils.rotate import rotate_and_crop
from utils.augmentations import RandomBorder, RandomTranslate
from utils.augmentations import RandomTextOverlay, RandomStripesOverlay



def get_concat_h_resize(im1, im2, resample=Image.BILINEAR, resize_big_image=True):
    if im1.height == im2.height:
        _im1 = im1
        _im2 = im2
    elif (((im1.height > im2.height) and resize_big_image) or
          ((im1.height < im2.height) and not resize_big_image)):
        _im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
    dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (_im1.width, 0))
    return dst



class webvision_dataset(Dataset): 
    def __init__(self, root_dir, pathlist, transform, mode, num_class,\
        transform_strong=None, root_dir_target="", pathlist_target="",\
            save_dir="", dry_run=False, use_fewshot=True, annotation="",\
                no_color_transform=False, fast_eval=False): 
        self.root_dir = root_dir
        self.pathlist = pathlist
        self.transform = transform
        self.mode = mode
        self.imgs_by_class = None
        self.transform_strong = transform_strong
        self.albs_transform_color = [alb.Equalize(p=0.5), alb.ColorJitter(p=0.5),\
            alb.ToGray(p=0.5), alb.Sharpen(p=0.5), alb.HueSaturationValue(p=0.5),\
                alb.RandomBrightness(p=0.5), alb.RandomBrightnessContrast(p=0.5),\
                    alb.RandomToneCurve(p=0.5)]
        self.albs_transform_basic = [RandomBorder(), RandomTranslate(),\
            RandomTextOverlay(), RandomStripesOverlay(),
                alb.OpticalDistortion(p=0.5),\
                    alb.GridDistortion(p=0.5, border_mode=cv2.BORDER_REPLICATE)]
        self.albs_transform_noise = [alb.ISONoise(p=0.5), alb.RandomFog(p=0.5, fog_coef_upper=0.5),\
            alb.RandomSnow(p=0.5, brightness_coeff=1.2), alb.RandomRain(p=0.5, drop_length=5),\
                alb.RandomShadow(p=0.5, num_shadows_lower=0, num_shadows_upper=1),\
                    alb.GaussNoise(p=0.5), alb.ImageCompression(quality_lower=95, p=1),\
                        alb.MotionBlur(p=0.5), alb.Blur(p=1),\
                            alb.GaussianBlur(p=0.5), alb.GlassBlur(sigma=0.2, p=1)]
        assert(os.path.exists(root_dir) and os.path.isfile(pathlist))
        assert(os.path.exists(save_dir))
        self.save_dir = save_dir
        self.no_color_transform = no_color_transform
        if mode == "train" and annotation != "" and os.path.exists(annotation):
            print("[TRAIN] Load samples by pseudo label json for web domain")
            annotation_json = json.load(open(annotation, "r"))
            samples = annotation_json['samples']
            root_dirs = annotation_json["roots"]
            index2roots = annotation_json["index2root"]
            samples_full = []
            for sample, root_dir in zip(samples, root_dirs):
                ## 增加根目录
                tf_record, offset = sample
                tf_record_full = os.path.join(index2roots[str(root_dir)], tf_record)
                samples_full.append([tf_record_full, offset])
            self.samples = samples_full
            self.targets = annotation_json['targets']
            self.domains = annotation_json['domains']
            print("number of samples", len(self.samples),\
                "number of labels", len(self.targets),\
                    "number of domain labels", len(self.domains))
        else:
            if mode == "train" or mode == "test" or (mode == "fewshot" and not use_fewshot):
                self.samples = []
                self.targets = []
                visited_class = set()
                with open(pathlist, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        tf_record_offset = line.strip().split(" ")[0]
                        tf_record = tf_record_offset.split("@")[0]
                        offset = int(tf_record_offset.split("@")[1])
                        tf_record_path = os.path.join(self.root_dir, tf_record)
                        # assert(os.path.isfile(tf_record_path))
                        meta_info = line.strip().replace(tf_record_offset + " ", "")
                        if mode != "test":
                            json_info = json.loads(meta_info)
                            target = int(json_info["label"])
                        elif mode == "test":
                            target = int(meta_info)
                        assert (target<num_class)
                        self.samples.append([tf_record_path, offset])
                        self.targets.append(target)
                        visited_class.add(target)
                if mode == "train" or (mode == "fewshot" and not use_fewshot):
                    self.domains = [0 for _ in range(len(self.samples))]
                    print("[TRAIN] Load samples by pathlist for web domain")
                elif mode == "test":
                    self.domains = [1 for _ in range(len(self.samples))]
                    print("[TEST] Load samples by pathlist for target domain")

                print("number of samples", len(self.samples),\
                    "number of labels", len(self.targets),\
                        "number of domain labels", len(self.domains),\
                            "number of classes", len(visited_class))               

            if root_dir_target != "" and pathlist_target != "" and os.path.exists(root_dir_target) and os.path.exists(pathlist_target):
                if mode != "test":
                    samples_supp = []
                    targets_supp = []
                    visited_class_supp = set()
                    with open(pathlist_target, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            tf_record_offset = line.strip().split(" ")[0]
                            meta_info = line.strip().replace(tf_record_offset + " ", "")
                            tf_record = tf_record_offset.split("@")[0]
                            offset = int(tf_record_offset.split("@")[1])
                            tf_record_path = os.path.join(root_dir_target, tf_record)
                            # assert(os.path.isfile(tf_record_path))
                            json_info = json.loads(meta_info)
                            target = json_info["label"]
                            assert (target<num_class)
                            samples_supp.append([tf_record_path, offset])
                            targets_supp.append(target)
                            visited_class_supp.add(target)
                    domains_supp = [1 for _ in range(len(samples_supp))]
                    if mode == "train" and use_fewshot:
                        print("[TRAIN] Load samples by pathlist for target domain")
                        if fast_eval != 0:
                            ## 使用抽样每类样本快速验证
                            imgs_by_target = {}
                            imgs_by_target_fewshot = {}
                            for sample, target, domain in zip(self.samples, self.targets, self.domains):
                                if not target in imgs_by_target:
                                    imgs_by_target[target] = []
                                imgs_by_target[target].append([sample, target, domain])
                            for sample, target, domain in zip(samples_supp, targets_supp, domains_supp):
                                if not target in imgs_by_target_fewshot:
                                    imgs_by_target_fewshot[target] = []
                                imgs_by_target_fewshot[target].append([sample, target, domain])
                            print("number of few-shots", len(imgs_by_target_fewshot))
                            N_samples = 200
                            ### must include DRUMSTICK
                            target_must_include = []
                            if fast_eval == 1:
                                ### webvision
                                target_list = sorted(list(imgs_by_target.keys()))[:50]
                                target_must_include = [542, 428, 134, 517, 677, 998]
                                target_list += target_must_include
                            elif fast_eval == 2:
                                ### g500=>n03250847(137=drumstick)
                                target_list = sorted(list(imgs_by_target.keys()))[:50]
                                target_must_include = [137, 246, 385, 227]
                                target_list += target_must_include
                            print("Fast Eval of WebVision Train Class ID", target_list)
                            samples_fewshot = []
                            targets_fewshot = []
                            domains_fewshot = []
                            for target in target_list:
                                img_list = imgs_by_target[target]
                                img_list_fewshot = imgs_by_target_fewshot[target]
                                if target in target_must_include:
                                    ## 使用所有样本数据进行训练学习
                                    img_list_sampled = img_list
                                else:
                                    ## 仅仅抽样部分数据进行训练学习
                                    img_list_sampled = random.sample(img_list, min(N_samples, len(img_list)))
                                img_list_sampled += img_list_fewshot
                                for img_item in img_list_sampled:
                                    samples_fewshot.append(img_item[0])
                                    targets_fewshot.append(img_item[1])
                                    domains_fewshot.append(img_item[2])
                            self.samples = samples_fewshot
                            self.targets = targets_fewshot
                            self.domains = domains_fewshot
                        else:
                            self.samples += samples_supp
                            self.targets += targets_supp
                            self.domains += domains_supp
                    elif mode == "fewshot":
                        print("[FEW-SHOT] Load samples by pathlist for target domain")
                        if use_fewshot:
                            ## 使用fewshot样本
                            self.samples = samples_supp
                            self.targets = targets_supp
                            self.domains = domains_supp
                        else:
                            ## 使用抽样每类样本
                            imgs_by_target = {}
                            for sample, target, domain in zip(self.samples, self.targets, self.domains):
                                if not target in imgs_by_target:
                                    imgs_by_target[target] = []
                                imgs_by_target[target].append([sample, target, domain])
                            samples_fewshot = []
                            targets_fewshot = []
                            domains_fewshot = []
                            N_samples = 50
                            target_list = list(imgs_by_target.keys())
                            for target in target_list:
                                img_list = imgs_by_target[target]
                                img_list_sampled = random.sample(img_list, min(N_samples, len(img_list)))
                                for img_item in img_list_sampled:
                                    samples_fewshot.append(img_item[0])
                                    targets_fewshot.append(img_item[1])
                                    domains_fewshot.append(img_item[2])
                            self.samples = samples_fewshot
                            self.targets = targets_fewshot
                            self.domains = domains_fewshot

                    print("number of samples", len(self.samples),\
                        "number of labels", len(self.targets),\
                            "number of domains", len(self.domains),\
                                "number of classes", len(visited_class_supp))
            else:
                print("image path {} or label path {} does not exist".format(root_dir_target, pathlist_target))

        if dry_run:
            if mode == "train":
                target_ids = set()
                random_index = []
                for idx, target_id in enumerate(self.targets):
                    if not (target_id in target_ids):
                        target_ids.add(target_id)
                        random_index.append(idx)
                random_index += np.arange(len(self.samples)-32, len(self.samples)).tolist()
                random.shuffle(random_index)
                self.samples = [self.samples[idx] for idx in random_index]
                self.targets = [self.targets[idx] for idx in random_index]
                self.domains = [self.domains[idx] for idx in random_index]
            else:
                self.samples = self.samples[:64]
                self.targets = self.targets[:64]
                self.domains = self.domains[:64]
            print("domains ", self.domains)
            print("[Dry-run] number of samples", len(self.samples),\
                "number of labels", len(self.targets),\
                    "number of domains", len(self.domains))

        print("Preparing mapping from image path to image index number")
        self.samples_index = []
        save_mapping_index = os.path.join(save_dir, "index_mapping_{}.txt".format(mode))
        with open(save_mapping_index, "w") as f_write:
            for index, item in enumerate(self.samples):
                tfrecord, offset = item
                target = self.targets[index]
                path = tfrecord + "@" + str(offset)
                self.samples_index.append(index)
                f_write.write(" ".join([str(index), str(path), str(target)]) + "\n")
        
        self.samples_copy = copy(self.samples)
        self.targets_copy = copy(self.targets)
        self.domains_copy = copy(self.domains)
        self.samples_index_copy = copy(self.samples_index)

    def _parser(self, feature_list):
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

    def get_tfrecord_image(self, record_file, offset):
        """read images from tfrecord"""
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
        image = self._parser(feature)
        return image

    def resample(self):
        print('=> down-sampling dataset scale for balancing')
        assert(self.mode == "train"), "repeat for data-resampling is only valid for traininig mode"
        ## 需要更新4个列表 samples; targets; domains; sample_index
        if self.imgs_by_class == None:
            imgs_by_class = {}
            ## 先按照类别整理所有图像样本
            for im, lab, domain, im_index in zip(self.samples_copy, self.targets_copy,\
                self.domains_copy, self.samples_index_copy):
                if not (lab in imgs_by_class):
                    imgs_by_class[lab] = [[], []]   # 第一个列表存储fewshot样本 第二个列表存储非fewshot样本
                if domain > 0:
                    imgs_by_class[lab][0].append([im, lab, domain, im_index])
                else:
                    imgs_by_class[lab][1].append([im, lab, domain, im_index])
            self.imgs_by_class = imgs_by_class
        
        samples_balance = []
        targets_balance = []
        domains_balance = []
        samples_index_balance = []
        ### N_ratio for downsampling
        N_ratio = 4
        # N_ratio = 16
        for lab in self.imgs_by_class.keys():
            ## 从平衡的样本集中随机抽取固定数目的样本
            imgs_all_lab = self.imgs_by_class[lab]
            imgs_fewshot_lab = imgs_all_lab[0]
            num_fewshot_imgs = len(imgs_fewshot_lab)
            imgs_web_lab = imgs_all_lab[1]
            num_web_imgs = len(imgs_web_lab)
            num_web_imgs_sampled = min(max(num_web_imgs//N_ratio, 4), num_web_imgs)
            imgs_web_sampled_lab = random.sample(imgs_web_lab, num_web_imgs_sampled)
            # print("category {} has {} web images and {} fewshot images".format(lab,\
            #     num_web_imgs_sampled, num_fewshot_imgs))
            imgs_sampled_lab = imgs_web_sampled_lab + imgs_fewshot_lab
            for item in imgs_sampled_lab:
                samples_balance.append(item[0])
                targets_balance.append(item[1])
                domains_balance.append(item[2])
                samples_index_balance.append(item[3])
        ## 更新样本集
        self.samples = samples_balance
        self.targets = targets_balance
        self.domains = domains_balance
        self.samples_index = samples_index_balance
        print('=> done {} number of resampled items'.format(len(self.samples)))
        return


    def repeat(self):
        print('=> repeating dataset for balancing')
        assert(self.mode == "train"), "repeat for data-resampling is only valid for traininig mode"
        ## 需要更新4个列表 samples; targets; domains; sample_index
        targets = np.array(self.targets_copy)
        uniq, freq = np.unique(targets, return_counts=True)
        inv = (1/freq)**0.5
        p = inv/inv.sum()
        weight = (10 * p)/(p.min())
        weight = weight.astype(int)
        weight = {u:w for u,w in zip(uniq, weight)}
        ## 从初始化的所有数据列表中进行re-balance抽样
        samples_balance = []
        targets_balance = []
        domains_balance = []
        samples_index_balance = []
        for im, lab, domain, im_index in zip(self.samples_copy, self.targets_copy,\
            self.domains_copy, self.samples_index_copy):
            samples_balance += [im]*weight[lab]
            targets_balance += [lab]*weight[lab]
            domains_balance += [domain]*weight[lab]
            samples_index_balance += [im_index]*weight[lab]
        ## 更新样本集
        self.samples = []
        self.targets = []
        self.domains = []
        self.samples_index = []
        ## 从平衡的样本集中随机抽取固定数目的样本
        index_shuf = list(range(len(samples_balance)))
        random.shuffle(index_shuf)
        for i in index_shuf[:len(self.targets_copy)]:
            self.samples.append(samples_balance[i])
            self.targets.append(targets_balance[i])
            self.domains.append(domains_balance[i])
            self.samples_index.append(samples_index_balance[i])
        print('=> done')
        return

    def alb_transform(self, image):
        """use albumentation transform for data augmentation"""
        image = np.array(image)
        alb_trans = [random.choice(self.albs_transform_basic),\
            random.choice(self.albs_transform_noise)]
        if not (self.no_color_transform):
            alb_trans.append(random.choice(self.albs_transform_color))
        alb_trans = alb.Compose(random.sample(alb_trans, random.randint(1, len(alb_trans))))
        image_aug = alb_trans(image=image)['image']
        image_aug = Image.fromarray(image_aug)
        return image_aug


    def __getitem__(self, index):
        tfrecord, offset = self.samples[index]
        target = self.targets[index]
        image = self.get_tfrecord_image(tfrecord, offset)
        # if self.mode == "train":
        #     img = self.alb_transform(image)
        # else:
        img = image
        path = tfrecord + "@" + str(offset)
        img = self.transform(img)
        if self.mode == "train":
            if self.no_color_transform:
                rotate_deg = np.random.randint(low=-45, high=45)
                image = rotate_and_crop(image, rotate_deg)
            img_aug = self.alb_transform(image)
            img_aug = self.transform_strong(img_aug)
            return img, target, img_aug, torch.Tensor([self.domains[index]]), path, torch.Tensor([self.samples_index[index]])
        elif self.mode == "test" or self.mode == "fewshot":
            return img, target, torch.Tensor([self.domains[index]]), path, torch.Tensor([self.samples_index[index]])
           
    def __len__(self):
        return len(self.samples)


class webvision_dataloader():
    def __init__(self, batch_size, num_class, num_workers, root_dir, pathlist,\
        root_dir_test_web, pathlist_test_web, root_dir_test_target, pathlist_test_target,\
            distributed, crop_size=0.8, root_dir_target="", pathlist_target="",\
                save_dir="", dry_run=False, use_fewshot=True, annotation="",\
                    no_color_transform=False, fast_eval=False, eval_only=False):
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.use_fewshot = use_fewshot
        self.pathlist = pathlist
        self.annotation = annotation
        self.distributed = distributed
        self.fast_eval = fast_eval
        self.root_dir_test_web = root_dir_test_web
        self.pathlist_test_web = pathlist_test_web
        self.root_dir_test_target = root_dir_test_target
        self.pathlist_test_target = pathlist_test_target
        self.root_dir_target = root_dir_target
        self.pathlist_target = pathlist_target
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_size, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            FancyPCA(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.no_color_transform = no_color_transform
        if no_color_transform:
            self.transform_strong = transforms.Compose([
                transforms.RandomResizedCrop(448, scale=(0.8, 1.0)),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                FancyPCA(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_strong = transforms.Compose([
                transforms.RandomResizedCrop(448, scale=(crop_size, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),            
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                FancyPCA(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.transform_test = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                FancyPCA(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])
        self.eval_only = eval_only
        if eval_only:
            self.transform_train = self.transform_test
            self.transform_strong = self.transform_test

    def run(self):
        print("------------------------------------------------------")
        save_dir_train = os.path.join(self.save_dir, "data_train")
        os.makedirs(save_dir_train, exist_ok=True)
        train_dataset = webvision_dataset(root_dir=self.root_dir, pathlist=self.pathlist, transform=self.transform_train,\
            mode="train", num_class=self.num_class, transform_strong = self.transform_strong,\
                root_dir_target=self.root_dir_target, pathlist_target=self.pathlist_target,\
                    save_dir=save_dir_train, dry_run=self.dry_run,\
                        use_fewshot=self.use_fewshot, annotation=self.annotation,\
                            no_color_transform=self.no_color_transform,\
                                fast_eval=self.fast_eval)
        print("------------------------------------------------------")
        save_dir_fewshot = os.path.join(self.save_dir, "data_fewshot")
        os.makedirs(save_dir_fewshot, exist_ok=True)
        fewshot_dataset = webvision_dataset(root_dir=self.root_dir, pathlist=self.pathlist, transform=self.transform_train,\
            mode="fewshot", num_class=self.num_class, transform_strong = self.transform_strong,\
                root_dir_target=self.root_dir_target, pathlist_target=self.pathlist_target,\
                    save_dir=save_dir_fewshot, dry_run=self.dry_run,\
                        use_fewshot=self.use_fewshot)
        print("------------------------------------------------------")
        save_dir_test_web = os.path.join(self.save_dir, "data_test_web")
        os.makedirs(save_dir_test_web, exist_ok=True)
        test_dataset_web = webvision_dataset(root_dir=self.root_dir_test_web, pathlist=self.pathlist_test_web,\
            transform=self.transform_test, mode='test', num_class=self.num_class,\
                save_dir=save_dir_test_web, dry_run=self.dry_run)
        print("------------------------------------------------------")
        save_dir_test_imgnet = os.path.join(self.save_dir, "data_test_imgnet")
        os.makedirs(save_dir_test_imgnet, exist_ok=True)
        test_dataset_target = webvision_dataset(root_dir=self.root_dir_test_target, pathlist=self.pathlist_test_target,\
            transform=self.transform_test, mode='test', num_class=self.num_class,\
                save_dir=save_dir_test_imgnet, dry_run=self.dry_run)
        print("------------------------------------------------------")
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            fewshot_sampler = torch.utils.data.distributed.DistributedSampler(fewshot_dataset, shuffle=False)
            test_sampler_web = torch.utils.data.distributed.DistributedSampler(test_dataset_web, shuffle=False)
            test_sampler_target = torch.utils.data.distributed.DistributedSampler(test_dataset_target, shuffle=False)
        else:
            self.train_sampler = None
            fewshot_sampler = None
            test_sampler_web = None
            test_sampler_target = None

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True)                                              

        fewshot_loader = DataLoader(
            dataset=fewshot_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=fewshot_sampler)   
             
        test_loader_web = DataLoader(
            dataset=test_dataset_web,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=test_sampler_web)                             

        test_loader_target = DataLoader(
            dataset=test_dataset_target,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=test_sampler_target)
    
        return train_loader, fewshot_loader, test_loader_web, test_loader_target

