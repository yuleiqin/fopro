## python library
import builtins
import math
import os
import json
import random
import shutil
import time
import warnings
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import struct
import io
from DataLoader.example_pb2 import Example
from config_train import parser
from utils.lr_scheduler_webFG import lr_scheduler as lr_scheduler_webFG
from train import init_prototype_fewshot
from model import FoPro, init_weights
from feat_tsne import run_and_plot_tsne
import DataLoader.webFG_dataset as webFG496
import DataLoader.webvision_dataset as webvision

import tensorboard_logger as tb_logger

import warnings
warnings.filterwarnings('ignore')


def get_tfrecord_image(record_file, offset):
    """read images from tfrecord"""
    def _parser(feature_list):
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
    image = _parser(feature)
    return image


def main():
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print('You have chosen to seed training.')
        # cudnn.deterministic = True
        # warnings.warn('This will turn on the CUDNN deterministic setting, '
        # 'which can slow down your training considerably! '
        # 'You may see unexpected behavior when restarting '
        # 'from checkpoints.')
    args.gpu = 0
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    args.distributed = False
    print("distributed training {}".format(args.distributed))
    ## prepare the directory for saving
    os.makedirs(args.exp_dir, exist_ok=True)
    args.tensorboard_dir = os.path.join(args.exp_dir, 'tensorboard')
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    ngpus_per_node = torch.cuda.device_count()
    print("{} single process running with {} gpus".format(args.world_size,\
        ngpus_per_node))

    ## start evaluation inference
    main_worker(args.gpu, ngpus_per_node, args)

    ## start qualitative illustration
    if args.webvision:
        target_domain_dir = args.root_dir_test_target
        target_domain_imglist = args.pathlist_test_target
        fast_eval = args.fast_eval
        check_statistics_webvision(args.exp_dir, target_domain_dir,\
            target_domain_imglist, fast_eval)
    else:
        target_domain_dir = args.root_dir
        target_domain_imglist = os.path.join(target_domain_dir, "val-list.txt")
        check_statistics_webFG496(args.exp_dir, target_domain_dir, target_domain_imglist)

    run_and_plot_tsne(save_root_path=args.exp_dir)
    return


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))    

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = FoPro(args)
    if not (args.pretrained):
        model.apply(init_weights)
    if args.gpu == 0:
        print(model)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    ## optionally resume from a checkpoint
    resume_path = args.resume
    assert os.path.exists(resume_path) and os.path.isfile(resume_path)
    print("=> loading checkpoint '{}'".format(resume_path))
    if args.gpu is None:
        checkpoint = torch.load(resume_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(resume_path, map_location=loc)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            # remove prefix
            state_dict[k.replace('module.','')] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)
    epoch = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume_path, epoch))

    cudnn.benchmark = True
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
                            pathlist_test_target=args.pathlist_test_target, distributed=args.distributed, crop_size=0.2,\
                                root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                                    save_dir=args.exp_dir, dry_run=args.dry_run,\
                                        use_fewshot=args.use_fewshot, fast_eval=args.fast_eval, eval_only=True)
        train_loader, fewshot_loader, test_loader_web, test_loader_target = loader.run() 
    else:
        ## load webFG496 dataset
        loader = webFG496.webFG496_dataloader(batch_size=args.batch_size, num_class=args.num_class,\
            num_workers=args.workers, root_dir=args.root_dir, distributed=args.distributed, crop_size=0.2,\
                root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                    save_dir=args.exp_dir, dry_run=args.dry_run,\
                        use_fewshot=args.use_fewshot, eval_only=True)
        train_loader, fewshot_loader, test_loader = loader.run()
    
    if args.gpu==0:
        logger = tb_logger.Logger(logdir=args.tensorboard_dir, flush_secs=2)
    else:
        logger = None

    if epoch < args.init_proto_epoch:
        ## 过一遍模型得到fewshot样本作为prototype
        if args.use_fewshot:
            print("===>Extract features of fewshot samples as prototypes")
            init_prototype_fewshot(model, fewshot_loader, args, epoch, is_eval=True)
        else:
            print("===>Extract features of training samples/average per class as prototypes")
            init_prototype_fewshot(model, train_loader, args, epoch, is_eval=True)
    else:
        print("===>Use model default prototypes")

    print("Save model prototypes")
    prototypes = model.prototypes
    save_path_prototypes = os.path.join(args.exp_dir, "prototypes.npy")
    np.save(save_path_prototypes, prototypes.cpu().numpy())

    acc1_train, acc5_train = test(model, train_loader, args, 0, logger,\
        dataset_name="WebTrain", is_trainset=True)
    print("Web Train Accuracy top 1 {} top 5 {}".format(acc1_train, acc5_train))

    acc1_fewshot, acc5_fewshot = test(model, fewshot_loader, args, 0, logger, dataset_name="FewShot")
    print("Web FewShot Accuracy top 1 {} top 5 {}".format(acc1_fewshot, acc5_fewshot))

    if args.webvision:
        ## test webvision dataset  
        acc1_web, acc5_web = test(model, test_loader_web, args, 0, logger, dataset_name="WebVision")
        print("Acc webvision top 1 {} top 5 {}".format(acc1_web, acc5_web))    
        acc1_imgnet, acc5_imgnet = test(model, test_loader_target, args, 0, logger, dataset_name="ImgNet")
        print("Acc imagenet top 1 {} top 5 {}".format(acc1_imgnet, acc5_imgnet))
    else:
        ## test webFineGrained dataset
        acc1, acc5 = test(model, test_loader, args, 0, logger, dataset_name="FineGrained")
        print("Acc FGdataset top 1 {} top 5 {}".format(acc1, acc5))
    return


def test(model, test_loader, args, epoch, tb_logger, dataset_name="WebVision", is_trainset=False):
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()    
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        save_path_csv = os.path.join(args.exp_dir, dataset_name+".csv")
        save_path_clean_csv = os.path.join(args.exp_dir, dataset_name+"_clean_flag.csv")
        csv_files = []
        # evaluate on webvision val set
        img_features = []
        save_path_features = os.path.join(args.exp_dir, dataset_name+"_feat.npy")
        img_path_is_clean_record = []

        for batch in tqdm(test_loader):
            ## outputs, feat, target, feat_reconstruct
            outputs, q_compress, relation_score, targets_idx, distances = model(batch, args,\
                is_analysis=True, is_eval=not (is_trainset), is_proto=True, is_clean=3)
            # if is_trainset:
            target, dist_min_idx, dist_median_idx, dist_max_idx, clean_idx, arcface_idx = targets_idx
            dist_target, dist_pred, dist_min, dist_median, dist_max, dist_sim = distances
            # else:
            #     target, dist_min_idx, dist_median_idx, dist_max_idx, clean_idx = targets_idx
            #     dist_target, dist_pred, dist_min, dist_median, dist_max = distances
            ## cosine distance the smaller the closer/more similar to prototype
            res, pred_top = accuracy(outputs, target, topk=(1, 5))
            relation_score_target = relation_score[target>=0, target]
            relation_score_pred = relation_score[pred_top>=0, pred_top]
            relation_score_max, rm_max_label = relation_score.max(1)
            ## img, target, torch.Tensor([self.domains[index]]), path, torch.Tensor([self.samples_index[index]])
            if is_trainset:
                pathlist = batch[4]
            else:
                pathlist = batch[3]
            
            # if is_trainset:
            for img_path, img_label, img_pred, img_dist_target, img_dist_pred,\
                img_dist_min, img_dist_min_id,\
                    img_dist_median, img_dist_median_id,\
                        img_dist_max, img_dist_max_id, img_dist_sim in zip(pathlist,\
                            target, pred_top, dist_target, dist_pred,\
                                dist_min, dist_min_idx,\
                                    dist_median, dist_median_idx,\
                                        dist_max, dist_max_idx, dist_sim.view(-1)):
                csv_files.append([img_path, img_label.item(), img_pred.item(),\
                    img_dist_target.item(), img_dist_pred.item(),\
                        img_dist_min.item(), int(img_dist_min_id.item()),\
                            img_dist_median.item(), int(img_dist_median_id.item()),\
                                img_dist_max.item(), int(img_dist_max_id.item()),\
                                    float(img_dist_sim.item())])

            for img_path, img_label, img_pred,\
                img_is_clean, img_is_arcface,\
                    img_rm_label, img_rm_pred,\
                        img_rm_max_id, img_rm_max in zip(
                            pathlist, target, pred_top, clean_idx, arcface_idx,\
                                relation_score_target.view(-1), relation_score_pred.view(-1),\
                                    rm_max_label.view(-1), relation_score_max.view(-1)):
                img_path_is_clean_record.append([img_path,\
                    int(img_label.item()), int(img_pred.item()),\
                        int(img_is_clean.item()), int(img_is_arcface.item()),\
                            float(img_rm_label.item()), float(img_rm_pred.item()),\
                                int(img_rm_max_id.item()), float(img_rm_max.item())])
            # else:
            #     for img_path, img_label, img_pred, img_dist_target, img_dist_pred,\
            #         img_dist_min, img_dist_min_id,\
            #             img_dist_median, img_dist_median_id,\
            #                 img_dist_max, img_dist_max_id in zip(pathlist,\
            #                     target, pred_top, dist_target, dist_pred,\
            #                         dist_min, dist_min_idx,\
            #                             dist_median, dist_median_idx,\
            #                                 dist_max, dist_max_idx):
            #         csv_files.append([img_path, img_label.item(), img_pred.item(),\
            #             img_dist_target.item(), img_dist_pred.item(),\
            #                 img_dist_min.item(), int(img_dist_min_id.item()),\
            #                     img_dist_median.item(), int(img_dist_median_id.item()),\
            #                         img_dist_max.item(), int(img_dist_max_id.item())])

            #     for img_path, img_label, img_pred, img_is_clean,\
            #         img_rm_label, img_rm_pred,\
            #             img_rm_max_id, img_rm_max in zip (pathlist,\
            #         target, pred_top, clean_idx,\
            #             relation_score_target.view(-1), relation_score_pred.view(-1),\
            #                 rm_max_label.view(-1), relation_score_max.view(-1)):
            #         img_path_is_clean_record.append([img_path,\
            #             int(img_label.item()), int(img_pred.item()), int(img_is_clean.item()),\
            #                 float(img_rm_label.item()), float(img_rm_pred.item()),\
            #                     int(img_rm_max_id.item()), float(img_rm_max.item())])

            if is_trainset:
                domain = batch[3]
                for img_domain, img_label, img_feat_compress in zip(domain, target,\
                    q_compress):
                    img_domain = img_domain.item()
                    img_label = img_label.item()
                    img_feat_compress = img_feat_compress.cpu().numpy()
                    img_features.append([img_domain, img_label] + img_feat_compress.tolist())
            
            acc1, acc5 = res
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])
        
        if is_trainset:
            np.save(save_path_features, np.array(img_features))

        with open(save_path_csv, "w") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Image Path', 'Image Label', 'Image Prediction',\
                'Distance To Target Prototype', 'Distance To Predicted Prototype',\
                    'Min DistanceToPrototype', 'Min DistanceToPrototype ID',\
                        'Median DistanceToPrototype', 'Median DistanceToPrototype ID',\
                            'Max DistanceToPrototype', 'Max DistanceToPrototype ID', 'Distance-Similarity'])
            for line in csv_files:
                csv_writer.writerow(line)

        with open(save_path_clean_csv, "w") as csvfile:
            csv_writer = csv.writer(csvfile)
            # if is_trainset:
            csv_writer.writerow(['Image Path', 'Image Label', 'Image Prediction',\
                'Is_Clean', 'Is_Arcface-low-dim-Cls',\
                    'Relation Score (Label)', 'Relation Score (Pred)',\
                        'Relation Max Class', 'Relation Score (Max)'])
            # else:
            #     csv_writer.writerow(['Image Path', 'Image Label', 'Image Prediction',\
            #         'Is_Clean',\
            #             'Relation Score (Label)', 'Relation Score (Pred)',\
            #                 'Relation Max Class', 'Relation Score (Max)'])
            for line in img_path_is_clean_record:
                csv_writer.writerow(line)

        # average across all processes
        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)
        print('%s Accuracy is %.2f%% (%.2f%%)'%(dataset_name,\
            acc_tensors[0], acc_tensors[1]))
        if args.gpu ==0:
            tb_logger.log_value('{} top1 Acc'.format(dataset_name),\
                acc_tensors[0], epoch)
            tb_logger.log_value('{} top5 Acc'.format(dataset_name),\
                acc_tensors[1], epoch)
    return acc_tensors[0].item(), acc_tensors[1].item()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)
        return

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        _, pred_top = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k, :].sum(0)
            correct_k = correct_k.view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred_top.view(-1)


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


def get_concat_v_resize(im1, im2, resample=Image.BILINEAR, resize_big_image=True):
    if im1.width == im2.width:
        _im1 = im1
        _im2 = im2
    elif (((im1.width > im2.width) and resize_big_image) or
          ((im1.width < im2.width) and not resize_big_image)):
        _im1 = im1.resize((im2.width, int(im1.height * im2.width / im1.width)), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((im1.width, int(im2.height * im1.width / im2.width)), resample=resample)
    dst = Image.new('RGB', (_im1.width, _im1.height + _im2.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (0, _im1.height))
    return dst


def check_statistics_webFG496(root_dir, target_domain_dir, target_domain_imglist):
    ## 读取真实的测试集图像 按类别分类得到参考图
    imgs_gt_by_class = {}
    with open(target_domain_imglist, "r") as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip().split(": ")
            img_path = info[0]
            img_label = int(info[1])
            if not (img_label in imgs_gt_by_class):
                imgs_gt_by_class[img_label] = []
            imgs_gt_by_class[img_label].append(os.path.join(target_domain_dir, img_path))

    train_csv_path = os.path.join(root_dir, "WebTrain.csv")
    fewshot_csv_path = os.path.join(root_dir, "FewShot.csv")
    test_csv_path = os.path.join(root_dir, "FineGrained.csv")
    def calc_confusion_matrix(csv_path, class_id2class_name=None, is_trainset=False):
        y_pred = []
        y_true = []
        x_path = []
        if class_id2class_name is None:
            ## 重新构建字典来保存映射
            class_id2class_name = {}
            with open(csv_path, "r") as f:
                csv_reader = csv.reader(f)
                for idx, line in enumerate(csv_reader):
                    if idx == 0:
                        continue
                    img_path = line[0]
                    img_label = line[1]
                    img_label = int(img_label)
                    class_name = os.path.basename(os.path.dirname(img_path)).replace(" ", "_")
                    class_id2class_name[img_label] = class_name
            print(class_id2class_name.keys())

        ## path_to_val_folder/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg,0,144
        imgs_by_class = {}
        with open(csv_path, "r") as f:
            csv_reader = csv.reader(f)
            for idx, line in enumerate(csv_reader):
                if idx == 0:
                    continue
                if len(line) == 11:
                    img_path, img_label, img_pred,\
                        img_dist_target, img_dist_pred,\
                            img_dist_min, img_dist_min_id,\
                                img_dist_median, img_dist_median_id,\
                                    img_dist_max, img_dist_max_id = line
                    img_dist_sim = -1
                else:
                    img_path, img_label, img_pred,\
                        img_dist_target, img_dist_pred,\
                            img_dist_min, img_dist_min_id,\
                                img_dist_median, img_dist_median_id,\
                                    img_dist_max, img_dist_max_id,\
                                        img_dist_sim = line
                img_label = int(img_label)
                img_pred = int(img_pred)
                img_dist_target = float(img_dist_target)
                img_dist_pred = float(img_dist_pred)
                class_name_label = class_id2class_name[img_label]
                class_name_pred = class_id2class_name[img_pred]
                img_dist_min = float(img_dist_min)
                class_name_dist_min = class_id2class_name[int(img_dist_min_id)]
                img_dist_median = float(img_dist_median)
                class_name_dist_median = class_id2class_name[int(img_dist_median_id)]
                img_dist_max = float(img_dist_max)
                class_name_dist_max = class_id2class_name[int(img_dist_max_id)]
                img_dist_sim = float(img_dist_sim)
                img_name_info = "dist_GT_%.2f(%s)_Pred_%.2f(%s)_Sim_%.2f_min_%.2f(%s)_max_%.2f_med_%.2f"%(
                    img_dist_target, class_name_label,\
                        img_dist_pred, class_name_pred,\
                            img_dist_sim,\
                            img_dist_min, class_name_dist_min,\
                                img_dist_max,\
                                    img_dist_median
                )
                if not img_label in imgs_by_class:
                    imgs_by_class[img_label] = []
                imgs_by_class[img_label].append([img_path, img_dist_target, img_name_info])
                x_path.append(img_path)
                y_true.append(img_label)
                y_pred.append(img_pred)

        imgs_by_class_clean = {}
        imgs_by_relation_score = {}

        clean_csv_path = csv_path.replace(".csv", "_clean_flag.csv")
        assert(os.path.exists(clean_csv_path))
        with open(clean_csv_path, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            for idx, line in enumerate(csv_reader):
                if idx == 0:
                    continue
                if len(line) == 9:
                    img_path, img_label, img_pred,\
                        clean_idx, arcface_idx,\
                            rel_label, rel_pred,\
                                rel_max_class, rel_max = line
                elif len(line) == 8:
                    img_path, img_label, img_pred,\
                        clean_idx,\
                            rel_label, rel_pred,\
                                rel_max_class, rel_max = line
                    arcface_idx = 1
                else:
                    raise ValueError("wrong format of csv file lines")
                img_label = int(img_label)
                img_pred = int(img_pred)
                if not img_label in imgs_by_relation_score:
                    imgs_by_relation_score[img_label] = []
                imgs_by_relation_score[img_label].append([img_path, img_label, rel_label,\
                    img_pred, rel_pred, rel_max_class, rel_max])
                if not img_label in imgs_by_class_clean:
                    imgs_by_class_clean[img_label] = [[],[]]
                if clean_idx:
                    imgs_by_class_clean[img_label][0].append([img_path, img_pred])
                if arcface_idx:
                    imgs_by_class_clean[img_label][1].append([img_path, img_pred])

        for img_label in imgs_by_class:
            img_pathlist = imgs_by_class[img_label]
            img_pathlist_sorted = sorted(img_pathlist, key=lambda x:x[1], reverse=False)
            # images are sorted by their distance to the prototype feature
            assert(len(img_pathlist_sorted) == len(img_pathlist))
            imgs_by_class[img_label] = img_pathlist_sorted

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        x_path = np.array(x_path)
        ## 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        cm_rm_self = np.diag(np.ones(shape=cm.shape[0]) * (np.inf) * (-1)) + cm
        ## by definition cm_ij is equal to the number of observations 
        ## known to be in group i and predicted to be in group j
        ## 首先排除对角线上的每个元素类
        cm_total_err = np.sum(cm, axis=1) - np.diag(cm)
        ## 计算每个类别预测成最多的错误类
        cm_top_err = np.argmax(cm_rm_self, axis=1)
        ## 统计该种错误出现的次数confusion_matrix_ij
        cm_top_err_num = np.max(cm_rm_self, axis=1)
        save_csv_path = os.path.join(os.path.dirname(csv_path), "statistics_CM_" + os.path.basename(csv_path))
        with open(save_csv_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Confusion Matrix"])
            top_row = ['GT\Pred']
            for i in range(cm.shape[0]):
                top_row.append(class_id2class_name[i])
            csv_writer.writerow(top_row)
            for i in range(cm.shape[0]):
                row_i = [class_id2class_name[i]]
                for j in range(len(cm[i])):
                    row_i.append(cm[i][j])
                csv_writer.writerow(row_i)
        save_csv_path = os.path.join(os.path.dirname(csv_path), "statistics_top_" + os.path.basename(csv_path))
        with open(save_csv_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["GT id", "GT Name", "Total Error", "Pred Top 1 Error id", "Pred Top 1 Error id", "Pred Top 1 Error Num"])
            for i in range(len(cm_top_err)):
                pred_err_id = cm_top_err[i]
                row_i = [i, class_id2class_name[i], cm_total_err[i],\
                    pred_err_id, class_id2class_name[pred_err_id],\
                        cm_top_err_num[pred_err_id]]
                csv_writer.writerow(row_i)
        ## 统计出现次数最多的错误类别
        ## 从错误数目最高的前10类图像中抽取每类最容易预测错误的类别图像
        ## 以及每个数据集前20个类别的样本
        top_10_error_class = np.argsort(cm_top_err_num)[::-1][:10]
        if is_trainset:
            top_20_label_class = np.arange(20)
            top_classes = top_10_error_class.tolist() + top_20_label_class.tolist()
            top_classes = list(set(top_classes))
        else:
            top_classes = top_10_error_class
        ## 且给出每个类别对应图片靠prototype最近的前20张图以及最远的后20张图；
        ## 以及对应每个类别随机抽20张样本看整体分布情况
        ## 每张图都匹配一张真实的测试集样本作为参考
        save_img_path = os.path.join(os.path.dirname(csv_path), os.path.basename(csv_path).replace(".csv", ""))
        os.makedirs(save_img_path, exist_ok=True)
        for class_i in top_classes:
            ## 当前类别名
            class_name_i = class_id2class_name[class_i]
            ## 类别i很容易被模型预测错误的类别
            pred_err_i = cm_top_err[class_i]
            ## 类别i被预测错成其他类别的数目
            pred_err_i_num = cm_top_err_num[class_i]
            ## 预测错的类别名
            pred_err_name_i = class_id2class_name[pred_err_i]
            print("current class {} is wrongly predicted as class {} with {} times".format(
                class_name_i, pred_err_name_i, pred_err_i_num))
            ## 筛选出所有类别为class_i且被预测成pred_err_i的图像列表
            is_class_i = (y_true == class_i)
            is_pred_i = (y_pred == pred_err_i)
            is_selected = (is_class_i & is_pred_i)
            ## 首先从当前类别的图像中抽取典型图像
            imgs_class_i = imgs_by_class[class_i]
            save_img_class_i_path = os.path.join(save_img_path, class_name_i)
            os.makedirs(save_img_class_i_path, exist_ok=True)
            ## 然后从测试集类别图像中抽取正确图像
            imgs_gt_class_i = imgs_gt_by_class[class_i]
            ## 同样从测试集类别图像中抽取错误预测类别的图像
            imgs_err_class_i = imgs_gt_by_class[pred_err_i]
            img_dist_by_path = {x[0]:x[2] for x in imgs_class_i}
            ## 存储距离近的一些结果(10)
            N_top = 20
            for idx, img_item in enumerate(imgs_class_i[:N_top]):
                img_path, img_dist, save_name = img_item
                im = Image.open(img_path).convert("RGB")
                im_gt = Image.open(random.choice(imgs_gt_class_i)).convert("RGB")
                im_concat = get_concat_h_resize(im, im_gt)
                save_img_class_i_dist_path = os.path.join(save_img_class_i_path,
                "%s_idx_%d_R(test_gt).jpg"%(save_name, idx))
                im_concat.save(save_img_class_i_dist_path)
            ## 存储距离远的一些结果(-10)
            for idx, img_item in enumerate(imgs_class_i[-N_top:]):
                img_path, img_dist, save_name = img_item
                im = Image.open(img_path).convert("RGB")
                im_gt = Image.open(random.choice(imgs_gt_class_i)).convert("RGB")
                im_concat = get_concat_h_resize(im, im_gt)
                save_img_class_i_dist_path = os.path.join(save_img_class_i_path,
                "%s_idx_%d_R(test_gt).jpg"%(save_name, N_top+idx))
                im_concat.save(save_img_class_i_dist_path)
            ## 筛选对应预测错的的类别以及其对应prototype距离
            save_img_class_i_wrong_path = os.path.join(save_img_class_i_path, "errors")
            os.makedirs(save_img_class_i_wrong_path, exist_ok=True)
            if np.sum(is_selected) > 0:
                x_path_selected = x_path[is_selected]
                for idx, img_path in enumerate(x_path_selected[:N_top]):
                    dist_save_name = img_dist_by_path[img_path]
                    save_img_name_i = "L(GT)_R(Pred)_{}_{}.jpg".format(dist_save_name, idx)
                    im = Image.open(img_path).convert("RGB")
                    im_pred_gt = Image.open(random.choice(imgs_err_class_i)).convert("RGB")
                    im_concat = get_concat_h_resize(im, im_pred_gt)
                    im_concat.save(os.path.join(save_img_class_i_wrong_path, save_img_name_i))
            else:
                print("not enough images of class {} wrongly predicted as {}".format(class_name_i,\
                    pred_err_name_i))
            ## 存储relation module的输出结果
            save_img_class_i_relation_path = os.path.join(save_img_class_i_path, "relation_samples")
            os.makedirs(save_img_class_i_relation_path, exist_ok=True)
            imgs_all_relation_scores = imgs_by_relation_score[class_i]
            imgs_all_relation_scores = sorted(imgs_all_relation_scores, key=lambda x:x[2])
            relation_samples = imgs_all_relation_scores[:20] + imgs_all_relation_scores[-20:]
            for idx, sample_item in enumerate(relation_samples):
                img_path, img_label, rel_label, img_pred, rel_pred, rel_max_class, rel_max = sample_item
                save_img_name_i = "{}_GT_{}({})_Pred_{}({})_Max_{}({}).jpg".format(idx,\
                    rel_label, class_name_i,\
                        rel_pred, class_id2class_name[img_pred],\
                            rel_max, class_id2class_name[int(rel_max_class)])
                shutil.copyfile(img_path, os.path.join(save_img_class_i_relation_path, save_img_name_i))
            ## 存储被模型认为是干净样本和参与arcface classifier更新的样本
            save_img_class_i_clean_path = os.path.join(save_img_class_i_path, "clean_samples")
            save_img_class_i_arcface_path = os.path.join(save_img_class_i_path, "arcface_low-dim_cls_samples")
            os.makedirs(save_img_class_i_clean_path, exist_ok=True)
            os.makedirs(save_img_class_i_arcface_path, exist_ok=True)
            samples_clean_class_i = imgs_by_class_clean[class_i][0]
            samples_arcface_class_i = imgs_by_class_clean[class_i][1]
            samples_clean_class_i = random.sample(samples_clean_class_i, min(20, len(samples_clean_class_i)))
            samples_arcface_class_i = random.sample(samples_arcface_class_i, min(20, len(samples_arcface_class_i)))
            for idx, sample_clean in enumerate(samples_clean_class_i):
                img_path_idx, img_pred_idx = sample_clean
                save_img_name_i = "L(GT)_R(Pred)_{}.jpg".format(idx)
                im = Image.open(img_path_idx).convert("RGB")
                im_pred_gt = Image.open(random.choice(imgs_gt_by_class[img_pred_idx])).convert("RGB")
                im_concat = get_concat_h_resize(im, im_pred_gt)
                im_concat.save(os.path.join(save_img_class_i_clean_path, save_img_name_i))
            for idx, sample_clean in enumerate(samples_arcface_class_i):
                img_path_idx, img_pred_idx = sample_clean
                save_img_name_i = "L(GT)_R(Pred)_{}.jpg".format(idx)
                im = Image.open(img_path_idx).convert("RGB")
                im_pred_gt = Image.open(random.choice(imgs_gt_by_class[img_pred_idx])).convert("RGB")
                im_concat = get_concat_h_resize(im, im_pred_gt)
                im_concat.save(os.path.join(save_img_class_i_arcface_path, save_img_name_i))
        return class_id2class_name
    print("================Testing Samples================")
    class_id2class_name = calc_confusion_matrix(test_csv_path, class_id2class_name=None, is_trainset=False)
    ## write mapping from class id to class name
    save_path_mapping = os.path.join(root_dir, "class_id2name.txt")
    with open(save_path_mapping, "w") as f_write:
        for class_id, class_name in class_id2class_name.items():
            class_id_name_full = "{}@{}\n".format(class_id, class_name)
            f_write.write(class_id_name_full)
    print("================FewShot Samples================")
    _ = calc_confusion_matrix(fewshot_csv_path, class_id2class_name=class_id2class_name, is_trainset=True)
    print("================Training Samples================")
    _ = calc_confusion_matrix(train_csv_path, class_id2class_name=class_id2class_name, is_trainset=True)
    return


def read_json(json_path_list):
    """read json file and return data"""
    with open(json_path_list) as json_file:
        json_dict = json.load(json_file)
    return json_dict


def check_statistics_webvision(root_dir, target_domain_dir,\
    target_domain_imglist, fast_eval=0):
    ## 读取真实的测试集图像 按类别分类得到参考图
    imgs_gt_by_class = {}
    with open(target_domain_imglist, "r") as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip().split(" ")
            img_path = info[0]
            img_label = int(info[1])
            if not (img_label in imgs_gt_by_class):
                imgs_gt_by_class[img_label] = []
            imgs_gt_by_class[img_label].append(os.path.join(target_domain_dir, img_path))

    train_csv_path = os.path.join(root_dir, "WebTrain.csv")
    fewshot_csv_path = os.path.join(root_dir, "FewShot.csv")
    test_web_csv_path = os.path.join(root_dir, "WebVision.csv")
    test_img_csv_path = os.path.join(root_dir, "ImgNet.csv")
    def calc_confusion_matrix(csv_path, class_id2class_name=None, is_trainset=False):
        y_pred = []
        y_true = []
        x_path = []
        if class_id2class_name is None:
            ## 重新构建字典来保存映射
            class_id2class_name = {}
            if fast_eval == 2:
                class_id2wdnet_id_path = "../imglists/mapping_google_500.txt"
            else:
                class_id2wdnet_id_path = "../imglists/mapping_webvision_1k.txt"
            imgnet_info = "../imagenet_class_1k_full_index.json"
            _, _, img_by_wdnet = read_json(imgnet_info)
            with open(class_id2wdnet_id_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    info = line.strip().split(" ")
                    class_id = int(info[0])
                    wdnet_id = info[1]
                    img_info = img_by_wdnet[wdnet_id]
                    if len(img_info) > 2:
                        class_name = img_info[2].replace(" ", "_")
                    else:
                        class_name = img_info[0].replace(" ", "_")
                    class_id2class_name[class_id] = class_name

        ## path_to_val_folder/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg,0,144
        imgs_by_class = {}
        with open(csv_path, "r") as f:
            csv_reader = csv.reader(f)
            for idx, line in enumerate(csv_reader):
                if idx == 0:
                    continue
                if len(line) == 11:
                    img_path, img_label, img_pred,\
                        img_dist_target, img_dist_pred,\
                            img_dist_min, img_dist_min_id,\
                                img_dist_median, img_dist_median_id,\
                                    img_dist_max, img_dist_max_id = line
                    img_dist_sim = -1
                else:
                    img_path, img_label, img_pred,\
                        img_dist_target, img_dist_pred,\
                            img_dist_min, img_dist_min_id,\
                                img_dist_median, img_dist_median_id,\
                                    img_dist_max, img_dist_max_id,\
                                        img_dist_sim = line

                img_label = int(img_label)
                img_pred = int(img_pred)
                img_dist_target = float(img_dist_target)
                img_dist_pred = float(img_dist_pred)
                class_name_label = class_id2class_name[img_label]
                class_name_pred = class_id2class_name[img_pred]
                img_dist_min = float(img_dist_min)
                class_name_dist_min = class_id2class_name[int(img_dist_min_id)]
                img_dist_median = float(img_dist_median)
                class_name_dist_median = class_id2class_name[int(img_dist_median_id)]
                img_dist_max = float(img_dist_max)
                class_name_dist_max = class_id2class_name[int(img_dist_max_id)]
                img_name_info = "dist_GT_%.2f(%s)_Pred_%.2f(%s)_Sim_%.2f_min_%.2f(%s)_max_%.2f_med_%.2f"%(
                    img_dist_target, class_name_label,\
                        img_dist_pred, class_name_pred,\
                            float(img_dist_sim),\
                            img_dist_min, class_name_dist_min,\
                                img_dist_max,\
                                    img_dist_median
                )
                if not img_label in imgs_by_class:
                    imgs_by_class[img_label] = []
                imgs_by_class[img_label].append([img_path, img_dist_target, img_name_info])
                x_path.append(img_path)
                y_true.append(img_label)
                y_pred.append(img_pred)

        imgs_by_class_clean = {}
        imgs_by_relation_score = {}

        clean_csv_path = csv_path.replace(".csv", "_clean_flag.csv")
        assert(os.path.exists(clean_csv_path))
        with open(clean_csv_path, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            for idx, line in enumerate(csv_reader):
                if idx == 0:
                    continue
                if len(line) == 9:
                    img_path, img_label, img_pred,\
                        clean_idx, arcface_idx,\
                            rel_label, rel_pred,\
                                rel_max_class, rel_max = line
                elif len(line) == 8:
                    img_path, img_label, img_pred,\
                        clean_idx,\
                            rel_label, rel_pred,\
                                rel_max_class, rel_max = line
                    arcface_idx = 1
                else:
                    raise ValueError("wrong format of csv file lines")
                img_label = int(img_label)
                img_pred = int(img_pred)
                if not img_label in imgs_by_relation_score:
                    imgs_by_relation_score[img_label] = []
                imgs_by_relation_score[img_label].append([img_path, img_label, rel_label,\
                    img_pred, rel_pred, rel_max_class, rel_max])
                if not img_label in imgs_by_class_clean:
                    imgs_by_class_clean[img_label] = [[],[]]
                if clean_idx:
                    imgs_by_class_clean[img_label][0].append([img_path, img_pred])
                if arcface_idx:
                    imgs_by_class_clean[img_label][1].append([img_path, img_pred])

        for img_label in imgs_by_class:
            img_pathlist = imgs_by_class[img_label]
            img_pathlist_sorted = sorted(img_pathlist, key=lambda x:x[1], reverse=False)
            # images are sorted by their distance to the prototype feature
            assert(len(img_pathlist_sorted) == len(img_pathlist))
            imgs_by_class[img_label] = img_pathlist_sorted

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        x_path = np.array(x_path)
        ## 计算混淆矩阵
        if fast_eval == 2:
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(500))
        else:
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(1000))
        cm_rm_self = np.diag(np.ones(shape=cm.shape[0]) * (np.inf) * (-1)) + cm
        ## by definition cm_ij is equal to the number of observations 
        ## known to be in group i and predicted to be in group j
        ## 首先排除对角线上的每个元素类
        cm_total_err = np.sum(cm, axis=1) - np.diag(cm)
        ## 计算每个类别预测成最多的错误类
        cm_top_err = np.argmax(cm_rm_self, axis=1)
        ## 统计该种错误出现的次数confusion_matrix_ij
        cm_top_err_num = np.max(cm_rm_self, axis=1)
        save_csv_path = os.path.join(os.path.dirname(csv_path), "statistics_CM_" + os.path.basename(csv_path))
        with open(save_csv_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Confusion Matrix"])
            top_row = ['GT\Pred']
            for i in range(cm.shape[0]):
                top_row.append(class_id2class_name[i])
            csv_writer.writerow(top_row)
            for i in range(cm.shape[0]):
                row_i = [class_id2class_name[i]]
                for j in range(len(cm[i])):
                    row_i.append(cm[i][j])
                csv_writer.writerow(row_i)
        save_csv_path = os.path.join(os.path.dirname(csv_path), "statistics_top_" + os.path.basename(csv_path))
        with open(save_csv_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["GT id", "GT Name", "Total Error", "Pred Top 1 Error id", "Pred Top 1 Error id", "Pred Top 1 Error Num"])
            for i in range(len(cm_top_err)):
                pred_err_id = cm_top_err[i]
                row_i = [i, class_id2class_name[i], cm_total_err[i],\
                    pred_err_id, class_id2class_name[pred_err_id],\
                        cm_top_err_num[pred_err_id]]
                csv_writer.writerow(row_i)
        ## 统计出现次数最多的错误类别
        ## 从错误数目最高的前10类图像中抽取每类最容易预测错误的类别图像
        ## 以及每个数据集前20个类别的样本 + 歧义类别(鼓槌\手推车\鸟\起重机\钉子)
        top_10_error_class = np.argsort(cm_top_err_num)[::-1][:10]
        class_must_include = []
        if fast_eval != 2:
            ### webvision
            class_must_include = [542, 428, 134, 517, 677, 998]
            top_10_error_class = top_10_error_class.tolist() + class_must_include
        else:
            ### google500
            class_must_include = [137, 246, 385, 227]
            top_10_error_class = top_10_error_class.tolist() + class_must_include
        if is_trainset:
            top_20_label_class = np.arange(20)
            top_classes = top_10_error_class + top_20_label_class.tolist()
            top_classes = list(set(top_classes))
        else:
            top_classes = top_10_error_class
        ## 且给出每个类别对应图片靠prototype最近的前20张图以及最远的后20张图；
        ## 以及对应每个类别随机抽20张样本看整体分布情况
        ## 每张图都匹配一张真实的测试集样本作为参考
        save_img_path = os.path.join(os.path.dirname(csv_path), os.path.basename(csv_path).replace(".csv", ""))
        os.makedirs(save_img_path, exist_ok=True)
        for class_i in top_classes:
            ## 当前类别名
            class_name_i = str(class_i) + "_" + class_id2class_name[class_i]
            ## 类别i很容易被模型预测错误的类别
            pred_err_i = cm_top_err[class_i]
            ## 类别i被预测错成其他类别的数目
            pred_err_i_num = cm_top_err_num[class_i]
            ## 预测错的类别名
            pred_err_name_i = class_id2class_name[pred_err_i]
            print("current class {} is wrongly predicted as class {} with {} times".format(
                class_name_i, pred_err_name_i, pred_err_i_num))
            ## 筛选出所有类别为class_i且被预测成pred_err_i的图像列表
            is_class_i = (y_true == class_i)
            is_pred_i = (y_pred == pred_err_i)
            is_selected = (is_class_i & is_pred_i)
            ## 首先从当前类别的图像中抽取典型图像
            imgs_class_i = imgs_by_class[class_i]
            save_img_class_i_path = os.path.join(save_img_path, class_name_i)
            os.makedirs(save_img_class_i_path, exist_ok=True)
            ## 然后从测试集类别图像中抽取正确图像
            imgs_gt_class_i = imgs_gt_by_class[class_i]
            ## 同样从测试集类别图像中抽取错误预测类别的图像
            imgs_err_class_i = imgs_gt_by_class[pred_err_i]
            img_dist_by_path = {x[0]:x[2] for x in imgs_class_i}
            ## 存储距离近的一些结果(10)
            if class_i in class_must_include:
                print("visiting class {} with all instances".format(class_i))
                N_top = len(imgs_class_i)//2
            else:
                N_top = 40
            for idx, img_item in enumerate(imgs_class_i[:N_top]):
                img_path, img_dist, save_name = img_item
                tf_path, tf_offset = img_path.split("@")
                im = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                gt_path = random.choice(imgs_gt_class_i)
                tf_path, tf_offset = gt_path.split("@")
                im_gt = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                im_concat = get_concat_h_resize(im, im_gt)
                save_img_class_i_dist_path = os.path.join(save_img_class_i_path,
                "%s_idx_%d_R(test_gt).jpg"%(save_name, idx))
                im_concat.save(save_img_class_i_dist_path)
            ## 存储距离远的一些结果(-10)
            for idx, img_item in enumerate(imgs_class_i[-N_top:]):
                img_path, img_dist, save_name = img_item
                tf_path, tf_offset = img_path.split("@")
                im = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                gt_path = random.choice(imgs_gt_class_i)
                tf_path, tf_offset = gt_path.split("@")
                im_gt = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                im_concat = get_concat_h_resize(im, im_gt)
                save_img_class_i_dist_path = os.path.join(save_img_class_i_path,
                "%s_idx_%d_R(test_gt).jpg"%(save_name, N_top+idx))
                im_concat.save(save_img_class_i_dist_path)
            ## 筛选对应预测错的的类别以及其对应prototype距离
            save_img_class_i_wrong_path = os.path.join(save_img_class_i_path, "errors")
            os.makedirs(save_img_class_i_wrong_path, exist_ok=True)
            if np.sum(is_selected) > 0:
                x_path_selected = x_path[is_selected]
                for idx, img_path in enumerate(x_path_selected[:N_top]):
                    dist_save_name = img_dist_by_path[img_path]
                    save_img_name_i = "L(GT)_R(Pred)_{}_{}.jpg".format(dist_save_name, idx)
                    tf_path, tf_offset = img_path.split("@")
                    im = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                    gt_path = random.choice(imgs_err_class_i)
                    tf_path, tf_offset = gt_path.split("@")
                    im_pred_gt = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                    im_concat = get_concat_h_resize(im, im_pred_gt)
                    im_concat.save(os.path.join(save_img_class_i_wrong_path, save_img_name_i))
            else:
                print("not enough images of class {} wrongly predicted as {}".format(class_name_i,\
                    pred_err_name_i))
            ## 存储relation module的输出结果
            save_img_class_i_relation_path = os.path.join(save_img_class_i_path, "relation_samples")
            os.makedirs(save_img_class_i_relation_path, exist_ok=True)
            imgs_all_relation_scores = imgs_by_relation_score[class_i]
            imgs_all_relation_scores = sorted(imgs_all_relation_scores, key=lambda x:x[2])
            relation_samples = imgs_all_relation_scores[:20] + imgs_all_relation_scores[-20:]
            for idx, sample_item in enumerate(relation_samples):
                img_path, img_label, rel_label, img_pred, rel_pred, rel_max_class, rel_max = sample_item
                save_img_name_i = "{}_GT_{}({})_Pred_{}({})_Max_{}({}).jpg".format(idx,\
                    rel_label, class_name_i,\
                        rel_pred, class_id2class_name[img_pred],\
                            rel_max, class_id2class_name[int(rel_max_class)])
                tf_path, tf_offset = img_path.split("@")
                im = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                im.save(os.path.join(save_img_class_i_relation_path, save_img_name_i))
            ## 存储被模型认为是干净样本和参与arcface classifier更新的样本
            save_img_class_i_clean_path = os.path.join(save_img_class_i_path, "clean_samples")
            save_img_class_i_arcface_path = os.path.join(save_img_class_i_path, "arcface_low-dim_cls_samples")
            os.makedirs(save_img_class_i_clean_path, exist_ok=True)
            os.makedirs(save_img_class_i_arcface_path, exist_ok=True)
            samples_clean_class_i = imgs_by_class_clean[class_i][0]
            samples_arcface_class_i = imgs_by_class_clean[class_i][1]
            samples_clean_class_i = random.sample(samples_clean_class_i, min(20, len(samples_clean_class_i)))
            samples_arcface_class_i = random.sample(samples_arcface_class_i, min(20, len(samples_arcface_class_i)))
            for idx, sample_clean in enumerate(samples_clean_class_i):
                img_path_idx, img_pred_idx = sample_clean
                save_img_name_i = "L(GT)_R(Pred)_{}.jpg".format(idx)
                tf_path, tf_offset = img_path_idx.split("@")
                im = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                gt_path = random.choice(imgs_gt_by_class[img_pred_idx])
                tf_path, tf_offset = gt_path.split("@")
                im_pred_gt = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                im_concat = get_concat_h_resize(im, im_pred_gt)
                im_concat.save(os.path.join(save_img_class_i_clean_path, save_img_name_i))
            for idx, sample_clean in enumerate(samples_arcface_class_i):
                img_path_idx, img_pred_idx = sample_clean
                save_img_name_i = "L(GT)_R(Pred)_{}.jpg".format(idx)
                tf_path, tf_offset = img_path_idx.split("@")
                im = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                gt_path = random.choice(imgs_gt_by_class[img_pred_idx])
                tf_path, tf_offset = gt_path.split("@")
                im_pred_gt = get_tfrecord_image(tf_path, int(tf_offset)).convert("RGB")
                im_concat = get_concat_h_resize(im, im_pred_gt)
                im_concat.save(os.path.join(save_img_class_i_arcface_path, save_img_name_i))
        return class_id2class_name
    
    save_path_mapping = os.path.join(root_dir, "class_id2name.txt")
    if os.path.exists(save_path_mapping):
        class_id2class_name = {}
        with open(save_path_mapping, "r") as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split("@")
                class_id = int(info[0])
                class_name = info[1]
                class_id2class_name[class_id] = class_name
    print("================Testing Samples webvision================")
    class_id2class_name = calc_confusion_matrix(test_web_csv_path, class_id2class_name=None, is_trainset=False)
    ## write mapping from class id to class name
    with open(save_path_mapping, "w") as f_write:
        for class_id, class_name in class_id2class_name.items():
            class_id_name_full = "{}@{}\n".format(class_id, class_name)
            f_write.write(class_id_name_full)
    print("================Testing Samples imgnet================")
    class_id2class_name = calc_confusion_matrix(test_img_csv_path, class_id2class_name=class_id2class_name, is_trainset=False)
    print("================Training Samples================")
    class_id2class_name = calc_confusion_matrix(train_csv_path, class_id2class_name=class_id2class_name, is_trainset=True)
    print("================FewShot Samples================")
    _ = calc_confusion_matrix(fewshot_csv_path, class_id2class_name=class_id2class_name, is_trainset=False)
    return


if __name__ == '__main__':
    print()
    main()
