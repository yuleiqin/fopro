#!/usr/bin/env python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import json
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
from utils.lr_scheduler_webFG import lr_scheduler as lr_scheduler_webFG
# from resnet import *
# import DataLoader.dataloader_balance as dataloader
# import DataLoader.dataloader as dataloader
from model import init_weights
from backbone.basenet import AlexNet_Encoder, VGG_Encoder, BCNN_encoder
from backbone.resnet import resnet50
from backbone.classifier import Normalize, MLP_classifier
import DataLoader.webFG_dataset as webFG496
import DataLoader.webvision_dataset as webvision
from config_train import parser

import tensorboard_logger as tb_logger

import numpy as np


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

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    return


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass    

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
  
    # create model
    print("=> creating {} model".format(args.arch))
    if args.arch == 'resnet50':
        ### this is the default
        encoder = resnet50(pretrained=args.pretrained, width=1)
    elif args.arch == 'resnet50x2':
        encoder = resnet50(pretrained=args.pretrained, width=2)
    elif args.arch == 'resnet50x4':
        encoder = resnet50(pretrained=args.pretrained, width=4)
    elif args.arch == 'vgg':
        encoder = VGG_Encoder(pretrained=args.pretrained)
    elif args.arch == 'bcnn':
        encoder = BCNN_encoder(pretrained=args.pretrained, num_out_channel=512**2)
    elif args.arch == 'alexnet':
        encoder = AlexNet_Encoder(pretrained=args.pretrained)
    else:
        raise NotImplementedError('model not supported {}'.format(args.arch))
    classifier = MLP_classifier(num_class=args.num_class,\
        in_channel=encoder.num_out_channel, use_norm=False)
    classifier.apply(init_weights)
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            encoder.cuda(args.gpu)
            classifier.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)

    ## args.resume用于加载已经训练好的feature extractor
    assert(os.path.exists(args.resume) and os.path.isfile(args.resume))
    if args.gpu is None:
        checkpoint = torch.load(args.resume)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)    
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q'):
            # remove prefix
            state_dict[k.replace('.encoder_q','')] = state_dict[k]
        elif k.startswith('module.classifier'):
            # keep as it is
            state_dict[k.replace('.classifier','')] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    print("=> loaded feature encoder checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
    encoder_dict = encoder.state_dict()
    state_dict_enc = {k:v for k, v in state_dict.items() if k in encoder_dict}
    encoder_dict.update(state_dict_enc)
    encoder.load_state_dict(encoder_dict, strict=True)
    classifier_dict = classifier.state_dict()
    state_dict_cls = {k:v for k, v in state_dict.items() if k in classifier_dict}
    classifier_dict.update(state_dict_cls)
    classifier.load_state_dict(classifier_dict, strict=True)

    ## resume from a checkpoint
    ## 加载已经训练了部分epochs的classifier
    resume_path = '{}/checkpoint_latest.tar'.format(args.exp_dir)
    if os.path.exists(resume_path) and os.path.isfile(resume_path):
        if args.gpu is None:
            checkpoint = torch.load(resume_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(resume_path, map_location=loc)
        print("=> loaded classifier checkpoint '{}' (epoch {})".format(
            resume_path, checkpoint['epoch'],
        ))
        state_dict = checkpoint['state_dict']
        classifier_dict = classifier.state_dict()
        state_dict_cls = {k:v for k, v in state_dict.items() if k in classifier_dict}
        classifier_dict.update(state_dict_cls)
        # print("classifier state dict ", classifier_dict.keys())
        classifier.load_state_dict(classifier_dict, strict=True)
        args.start_epoch = checkpoint['epoch']
        if args.webvision:
            if 'best_acc_web' in checkpoint:
                acc_max_web = checkpoint['best_acc_web']
            else:
                acc_max_web = 0
            if 'best_acc_imgnet' in checkpoint:
                acc_max_imgnet = checkpoint['best_acc_imgnet']
            else:
                acc_max_imgnet = 0
        else:
            if 'best_acc' in checkpoint:
                acc_max = checkpoint['best_acc']
            else:
                acc_max = 0
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))
        if args.webvision:
            acc_max_web, acc_max_imgnet = 0, 0
        else:
            acc_max = 0
    
    cudnn.benchmark = True
    
    if args.finetune:
        print("Optimize encoder and classifier simultaneously")
        optimizer_encoder = torch.optim.SGD(encoder.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer_encoder = None    
    ## 这部分代码仅仅训练的是分类器
    ## 优化器选择与学习率衰减策略根据数据集差异有所不同
    ## retrain阶段全部使用SGD优化器
    # if args.webvision
    print("use SGD optimizer")
    optimizer = torch.optim.SGD(classifier.parameters(), args.lr,
    momentum=args.momentum, weight_decay=args.weight_decay)
    def adjust_learning_rate(optimizer, epoch, args, is_classifier=True):
        """Decay the learning rate based on schedule"""
        if is_classifier:
            lr = args.lr
        else:
            lr = args.lr * 0.1
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return

    if args.resume:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Load optimizer success")
        except:
            print("Load optimizer failed")
    # Data loading code
    assert(os.path.exists(args.root_dir)), "please make sure the path to web data is valid {}".format(args.root_dir)
    assert(os.path.exists(args.root_dir_t)), "please make sure the path to fewshot target domain data is valid {}".format(args.root_dir_t)
    assert(os.path.isfile(args.pathlist_t)), "please make sure the pathlist path to fewshot target domain data is valid {}".format(args.pathlist_t)
    assert(os.path.exists(args.annotation)), "please make sure the pathlist path to pseudo label json is valid {}".format(args.annotation)
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
                                        use_fewshot=args.use_fewshot,\
                                            annotation=args.annotation)
        train_loader, _, test_loader_web, test_loader_target = loader.run() 
    else:
        ## load webFG496 dataset
        loader = webFG496.webFG496_dataloader(batch_size=args.batch_size, num_class=args.num_class,\
            num_workers=args.workers, root_dir=args.root_dir, distributed=args.distributed, crop_size=0.8,\
                root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                    save_dir=args.exp_dir, dry_run=args.dry_run,\
                        use_fewshot=args.use_fewshot, annotation=args.annotation)
        train_loader, _, test_loader = loader.run()
    
    if args.gpu==0:
        logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir,'tensorboard'), flush_secs=2)
    else:
        logger = None

    class_weight = extract_class_weight(annotation=args.annotation, N_class=args.num_class)
    class_weight = torch.Tensor(class_weight).cuda(args.gpu)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            loader.train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        if args.finetune:
            adjust_learning_rate(optimizer_encoder, epoch, args, is_classifier=False)
        ## 仅仅训练分类器
        train(train_loader, encoder, classifier, criterion, optimizer, optimizer_encoder,\
            epoch, args, logger, class_weight)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if args.webvision:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': classifier.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'best_acc_web': acc_max_web,
                    'best_acc_imgnet': acc_max_imgnet
                }, is_best=False, filename='{}/checkpoint_latest.tar'.format(args.exp_dir))
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': classifier.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'best_acc': acc_max
                }, is_best=False, filename='{}/checkpoint_latest.tar'.format(args.exp_dir))
        ## 验证集跑eval结果
        if args.webvision:
            ## test webvision dataset
            acc1_web, acc5_web = test(encoder, classifier, test_loader_web, args, epoch, logger, dataset_name="WebVision")
            acc1_imgnet, acc5_imgnet = test(encoder, classifier, test_loader_target, args, epoch, logger, dataset_name="ImgNet")
            if acc1_web > acc_max_web:
                acc_max_web = acc1_web
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc_web': [acc1_web, acc5_web],
                        'best_acc_imgnet': [acc1_imgnet, acc5_imgnet]
                    }, is_best=False, filename='{}/checkpoint_best_web.tar'.format(args.exp_dir))                
            if acc1_imgnet > acc_max_imgnet:
                acc_max_imgnet = acc1_imgnet
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc_web': [acc1_web, acc5_web],
                        'best_acc_imgnet': [acc1_imgnet, acc5_imgnet]
                    }, is_best=False, filename='{}/checkpoint_best_imgnet.tar'.format(args.exp_dir)) 
        else:
            ## test webFineGrained dataset
            acc1, acc5 = test(encoder, classifier, test_loader, args, epoch, logger, dataset_name="FineGrained")
            if acc1 > acc_max:
                acc_max = acc1
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': [acc1, acc5]
                    }, is_best=False, filename='{}/checkpoint_best.tar'.format(args.exp_dir))
        if args.webvision:
            print("accuracy top 1 web = {} top 1 imagenet = {}".format(acc_max_web, acc_max_imgnet))
        else:
            print("accuracy top 1 = {}".format(acc_max))
    return


def extract_class_weight(annotation, N_class):
    weights_all = [1.0 for _ in range(N_class)]
    with open(annotation, "r") as f:
        json_file = json.load(f)
    targets = json_file["targets"]
    for target in targets:
        weights_all[int(target)] += 1.
    count_mean = np.mean(np.array(weights_all))
    weights_all = [float(count_mean)/float(weight) for weight in weights_all]
    return weights_all


def train(train_loader, encoder, classifier, criterion,\
    optimizer, optimizer_encoder, epoch, args, tb_logger, class_weight):
    if args.rebalance:
        ## 对样本进行重采样并选取固定大小
        # train_loader.dataset.repeat()
        train_loader.dataset.resample()
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    losses_cls = AverageMeter('Loss@Cls', ':2.2f')
    acc_cls = AverageMeter('Acc@Cls', ':4.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_cls, acc_cls],
        prefix="Epoch: [{}]".format(epoch))

    if args.finetune:
        # finetune encoder backbone
        encoder.train()
    else:
        encoder.eval()
    classifier.train()
    
    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = batch[0]
        target = batch[1]
        img = img.cuda(args.gpu, non_blocking=True) 
        target = target.cuda(args.gpu, non_blocking=True)
        prob_mixup = np.random.random()
        ###check if needs mix-up augmentation
        if args.mixup and (prob_mixup > 0.5):
            lam_mixup = np.random.beta(0.8, 0.8)
            rand_index = torch.randperm(img.size()[0]).cuda()
            target_mix_up = target[rand_index]
            img = lam_mixup * img + (1.0 - lam_mixup) * img[rand_index, :]

        if args.finetune:
            feature = encoder(img)
        else:
            with torch.no_grad():
                feature = encoder(img)
        output = classifier(feature)

        loss = criterion(output, target)
        if args.webvision:
            sample_weight = torch.index_select(class_weight, dim=0, index=target.view(-1).type(torch.int64))
            loss *= sample_weight

        if args.mixup and (prob_mixup > 0.5):
            loss *= lam_mixup
            loss_mixup = criterion(output, target_mix_up)*(1. - lam_mixup)
            if args.webvision:
                loss_mixup *= sample_weight
            loss += loss_mixup

        if args.use_soft_label:
            target_soft = F.softmax(output.detach().clone(), dim=1)
            gt_score = target_soft[target>=0, target]
            loss_cls_soft = - torch.sum(target_soft * F.log_softmax(output, dim=1), dim=1)*(1-gt_score)
            if args.webvision:
                loss_cls_soft *= sample_weight
            loss *= gt_score
            loss += loss_cls_soft

        loss = loss.mean()
        losses_cls.update(loss.item())

        acc = accuracy(output, target)[0] 
        acc_cls.update(acc[0])
        
        # compute gradient and do SGD step
        if args.finetune:
            optimizer_encoder.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.finetune:
            optimizer_encoder.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
            
    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
    return


def test(encoder, classifier, test_loader, args, epoch, tb_logger, dataset_name="WebVision"):
    with torch.no_grad():
        print('==> Evaluation...')     
        encoder.eval()
        classifier.eval()
        top1_acc = AverageMeter("Top1@{}".format(dataset_name))
        top5_acc = AverageMeter("Top5@{}".format(dataset_name))
        
        # evaluate on webvision val set
        for batch_idx, batch in enumerate(test_loader):
            ## outputs, feat, target, feat_reconstruct
            img = batch[0]
            target = batch[1]
            img = img.cuda(args.gpu, non_blocking=True) 
            target = target.cuda(args.gpu, non_blocking=True)
            feature = encoder(img)
            outputs = classifier(feature)
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])
        
        # average across all processes
        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)
        dist.all_reduce(acc_tensors)        
        acc_tensors /= args.world_size
        
        print('%s Accuracy is %.2f%% (%.2f%%)'%(dataset_name,\
            acc_tensors[0],acc_tensors[1]))
        if args.gpu ==0:
            tb_logger.log_value('{} top1 Acc'.format(dataset_name),\
                acc_tensors[0], epoch)
            tb_logger.log_value('{} top5 Acc'.format(dataset_name),\
                acc_tensors[1], epoch)             
    return acc_tensors[0].item(), acc_tensors[1].item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    return


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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) + 1e-7
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k, :].sum(0)
            correct_k = correct_k.view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    
if __name__ == '__main__':
    main()
