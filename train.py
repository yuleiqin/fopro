## python library
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
## pytorch library
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
from tqdm import tqdm
## model python files
from config_train import parser
from utils.lr_scheduler_webFG import lr_scheduler as lr_scheduler_webFG
from model import MoPro, init_weights
# import DataLoader.dataloader as dataloader
import DataLoader.webFG_dataset as webFG496
import DataLoader.webvision_dataset as webvision
from utils.weight_class import extract_class_weight
import tensorboard_logger as tb_logger

import warnings
warnings.filterwarnings('ignore')


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
    print("distributed training {}".format(args.distributed))
    ## prepare the directory for saving
    os.makedirs(args.exp_dir, exist_ok=True)
    args.tensorboard_dir = os.path.join(args.exp_dir, 'tensorboard')
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("{} worldsize for multiprocessing distributed with {} gpus".format(args.world_size,\
            ngpus_per_node))
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("{} single process running with {} gpus".format(args.world_size,\
            ngpus_per_node))
        main_worker(args.gpu, ngpus_per_node, args)
    return


def change_status_encoder(model, requires_grad=True):
    """更改encoder参数是否需要梯度更新
    """
    updated_parameters_names = []
    for name_p, p in model.named_parameters():
        if name_p.startswith("module.encoder_q"):
            p.requires_grad = requires_grad
        if p.requires_grad:
            updated_parameters_names.append(name_p)
    params = [p for p in model.parameters() if p.requires_grad]
    print("Set encoder requires_grad = {}".format(requires_grad))
    print("Updated parameter names", updated_parameters_names)
    return model, params


def change_status_relation(model, requires_grad=True):
    """更改非relation module的参数是否需要梯度更新
    """
    updated_parameters_names = []
    for name_p, p in model.named_parameters():
        if not (name_p.startswith("module.relation")):
            p.requires_grad = requires_grad
        if p.requires_grad:
            updated_parameters_names.append(name_p)
    params = [p for p in model.parameters() if p.requires_grad]
    print("Set modules except relation module requires_grad = {}".format(requires_grad))
    print("Updated parameter names", updated_parameters_names)
    return model, params


def optimize_encoder(model, args):
    """将encoder部分参数设置为需要梯度更新 同时将其参数加入optimizer进行优化
    """
    model, params = change_status_encoder(model, True)
    if args.webvision:
        ## webvision/google500
        optimizer = torch.optim.SGD(params, args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        ## webFineGrained496
        optimizer = torch.optim.Adam(params, args.lr,
        weight_decay=args.weight_decay)
    return model, optimizer


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))    
        
    ## suppress printing if not master
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
    ## create model
    print("=> creating model '{}'".format(args.arch))
    model = MoPro(args)
    if not args.pretrained:
        ## 如果不使用预训练参数则需要随机初始化
        model.apply(init_weights)
    if args.gpu == 0:
        print(model)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    ## 方便使用权重对样本进行加权计算损失
    criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)
    ## 优化器选择与学习率衰减策略根据数据集差异有所不同
    if args.frozen_encoder_epoch != 0:
        ## 保证feature encoder backbone不变微调classifier
        model, params = change_status_encoder(model, False)
    else:
        ## all model parameters need to be optimized
        params = model.parameters()
    if hasattr(args, "ft_relation") and args.ft_relation:
        model, params = change_status_relation(model, False)
    if args.webvision and not (args.ft_relation):
        ## webvision/google500
        optimizer = torch.optim.SGD(params, args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
        if not (args.cos):
            for milestone in args.schedule:
                print(milestone)
        def adjust_learning_rate(optimizer, epoch, args):
            """Decay the learning rate based on schedule"""
            lr = args.lr
            if args.warmup_epoch != 0 and epoch <= args.warmup_epoch:
                lr *= (max(epoch, 1.0)/args.warmup_epoch)
            else:
                if args.cos:  # cosine lr schedule
                    lr *= 0.5 * (1. + math.cos(math.pi * (epoch-args.warmup_epoch)/(args.epochs-args.warmup_epoch)))
                else:
                    ## stepwise lr schedule
                    for milestone in args.schedule:
                        lr *= 0.1 if epoch >= milestone else 1.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return
    else:
        ## webFineGrained496
        optimizer = torch.optim.Adam(params, args.lr,
        weight_decay=args.weight_decay)
        mom1 = 0.9
        mom2 = 0.1
        epoch_decay_start = 40
        warmup_epochs = args.warmup_epoch
        alpha_plan = lr_scheduler_webFG(args.lr, args.epochs+1,\
            warmup_end_epoch=warmup_epochs, mode='cosine')
        beta1_plan = [mom1] * (args.epochs+1)
        for i in range(epoch_decay_start, args.epochs+1):
            beta1_plan[i] = mom2
        def adjust_learning_rate(optimizer, epoch, args):
            for param_group in optimizer.param_groups:
                param_group['lr'] = alpha_plan[epoch]
                param_group['betas'] = (beta1_plan[epoch], 0.999)  # only change beta1
            return
    ## optionally resume from a checkpoint
    resume_path = '{}/checkpoint_latest.tar'.format(args.exp_dir)
    resume_continue = True
    if not os.path.exists(resume_path):
        resume_path = args.resume
        resume_continue = False
    if os.path.exists(resume_path) and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        if args.gpu is None:
            checkpoint = torch.load(resume_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(resume_path, map_location=loc)
        ## 如果是仅微调relation module的时候就从0开始训练
        ## 且不加载relation模块的参数
        if hasattr(args, "ft_relation") and args.ft_relation and (not resume_continue):
            model_dict =  model.state_dict()
            state_dict = checkpoint['state_dict']
            state_dict = {k:v for k, v in state_dict.items() if not ("relation" in k)}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
        elif args.pre_relation and (not resume_continue):
            ## 只能加载预训练参数而不是继续训练
            print("LOAD PRETRAINED PARAMETERS")
            model_dict =  model.state_dict()
            state_dict = checkpoint['state_dict']
            state_dict = {k:v for k, v in state_dict.items() if k in model_dict and (model_dict[k].size() == v.size())}
            print("Succesfully Loader Parameters Include", state_dict.keys())
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
        else:
            ## 阶段1&阶段3训练
            model.load_state_dict(checkpoint['state_dict'])
            if checkpoint['epoch'] >= args.start_epoch and resume_continue:
                ## 仅当checkpoint中的epoch数目大于初始值时才更新
                args.start_epoch = checkpoint['epoch']
            if args.pretrained and args.start_epoch > args.frozen_encoder_epoch and (not args.ft_relation):
                ## finetune所有参数, 将optimizer进行替换
                print("optimizer all encoder")
                model, optimizer = optimize_encoder(model, args)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Load optimizer success")
            except:
                print("Load optimizer failed")
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, checkpoint['epoch']))
        if args.webvision:
            if "best_acc_web" in checkpoint:
                acc_max_web = checkpoint["best_acc_web"]
                if type(acc_max_web) is list and len(acc_max_web) > 0:
                    acc_max_web = acc_max_web[0]
            else:
                acc_max_web = 0
            if "best_acc_imgnet" in checkpoint:
                acc_max_imgnet = checkpoint["best_acc_imgnet"]
                if type(acc_max_imgnet) is list and len(acc_max_imgnet) > 0:
                    acc_max_imgnet = acc_max_imgnet[0]
            else:
                acc_max_imgnet = 0
            if 'epoch_best_web' in checkpoint:
                epoch_best_web = checkpoint["epoch_best_web"]
            else:
                epoch_best_web = 0
            if 'epoch_best_imgnet' in checkpoint:
                epoch_best_imgnet = checkpoint["epoch_best_imgnet"]
            else:
                epoch_best_imgnet = 0
        else:
            if "best_acc" in checkpoint:
                acc_max = checkpoint["best_acc"]
            else:
                acc_max = 0
            if 'epoch_best' in checkpoint:
                epoch_best = checkpoint["epoch_best"]
            else:
                epoch_best = 0
    else:
        if args.webvision:
            acc_max_web, acc_max_imgnet = 0, 0
            epoch_best_web, epoch_best_imgnet = 0, 0
        else:
            acc_max = 0
            epoch_best = 0
        print("=> no checkpoint found at '{}'".format(resume_path))

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
                            pathlist_test_target=args.pathlist_test_target, distributed=args.distributed, crop_size=0.8,\
                                root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                                    save_dir=args.exp_dir, dry_run=args.dry_run,\
                                        use_fewshot=args.use_fewshot, annotation=args.annotation,\
                                            no_color_transform=args.no_color_transform, eval_only=args.no_aug)
        train_loader, fewshot_loader, test_loader_web, test_loader_target = loader.run()
    else:
        ## load webFG496 dataset
        loader = webFG496.webFG496_dataloader(batch_size=args.batch_size, num_class=args.num_class,\
            num_workers=args.workers, root_dir=args.root_dir, distributed=args.distributed, crop_size=0.2,\
                root_dir_target=args.root_dir_t, pathlist_target=args.pathlist_t,\
                    save_dir=args.exp_dir, dry_run=args.dry_run,\
                        use_fewshot=args.use_fewshot, annotation=args.annotation,\
                            no_color_transform=args.no_color_transform, eval_only=args.no_aug)
        train_loader, fewshot_loader, test_loader = loader.run()
    
    if args.gpu==0:
        logger = tb_logger.Logger(logdir=args.tensorboard_dir, flush_secs=2)
    else:
        logger = None

    assert (args.init_proto_epoch >= 0) and (args.init_proto_epoch < args.epochs),\
        "please make sure the epoch to update initial prototype properly"
    assert (args.init_proto_epoch <= args.start_clean_epoch),\
        "please make sure label cleaning is performed after prototype initialization"
    assert (args.start_clean_epoch < args.relation_clean_epoch),\
        "please make sure label cleaning by relation module is conducted after normal cleaning"
    # ## drop rate: only keeps the top small distance samples to update prototype
    # drop_rate_schedule = np.ones(args.epochs) * args.drop_rate
    # drop_rate_schedule[:args.init_proto_epoch] = 0
    # assert (args.init_proto_epoch + args.T_k < args.epochs),\
    #     "please make sure the number of epochs for decaying drop rate valid"
    # drop_rate_schedule[args.init_proto_epoch:args.init_proto_epoch+args.T_k] = np.linspace(0, args.drop_rate, args.T_k)
    print("=> start training from epoch {}".format(args.start_epoch))
    for epoch in range(args.start_epoch, args.epochs):
        ## set dataset sampler
        if hasattr(args, "pre_relation") and args.pre_relation:
            ## pretrain encoder+classifier before relation module
            if epoch > args.start_clean_epoch:
                return
        if args.distributed:
            loader.train_sampler.set_epoch(epoch)
        ## set model learning rate
        adjust_learning_rate(optimizer, epoch, args)  
        ## initialize prototype features
        if epoch == args.init_proto_epoch:
            init_prototype_fewshot(model, fewshot_loader, args, epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_init_proto.tar'.format(args.exp_dir))

        if not (hasattr(args, "ft_relation") and args.ft_relation):
            ## 仅当并不是只finetune relation module时才切换梯度更新方式
            if epoch == args.frozen_encoder_epoch:
                ## 将encoder模块设置为可更新梯度
                model, optimizer = optimize_encoder(model, args)
        ## train
        train(train_loader, model, criterion, optimizer, epoch, args, logger)
        ## save checkpoint model
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if args.webvision:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'best_acc_web': acc_max_web,
                    'best_acc_imgnet': acc_max_imgnet,
                    'epoch_best_web': epoch_best_web,
                    'epoch_best_imgnet': epoch_best_imgnet
                }, is_best=False, filename='{}/checkpoint_latest.tar'.format(args.exp_dir))
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'best_acc': acc_max,
                    'epoch_best': epoch_best
                }, is_best=False, filename='{}/checkpoint_latest.tar'.format(args.exp_dir))

            if (not args.ft_relation) and epoch > 0 and epoch <= 30 and epoch % 5 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_e{}.tar'.format(args.exp_dir, epoch))
        ## 更新prototype紧密度&清零访问次数+距离
        if (epoch>args.init_proto_epoch) and (epoch>args.start_clean_epoch):
            ## 当且仅当有样本参与更新prototype之后才能更新对应的密度
            model(None, args, is_proto_init=4)
        if args.webvision:
            ## test webvision dataset
            acc1_web, acc5_web = test(model, test_loader_web, args, epoch, logger, dataset_name="WebVision")
            acc1_imgnet, acc5_imgnet = test(model, test_loader_target, args, epoch, logger, dataset_name="ImgNet")
            if acc1_web > acc_max_web:
                acc_max_web = acc1_web
                epoch_best_web = epoch
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc_web': [acc1_web, acc5_web],
                        'best_acc_imgnet': [acc1_imgnet, acc5_imgnet]
                    }, is_best=False, filename='{}/checkpoint_best_web.tar'.format(args.exp_dir))                
            if acc1_imgnet > acc_max_imgnet:
                acc_max_imgnet = acc1_imgnet
                epoch_best_imgnet = epoch
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc_web': [acc1_web, acc5_web],
                        'best_acc_imgnet': [acc1_imgnet, acc5_imgnet]
                    }, is_best=False, filename='{}/checkpoint_best_imgnet.tar'.format(args.exp_dir)) 
        else:
            ## test webFineGrained dataset
            acc1, acc5 = test(model, test_loader, args, epoch, logger, dataset_name="FineGrained")
            if acc1 > acc_max:
                acc_max = acc1
                epoch_best = epoch
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': [acc1, acc5]
                    }, is_best=False, filename='{}/checkpoint_best.tar'.format(args.exp_dir))
        if args.webvision:
            print("accuracy top 1 web = {} @epoch {}".format(acc_max_web, epoch_best_web))
            print("accuracy top 1 imagenet = {} @epoch {}".format(acc_max_imgnet, epoch_best_imgnet))
        else:
            print("accuracy top 1 = {} @epoch {}".format(acc_max, epoch_best))
    return


def train(train_loader, model, criterion, optimizer, epoch, args, tb_logger):
    if args.rebalance:
        ## 对样本进行重采样并选取固定大小
        train_loader.dataset.resample()

    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')   
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    acc_inst = AverageMeter('Acc@Inst', ':2.2f')
    acc_cls_lowdim = AverageMeter('Acc@Cls_lowdim', ':2.2f')
    mse_reconstruct = AverageMeter('Mse@Reconstruct', ':2.2f')
    acc_relation = AverageMeter('Acc@Relation', ':2.2f')
    acct_relation = AverageMeter('AccTarget@Relation', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_inst, acc_proto,\
            mse_reconstruct, acc_relation, acct_relation, acc_cls_lowdim],
        prefix="Epoch: [{}]".format(epoch))
    print('==> Training...')
    # switch to train mode
    model.train()
    end = time.time()
    num_batch_norel = 0
    if args.webvision:
        class_weight = extract_class_weight(pathlist=args.pathlist_web, N_class=args.num_class)
        class_weight = torch.Tensor(class_weight).cuda(args.gpu)

    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        loss = 0
        # 1) classification | prediction [cls_out] and label [target]
        # 2) instance contrastive learning | prediction [logits] and label [inst_labels]
        # 3) prototype contrastive learning | prediction [logits_proto] and label [target]
        # 4) reconstruction learning | prediction [q_reconstruct] and label [q]
        # 5) relation scoring | prediction [relation_out] and label [relation_target]
        if epoch <= args.start_clean_epoch:
            is_clean = 0
        elif epoch <= args.relation_clean_epoch:
            is_clean = 1
        else:
            if hasattr(args, "clean_fusion") and args.clean_fusion:
                is_clean = 3
            else:
                is_clean = 2
        if hasattr(args, "pre_relation") and args.pre_relation:
            is_relation = False
        elif hasattr(args, "ft_relation") and args.ft_relation:
            is_relation = True
        elif epoch % args.update_relation_freq == 0:
            is_relation = True
        else:
            is_relation = False
        cls_out, target, target_soft, gt_score, inst_logits, inst_labels, logits_proto,\
            q_reconstruct, q, relation_out, relation_target,\
                output_arcface, target_arcface = model(batch, args,\
                    is_eval=False,\
                        is_proto=(epoch>args.init_proto_epoch),\
                            is_clean=is_clean,\
                                is_relation=is_relation)

        if args.webvision:
            sample_weight = torch.index_select(class_weight, dim=0, index=target.view(-1).type(torch.int64))
        ## classification loss
        if hasattr(args, "use_soft_label") and args.use_soft_label and (epoch>args.init_proto_epoch):
            ## 如果gt score很高, 则应该相信原始样本的输出标签
            ## 如果gt score很低, 则应该相信模型自身的输出标签
            loss_cls_hard = criterion(cls_out, target) * gt_score
            loss_cls_soft = - torch.sum(target_soft * F.log_softmax(cls_out, dim=1), dim=1) * (1-gt_score)
            if args.webvision:
                loss_cls = (loss_cls_hard * sample_weight).mean() + torch.sum(loss_cls_soft * sample_weight)/target.size(0)
            else:
                loss_cls = loss_cls_hard.mean() + torch.sum(loss_cls_soft)/target.size(0)
        else:
            if args.webvision:
                loss_cls = (criterion(cls_out, target) * sample_weight).mean()
            else:
                loss_cls = criterion(cls_out, target).mean()
        loss += loss_cls
        ## log class accuracy
        acc = accuracy(cls_out, target)[0] 
        acc_cls.update(acc[0])
        ## prototypical contrastive loss
        if epoch > args.init_proto_epoch:
            loss_proto = criterion(logits_proto, target).mean()
            loss += args.w_proto * loss_proto
            acc = accuracy(logits_proto, target)[0]
            acc_proto.update(acc[0])
        ## fewshot/confident sample classifiation loss for low-dim
        if (output_arcface is not None) and (target_arcface is not None):
            if args.webvision:
                sample_weight_lowdim = torch.index_select(class_weight, dim=0, index=target_arcface.view(-1).type(torch.int64))
                loss_cls_lowdim = (criterion(output_arcface, target_arcface) * sample_weight_lowdim).mean()
            else:
                loss_cls_lowdim = criterion(output_arcface, target_arcface).mean()
            loss += args.w_cls_lowdim * loss_cls_lowdim
            acc = accuracy(output_arcface, target_arcface)[0]
            acc_cls_lowdim.update(acc[0])
        ## reconstruction loss
        if args.low_dim != -1:
            loss_reconstruct = F.mse_loss(q_reconstruct, q.detach().clone())
            loss += args.w_recn * loss_reconstruct
            mse_reconstruct.update(loss_reconstruct.item())
        ## relation score regression loss
        if (relation_out is not None) and (relation_target is not None):
            relation_score = relation_out[relation_target>=0, relation_target]
            if hasattr(args, "sigmoid_relation") and args.sigmoid_relation:
                relation_out = relation_out / (torch.sum(relation_out, dim=1, keepdim=True) + 1e-6)
                relation_target_onehot = F.one_hot(relation_target, num_classes=args.num_class)
                loss_relation = torch.sum(-torch.sum(relation_target_onehot * torch.log(relation_out), dim=1))/relation_target.size(0)
            else:
                loss_relation = criterion(relation_out, relation_target.long()).mean()
                relation_score = F.sigmoid(relation_score)
            loss += args.w_relation * loss_relation
            ## 如果使用所有类别更新relation score来查看准确度
            acc = accuracy(relation_out, relation_target)[0]
            acc_relation.update(acc[0])
            ## 如果使用目标类别看relation score是否预测大于0.5(匹配)
            relation_out_bool = (relation_score > 0.5)
            acc_target = torch.sum(relation_out_bool) * 100 / relation_out_bool.size(0)
            acct_relation.update(acc_target)
        else:
            num_batch_norel += 1
        ## instance contrastive loss
        if epoch > args.init_proto_epoch:
            loss_inst = criterion(inst_logits, inst_labels).mean()
            loss += args.w_inst * loss_inst
            acc = accuracy(inst_logits, inst_labels)[0]
            acc_inst.update(acc[0])
        ## compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.gpu == 0 and i % args.print_freq == 0:
            progress.display(i)

    if args.ft_relation:
        print("In total {} batches have no relation samples for update loss".format(num_batch_norel))

    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Instance Acc', acc_inst.avg, epoch)
        tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Reconstruction Mse', mse_reconstruct.avg, epoch)
        tb_logger.log_value('Relation Acc Target', acct_relation.avg, epoch)
        tb_logger.log_value('Relation Acc', acc_relation.avg, epoch)
        tb_logger.log_value('Train LowDim Acc', acc_cls_lowdim.avg, epoch)
    return
        

def init_prototype_fewshot(model, fewshot_loader, args, epoch, is_eval=False):
    with torch.no_grad():
        print('==> Initialize FewShot Prototype...[Epoch {}]'.format(epoch))     
        model.eval()
        ## 初始化
        model(None, args, is_proto_init=1, is_eval=is_eval)
        ## 累加prototype特征并计数
        for batch in tqdm(fewshot_loader):
            model(batch, args, is_proto_init=2, is_eval=is_eval)
        ## 平均并归一化
        model(None, args, is_proto_init=3, is_eval=is_eval)
        ## make sure prototype is already updated
        if not is_eval:
            dist.barrier()
    return


def test(model, test_loader, args, epoch, tb_logger, dataset_name="WebVision"):
    with torch.no_grad():
        print('==> Evaluation...')     
        model.eval()    
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        
        # evaluate on webvision val set
        for batch in test_loader:
            ## outputs, feat, target, feat_reconstruct
            outputs, target, _, _ = model(batch, args, is_eval=True)
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


def save_checkpoint(state, is_best, filename='checkpoint_latest.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoint_best.pth')
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
        self.avg = self.sum / (self.count + 1e-3)

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
        batch_size = (target.size(0) + 1e-7)
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
