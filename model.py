import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append('.')
from backbone.basenet import AlexNet_Encoder, VGG_Encoder, BCNN_encoder
from backbone.resnet import resnet50
from backbone.classifier import Normalize, MLP_classifier
import math


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

# class JSD(nn.Module):
#     def __init__(self):
#         super(JSD, self).__init__()
#         self.kl = nn.KLDivLoss(reduction='none', log_target=True)
    
#     def forward(self, p, q):
#         m = (0.5 * (p + q)).log()
#         return 0.5 * (self.kl(m, p.log()) + self.kl(q.log(), m))

def JSD(p, q, reduction="batchmean"):
    log_mean_output = (0.5 * (p + q)).log()
    return 0.5 * (F.kl_div(log_mean_output, p, reduction=reduction) +\
        F.kl_div(log_mean_output, q, reduction=reduction))


class FoPro(nn.Module):

    def __init__(self, args):
        super(FoPro, self).__init__()
        ##=========================================================================##
        ## 设置特征抽取器
        ##=========================================================================##
        if args.arch == 'resnet50':
            ### this is the default
            base_encoder_q = resnet50(pretrained=args.pretrained, width=1)
            base_encoder_k = resnet50(pretrained=args.pretrained, width=1)
        elif args.arch == 'resnet50x2':
            base_encoder_q = resnet50(pretrained=args.pretrained, width=2)
            base_encoder_k = resnet50(pretrained=args.pretrained, width=2)
        elif args.arch == 'resnet50x4':
            base_encoder_q = resnet50(pretrained=args.pretrained, width=4)
            base_encoder_k = resnet50(pretrained=args.pretrained, width=4)
        elif args.arch == 'vgg':
            ## 默认num_out_channel=4096
            base_encoder_q = VGG_Encoder(pretrained=args.pretrained)
            base_encoder_k = VGG_Encoder(pretrained=args.pretrained)
        elif args.arch == 'bcnn':
            ## 默认num_out_channel=512**2
            base_encoder_q = BCNN_encoder(pretrained=args.pretrained, num_out_channel=512**2)
            base_encoder_k = BCNN_encoder(pretrained=args.pretrained, num_out_channel=512**2)
        elif args.arch == 'alexnet':
            ## 默认num_out_channel=4096
            base_encoder_q = AlexNet_Encoder(pretrained=args.pretrained)
            base_encoder_k = AlexNet_Encoder(pretrained=args.pretrained)
        else:
            raise NotImplementedError('model not supported {}'.format(args.arch))
        ## encoder
        self.encoder_q = base_encoder_q
        ## momentum encoder
        self.encoder_k = base_encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        ##=========================================================================##
        ## 设置高维向量=>低维向量的投影器
        ## 如果args.low_dim == -1则不设置projection直接用分类embedding
        ##=========================================================================##
        if args.low_dim != -1:
            self.low_dim = args.low_dim
            if args.arch == 'bcnn':
                ## bcnn最后输出的维度过大 多层MLP易爆显存
                self.projection = nn.Sequential(*[
                    nn.Linear(self.encoder_q.num_out_channel, self.low_dim),
                    Normalize(2),
                ])
                self.projection_back = nn.Sequential(*[
                    nn.Linear(self.low_dim, self.encoder_q.num_out_channel),
                    # nn.Dropout(p=0.5),
                ])
            else:
                self.projection = nn.Sequential(*[
                    nn.Linear(self.encoder_q.num_out_channel, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, self.low_dim),
                    Normalize(2)
                ])
                self.projection_back = nn.Sequential(*[
                    nn.Linear(self.low_dim, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, self.encoder_q.num_out_channel),
                ])
        else:
            self.low_dim = self.encoder_q.num_out_channel
            self.projection = nn.Sequential(*[
                nn.Identity(),
                Normalize(2)
            ])
            self.projection_back = nn.Sequential(*[
                nn.Identity(),
            ])
        ##=========================================================================##
        ## 设置关系抽取器
        ## 如果使用参数化的relation module来自己学习距离关系则去掉以下注释
        ## 避免过多参数量可以用l2 norm + cosine similarity代表relation score
        ## 但直接用cosine similarity可能并不是最优解
        ##=========================================================================##
        if hasattr(args, "sigmoid_relation") and args.sigmoid_relation:
            use_sigmoid = True
        else:
            use_sigmoid = False
        N_hidden_layer = 128
        # if hasattr(args, "cls_relation") and args.cls_relation:
        #     ## 使用分类任务损失作为relation module的训练损失
        #     ## 使用probability embedding进行不确定性建模(模仿正态分布)
        #     ## 最后一层conv代表gamma和beta
        #     self.relation = nn.Sequential(*[
        #         nn.Linear(self.low_dim * 2, N_hidden_layer),
        #         nn.BatchNorm1d(N_hidden_layer),
        #         nn.Conv1d(1, 1, 1),
        #         nn.ReLU(),
        #         nn.Linear(N_hidden_layer, 1),        
        #     ])
        self.relation = MLP_classifier(num_class=1, in_channel=self.low_dim * 2,\
            num_hidden=N_hidden_layer, use_norm=False, use_sigmoid=use_sigmoid)
        self.relation.apply(init_weights)
        ##=========================================================================##
        ## 设置分类器 不使用norm
        ##=========================================================================##
        self.classifier = MLP_classifier(num_class=args.num_class,\
            in_channel=self.encoder_q.num_out_channel, use_norm=False)
        self.classifier.apply(init_weights)
        ##=========================================================================##
        ## 设置FewShot+干净样本的arcface分类器 使用norm
        ##=========================================================================##
        self.classifier_projection = MLP_classifier(num_class=args.num_class,\
            in_channel=args.low_dim, use_norm=True)
        self.classifier_projection.apply(init_weights)
        ##=========================================================================##
        ## 设置自监督对比学习用的已访问队列
        ##=========================================================================##
        self.register_buffer("queue", torch.randn(self.low_dim, args.moco_queue))        
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        ##=========================================================================##
        ## 设置每个类别的prototype, 访问次数, 访问距离
        ##=========================================================================##
        self.register_buffer("prototypes", torch.zeros(args.num_class, self.low_dim))
        self.register_buffer("prototypes_visited", torch.zeros(args.num_class))
        self.register_buffer("prototypes_density", torch.ones(args.num_class)*args.temperature)
        self.register_buffer("prototypes_distance", torch.zeros(args.num_class))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)
        return
    
    @torch.no_grad()
    def _initialize_prototype_features(self):
        """
        initialize prototype features by average
        """
        ## average over all
        self.prototypes = self.prototypes / (self.prototypes_visited.view(-1, 1) + 1e-6)
        ## normalize
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        self._print_prototype_features()
        ## 重新对prototype visited数目清零
        self.prototypes_visited *= 0
        return
    
    @torch.no_grad()
    def _zero_prototype_features(self):
        self.prototypes *= 0.
        self.prototypes_distance *= 0.
        self.prototypes_visited *= 0.
        return
    
    @torch.no_grad()
    def _print_prototype_features(self):
        print("prototype visited total",\
            torch.sum(self.prototypes_visited).item())
        # for class_id in range(len(self.prototypes_visited)):
        #     print("class {} has {} instance features".format(class_id, self.prototypes_visited[class_id]))
        prototype_nonzero = self.prototypes[self.prototypes_visited>0]
        print("prototype features", prototype_nonzero.size(), prototype_nonzero)
        return

    @torch.no_grad()
    def _update_prototype_density(self):
        """
        update prototype density for temperature
        """
        ## update density density
        for class_id in range(len(self.prototypes_visited)):
            num_visited = self.prototypes_visited[class_id]
            distance_sum = self.prototypes_distance[class_id]
            self.prototypes_density[class_id] = distance_sum / (num_visited*torch.log(num_visited+10.)+1e-7)
            # print("class_id {} has {} samples with total distance to prototypes {} and density {}".format(
            #     class_id, num_visited, distance_sum, self.prototypes_density[class_id]
            # ))
        self.prototypes_distance *= 0
        self.prototypes_visited *= 0
        return
    
    @torch.no_grad()
    def _print_norm_tensor(self, input_tensor, tensor_name="tensor", power=2):
        """
        print the l-N norm of input tensor
        """
        norm_tensor = input_tensor.pow(power).sum(1, keepdim=True).pow(1./power)
        print("L{} norm of {} = {}".format(power, tensor_name, norm_tensor))
        return

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, args):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer
        # update the pointer
        self.queue_ptr[0] = ptr
        return

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # get batchsize from current gpu
        batch_size_this = x.shape[0]
        # gather from all gpus
        x_gather = concat_all_gather(x)
        # get batchsize from all gpus
        batch_size_all = x_gather.shape[0]
        # get the number of gpus
        num_gpus = batch_size_all // batch_size_this
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        # x (original order) -> x[idx_shuffle] (shuffled order)
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)
        # index for restoring
        # idx_shuffle -> argsort from 0~batch_size_all-1 -> recover by idx_unshuffle
        idx_unshuffle = torch.argsort(idx_shuffle)
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        # get re-ordered x by id_shuffle on this gpu
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # get batchsize from current gpu
        batch_size_this = x.shape[0]
        # gather from all gpus
        x_gather = concat_all_gather(x)
        # get batchsize from all gpus
        batch_size_all = x_gather.shape[0]
        # get the number of gpus
        num_gpus = batch_size_all // batch_size_this
        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        # get the local batch x from restored order
        return x_gather[idx_this]
    
    def forward(self, batch, args,\
        is_eval=False, is_proto=False, is_clean=0,\
            is_proto_init=0, is_analysis=False, is_relation=False):
        """前向传播for train and eval
        batch: 当前输入,包括图像;标签;domain标签(web or fewshot);(输入强数据增强图像);图像pathlist
        args: 全局训练参数设置
        is_eval: 是否验证集推理, 如果是则仅仅返回分类结果&标签&重建的高维特征&高维特征
        is_proto: 是否进行prototype更新, 如果不更新则不会进行对比学习&噪声过滤&标签修正
        is_clean: 是否进行标签修正&噪声过滤(初始时使用人工定义准则), =0:不去噪;=1人工定义准则去噪;=2使用relation module去噪;=3综合使用
        is_proto_init: 是否初始化prototype; =1:归零; =2:累加特征; =3求取平均
        """
        ##=========================================================================##
        ## 初始化模型prototype=归零&均匀化
        ##=========================================================================##
        if is_proto_init == 1:
            ## 初始化prototype特征1=>计数归零
            self._zero_prototype_features()
            return
        if is_proto_init == 3:
            ## 初始化prototype特征3=>平均化
            self._initialize_prototype_features()
            return
        if is_proto_init == 4:
            ## 更新分类输出的各类temperature
            self._update_prototype_density()
            return
        img = batch[0].cuda(args.gpu, non_blocking=True)        
        target = batch[1].cuda(args.gpu, non_blocking=True)          
        ##=========================================================================##
        ## 初始化模型prototype=使用fewshot数据按类累加特征
        ##=========================================================================##
        if is_proto_init == 2:
            ## 初始化prototype特征2=>按类累加
            with torch.no_grad():  # no gradient
                if not is_eval:
                    # shuffle for making use of BN
                    img, idx_unshuffle = self._batch_shuffle_ddp(img)
                    k_compress = self.projection(self.encoder_k(img))
                    # undo shuffle
                    k_compress = self._batch_unshuffle_ddp(k_compress, idx_unshuffle)
                    # gather all features across gpus
                    features = concat_all_gather(k_compress)
                    # gather all targets across gpus
                    targets = concat_all_gather(target)
                else:
                    features = self.projection(self.encoder_k(img))
                    targets = target
                for feat, label in zip(features, targets):
                    self.prototypes[int(label)] += feat
                    self.prototypes_visited[int(label)] += 1  # count visited times for average
            return
        ##=========================================================================##
        ## 特征抽取器提取特征; 压缩部分得到低维特征表示; 分类器进行分类
        ##=========================================================================##
        q = self.encoder_q(img)
        q_compress = self.projection(q)
        output = self.classifier(q)
        ##=========================================================================##
        ## 特征压缩与重建部分
        ## 1) 训练初期需要利用projection-reconstruction来训练projector生成合理的low-dim特征
        ## 2) 训练后期仅仅更新reconstructor防止pytorch报错(存在未更新参数)
        ##=========================================================================##
        if is_proto:
            ## 更新重建部分仅仅依赖于q而不需要反传梯度至特征提取器
            q_reconstruct = self.projection_back(q_compress.detach().clone())
        else:
            ## 同时更新压缩和重建部分
            q_reconstruct = self.projection_back(self.projection(q.detach().clone()))
        ##=========================================================================##
        ## 测试推理仅返回预测结果
        ##=========================================================================##
        if is_eval and not (is_analysis):
            return output, target, q_reconstruct, q
        ##=========================================================================##
        ## 来自FewShot domain的样本一定是干净样本
        ##=========================================================================##
        if is_eval:
            domain = batch[2].cuda(args.gpu, non_blocking=True)
        else:
            domain = batch[3].cuda(args.gpu, non_blocking=True)
        fewshot_idx = (domain > 0).view(-1)
        ##=========================================================================##
        ## 动量更新特征提取器(除了仅finetune relation module时不更新)
        ##=========================================================================##
        if not (hasattr(args, "ft_relation") and args.ft_relation):
            with torch.no_grad():  # no gradient
                self._momentum_update_key_encoder(args)
        ##=========================================================================##
        ## 计算基于prototype的对比学习 最后仅仅选择干净样本进行梯度反传
        ## 同时准备用于soft-target学习的标签对象
        ##=========================================================================##
        if is_proto:
            ## compute protoypical logits
            prototypes = self.prototypes.detach().clone()
            logits_proto = torch.mm(q_compress, prototypes.t())
            logits_proto_raw = logits_proto.detach().clone()
            ## 针对fewshot样本 可以考虑margin来尤其拉近距离
            if args.margin != 0:
                ## Additive margin softmax
                target_fewshot = target[fewshot_idx]
                logits_proto[fewshot_idx,target_fewshot] -= args.margin
            if args.use_temperature_density:
                ## 可以根据每个类别的密度紧致程度进行缩放
                density_temperature = self.prototypes_density.detach().clone().view(1, -1)
                logits_proto = logits_proto/density_temperature
                logits_proto_raw = logits_proto_raw/density_temperature
            else:
                logits_proto = logits_proto/args.temperature
                logits_proto_raw = logits_proto_raw/args.temperature
            with torch.no_grad():
                ## 每个样本输出是由High-Dim分类器输出概率&样本-prototype相似度加权得到的
                target_soft = args.alpha*F.softmax(output, dim=1) + (1-args.alpha)*F.softmax(logits_proto_raw, dim=1)
                ## 注意生成target soft时保证fewshot样本的标签不会改变
                target_soft[fewshot_idx] = F.one_hot(target[fewshot_idx].long(), num_classes=args.num_class).float()
        else:
            logits_proto = 0
            # target_soft = F.one_hot(target, num_classes=args.num_class)
            target_soft = F.softmax(output, dim=1)
        ##=========================================================================##
        ## 初始阶段使用自己定义的去噪方式对样本标签进行修正&对噪声样本进行过滤
        ## 初始化gt_score仅在is_proto=True, is_clean=False情况下有效
        ## 仅使用干净归档样本进行损失函数计算&prototype更新
        ## 后续阶段is_proto&is_clean=True时使用relation module去噪
        ## 最终的干净可归档样本就是clean_idx/correct_idx/fewshot_idx等并集合
        ##=========================================================================##
        gt_score = target_soft[target>=0,target]
        correct_idx = fewshot_idx | (gt_score>1./args.num_class)
        if is_proto and is_clean:
            if is_clean == 1:
                clean_idx_pred = gt_score>(1/args.num_class)
                ## 分配Pred标签的前提是该标签的值大于阈值, 可以覆盖GT标签
                max_score, hard_label = target_soft.max(1)
                correct_idx = max_score>args.pseudo_th
                ## 从correct_idx中剔除fewshot index样本(不参与标签修改)
                correct_idx = correct_idx & (~fewshot_idx)
                ## 伪标签=>修改图像标签为预测置信度高的类别
                target[correct_idx] = hard_label[correct_idx]
                clean_idx = clean_idx_pred | correct_idx | fewshot_idx
            elif is_clean == 2 or is_clean == 3:
                ## 使用relation module而不是人工定义的准则来定义样本(encoder q特征)
                with torch.no_grad():
                    prototypes = self.prototypes.detach().clone()
                    q_compress_tiled = torch.repeat_interleave(q_compress.detach().clone(), repeats=args.num_class, dim=0)
                    prototypes_tiled = prototypes.repeat(q_compress.size(0), 1)
                    soft_label = self.relation(torch.cat((q_compress_tiled, prototypes_tiled), dim=1)).view(-1, args.num_class)
                    if not (hasattr(args, "sigmoid_relation") and args.sigmoid_relation):
                        soft_label = F.sigmoid(soft_label)
                    ## soft_label = soft_label / (torch.sum(soft_label, dim=1, keepdim=True) + 1e-6)
                    ## soft_label = F.softmax(soft_label, dim=1)
                    gt_score_relation = soft_label[target>=0,target]
                    clean_idx_relation = gt_score_relation>(args.pseudo_th)  ## 保留GT标签的前提是输出的概率值大于阈值
                if is_clean == 2:
                    ## 完全依赖relation module挑选高置信度的样本
                    max_score, hard_label = soft_label.max(1)
                    correct_idx = max_score>args.pseudo_th
                    correct_idx = correct_idx & (~fewshot_idx) & (~clean_idx_relation)
                    target[correct_idx] = hard_label[correct_idx]
                    clean_idx = clean_idx_relation | correct_idx | fewshot_idx
                else:
                    ## 使用relation module
                    clean_idx_pred = gt_score>(1/args.num_class)
                    max_score, hard_label = target_soft.max(1)  
                    correct_idx = max_score>args.pseudo_th
                    correct_idx = correct_idx & (~fewshot_idx) & (~clean_idx_relation)
                    target[correct_idx] = hard_label[correct_idx]
                    clean_idx = clean_idx_pred | clean_idx_relation | correct_idx | fewshot_idx
            else:
                raise ValueError("is_clean (if True) should only be 1 or 2")

            if (not is_eval) and (not is_analysis):
                ##=========================================================================##
                ## 开始更新prototype使用distribute gather来使得每个模型同步更新prototype
                ## 使用momentum encoder获得比较平稳的特征
                ## 这部分样本应当选取的是置信度高的
                ##=========================================================================##
                with torch.no_grad():  # no gradient
                    # shuffle & undo shuffle for making use of BN
                    img, idx_unshuffle = self._batch_shuffle_ddp(img)
                    k_compress = self.projection(self.encoder_k(img))
                    k_compress = self._batch_unshuffle_ddp(k_compress, idx_unshuffle)
                    features = concat_all_gather(k_compress)
                    ## 记录下所有样本距离对应(伪)类别prototype的距离l2
                    prototypes = self.prototypes.clone().detach()
                    targets = concat_all_gather(target).view(-1)
                    prototype_targets = torch.index_select(prototypes, dim=0, index=targets.view(-1).type(torch.int64))
                    dists_prototypes = torch.norm(features-prototype_targets, dim=1)
                    ## 按照FoPro的方法是仅仅保留clean样本来更新prototype
                    clean_idx_all = concat_all_gather(clean_idx.long()) 
                    clean_idx_all = clean_idx_all.bool()
                    # update momentum prototypes with pseudo-labels
                    for feat, label, dist in zip(features[clean_idx_all],\
                        targets[clean_idx_all], dists_prototypes[clean_idx_all]):
                        self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
                        self.prototypes_visited[label] += 1  # 记录更新当前类prototype的样本数量
                        self.prototypes_distance[label] += dist  # 记录下该样本距离类prototype的L2距离
                    ## normalize prototypes
                    self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        ##=========================================================================##
        ## 用于分析当前结果的变化情况
        ##=========================================================================##
        if is_analysis:
            target_index = target.view(-1, 1).type(torch.int64)
            prototypes = self.prototypes.detach().clone()
            ## 计算与各个prototype之间的距离
            logits_proto = torch.mm(q_compress, prototypes.t())
            cos_mean = torch.mean(logits_proto, dim=1)
            cos_max, cos_max_idx = torch.max(logits_proto, dim=1)
            cos_min, cos_min_idx = torch.min(logits_proto, dim=1)
            cos_median, cos_median_idx = torch.median(logits_proto, dim=1)
            dist_mean = 1.-cos_mean
            dist_min = 1.-cos_max
            dist_max = 1.-cos_min
            dist_median = 1.-cos_median
            dist_target = 1.-torch.gather(logits_proto, dim=1, index=target_index)
            ## 返回测试集的结果变化
            # if is_eval:
            #     return output, q_compress, soft_label,\
            #         [target, cos_max_idx, cos_median_idx, cos_min_idx, clean_idx],\
            #             [dist_target, dist_min, dist_mean, dist_median, dist_max]
        ## 定义可归档样本
        with torch.no_grad(): # no gradient
            if is_proto:
                ## 1) fewshot样本
                ## 2) 与各个prototype之间的相似度 & prototype与prototype之间的相似度的比值
                prototypes = self.prototypes.detach().clone()
                proto_proto = torch.mm(prototypes, prototypes.t())  ## N_cls * N_cls
                proto_proto_sim = torch.index_select(proto_proto, dim=0, index=target.view(-1))
                logits_proto_sim = torch.mm(q_compress, prototypes.t()) ## N_batch * N_cls
                ## 度量方式a. 计算二者分布差的L2范数=>最大值归一化
                # dist_sim = torch.norm(proto_proto_sim - logits_proto_sim, dim=1)/(math.sqrt(2**2 * args.num_class))
                dist_sim_diff = torch.abs(proto_proto_sim - logits_proto_sim)  ## N_batch * N_cls
                dist_sim = 0.5 * dist_sim_diff[target>=0, target]  ## 点选每个样本与对应类别的prototype距离
                dist_sim_diff[target>=0, target] = 0  ## 抹掉该距离方便后续加权
                dist_sim += 0.5 * torch.sum(dist_sim_diff, dim=1)
                ## 度量方式b. 计算二者分布的JS散度=>最大值归一化
                # dist_sim = (JSD(F.softmax(proto_proto_sim.detach(), dim=-1),\
                #     F.softmax(logits_proto_sim.detach(), dim=-1), reduction="none").sum(dim=-1))/math.log(2)
                # print("distance similarity L2norm of fewshot samples", dist_sim[fewshot_idx])
                # print("distance similarity L2norm of all samples", dist_sim)
                if (not is_clean):
                    arcface_idx = fewshot_idx | (dist_sim <= args.dist_th)
                else:
                    ## 使用clean的准则挑选样本
                    arcface_idx = clean_idx
            else:
                ## 没有fewshot样本只能用所有高置信度样本
                arcface_idx = fewshot_idx | (gt_score>args.pseudo_th)

        if is_analysis:
            ## 返回训练集的结果变化
            # if not (is_eval):
            return output, q_compress, soft_label,\
                [target, cos_max_idx, cos_median_idx,\
                    cos_min_idx, clean_idx, arcface_idx],\
                        [dist_target, dist_min, dist_mean, dist_median, dist_max, dist_sim]
        ##=========================================================================##
        ## 利用Fewshot+干净数据样本的投影通过arcface计算损失
        ##=========================================================================##
        img_aug = batch[2].cuda(args.gpu, non_blocking=True)
        q_aug_compress = self.projection(self.encoder_q(img_aug))
        if torch.sum(arcface_idx) > 0:
            q_compress_arcface = q_compress[arcface_idx]
            q_aug_compress_arcface = q_aug_compress[arcface_idx]
            output_arcface = torch.cat((self.classifier_projection(q_compress_arcface),\
                self.classifier_projection(q_aug_compress_arcface)), dim=0)
            target_arcface = torch.cat((target[arcface_idx], target[arcface_idx]), dim=0)
            ## additive margin & temperature softmax (输出0~1之间)
            if args.margin != 0:
                output_arcface[target_arcface>=0,target_arcface] -= args.margin
            output_arcface /= args.temperature
            # print("size output arcface {} target arcface {}".format(output_arcface.size(), target_arcface.size()))
        else:
            output_arcface, target_arcface = None, None
        ##=========================================================================##
        ## 强数据增强用于自监督对比学习
        ##=========================================================================##
        with torch.no_grad():  # no gradient
            ## shuffle for making use of BN undo shuffle
            img_aug, idx_unshuffle = self._batch_shuffle_ddp(img_aug)
            k_compress = self.projection(self.encoder_k(img_aug))
            k_compress = self._batch_unshuffle_ddp(k_compress, idx_unshuffle)
        ## compute instance logits
        l_pos = torch.einsum('nc,nc->n', [q_compress, k_compress]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_compress, self.queue.detach().clone()])
        inst_logits = torch.cat([l_pos, l_neg], dim=1)
        ## apply temperature on softmax
        inst_logits /= args.temperature
        inst_labels = torch.zeros(inst_logits.shape[0], dtype=torch.long).cuda()
        ##=========================================================================##
        ## 访问过的队列 入库 & 出库
        ##=========================================================================##
        self._dequeue_and_enqueue(k_compress, args)
        ##=========================================================================##
        ## 使用relation模块判断噪声样本
        ## 首先进行训练=>然后替代(softmax+prototype-similarity)/2用于判断干净\噪声样本
        ## relation学习模块机制 针对clean样本使用回归方式拟合0-1目标 而不是K-way softmax方式
        ## relation_score值为0~1之间
        ##=========================================================================##
        if args.pre_relation or args.ft_relation:
            ## 一开始训练 & 微调 relation时使用arcface idx
            relation_idx = arcface_idx
        else:
            ## 后续只要是干净样本都加入训练
            relation_idx = correct_idx
        N_random = 4
        if is_relation and is_proto and torch.sum(relation_idx) > 0:
            ## 必须先得有prototype才能更新训练只利用置信度很高的样本来进行更新
            ## 首先构建正负样本对
            with torch.no_grad():
                prototypes = self.prototypes.detach().clone()
                q_compress_relation = q_compress[relation_idx].detach().clone()
                # q_aug_compress_relation = q_aug_compress[relation_idx].detach().clone()
                N_relation = q_compress_relation.size(0)
                target_relation = target[relation_idx]
                ##=========================================================================##
                ## 方案一=如果是选择softmax分类预测来训练relation module
                ##=========================================================================##
                q_compress_relation_tiled = torch.repeat_interleave(q_compress_relation, repeats=args.num_class, dim=0)
                prototypes_relation_tiled = prototypes.repeat(N_relation, 1)
                ##=========================================================================##
                ## 方案二=如果是选择正负样本对的方式来训练relation module
                ##=========================================================================##
                # prototype_target = torch.index_select(prototypes, dim=0, index=target_relation.view(-1).type(torch.int64))
                ## 正样本=1) k_compress和prototype_target;
                ## 找到每个样本相似度(除了对应target prototype外)最高的类prototype
                ## 负样本选择方式: 1) k_compress和最难的负样本 & k_compress和剩下的N个随机负样本以避免退化情况
                # logits_proto_sim_relation = torch.mm(q_compress_relation, prototypes.t()) ## N_batch * N_cls
                # logits_proto_sim_relation[:,target_relation] -= 100
                # sim_relation_sorted_idx = torch.argsort(logits_proto_sim_relation, dim=1, descending=True)
                ## 每行第一列为最困难的样本; 每行随机抽取N列作为随机负样本
                # hard_negative_idx = torch.gather(sim_relation_sorted_idx, dim=1,\
                #     index=torch.zeros(N_relation).view(-1, 1).type(torch.int64).cuda())
                # random_negative_idx = torch.gather(sim_relation_sorted_idx, dim=1,\
                #     index=torch.randint(1, args.num_class-1, (N_relation, N_random)).view(-1, N_random).type(torch.int64).cuda())
                # random_negative_idx = random_negative_idx.transpose(1, 0).flatten()
                # prototype_hard_negative = torch.index_select(prototypes, dim=0, index=hard_negative_idx.view(-1).type(torch.int64))
                # prototype_rand_negative = torch.index_select(prototypes, dim=0, index=random_negative_idx.view(-1).type(torch.int64))
                ## 负样本选择方式: 2) 只随机挑选1个负样本以避免退化情况
                # random_negative_idx = torch.randint(1, args.num_class-1, (N_relation, N_random)).view(-1, N_random).type(torch.int64).cuda()
                # random_negative_idx = (random_negative_idx + target_relation.view(-1, 1).type(torch.int64)) % args.num_class
                # random_negative_idx = random_negative_idx.transpose(1, 0).flatten()
                # prototype_rand_negative = torch.index_select(prototypes, dim=0, index=random_negative_idx.view(-1).type(torch.int64))
                # q_compress_relation_tiled = torch.cat([q_compress_relation]*(1+N_random), dim=0)
                # prototypes_relation_tiled = torch.cat((prototype_target, prototype_rand_negative), dim=0)
            relation_score = self.relation(torch.cat((q_compress_relation_tiled, prototypes_relation_tiled), dim=1))
            ## 方案一=分类训练更新
            relation_target = target_relation
            relation_score = relation_score.view(N_relation, args.num_class)
            ## 方案二=构建完正负样本对后输入relation module进行更新
            # relation_target = torch.cat((torch.ones(N_relation), torch.zeros(N_relation*N_random)), dim=0).cuda()
        else:
            relation_score, relation_target = None, None
        ##=========================================================================##
        ## 仅仅更新干净样本
        ##=========================================================================##
        if is_clean and is_proto:
            output = output[clean_idx]
            target = target[clean_idx]
            logits_proto = logits_proto[clean_idx]
            target_soft = target_soft[clean_idx]
            gt_score = gt_score[clean_idx]

        return output, target, target_soft, gt_score, inst_logits, inst_labels, logits_proto,\
            q_reconstruct, q, relation_score, relation_target,\
                output_arcface, target_arcface


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ =="__main__":
    print(math.log(2))
    # p = torch.Tensor([[1, 3, 5, 7],[10, 0, 0, 0]])
    # q = torch.Tensor([[0, 10, 0, 0], [4, 0, 0, 0]])
    # p_softmax = F.softmax(p, dim=-1)
    # q_softmax = F.softmax(q, dim=-1)
    p_softmax = torch.Tensor([[1, 0, 0, 0],[1, 0, 0, 0]])
    q_softmax = torch.Tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
    # m = (p_softmax+q_softmax)/2.0
    # print("p softmax ", p_softmax)
    # print("q softmax ", q_softmax)
    # print("(p + q)/2 ", m)
    # # print(JSD(p, q))
    # p2m = p_softmax * torch.log(p_softmax/m)
    # q2m = q_softmax * torch.log(q_softmax/m)
    # print(((p2m + q2m)*0.5).sum(dim=-1))
    print(JSD(p_softmax, q_softmax, reduction="none").sum(dim=-1))


