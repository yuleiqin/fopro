#!/usr/bin/env bash
set -e
set -x

## GPU CONFIG
export HOST_GPU_NUM=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DISTRIBUTED_BACKEND="nccl"
export NCCL_DEBUG_SUBSYS="INIT"
export NCCL_LAUNCH_MODE="GROUP"
export NCCL_P2P_DISABLE="0"
export NCCL_IB_DISABLE="1"
export NCCL_DEBUG="INFO"
export NCCL_SOCKET_IFNAME="eth1"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_SL="3"
export NCCL_NET_GDR_READ="1"
export NCCL_NET_GDR_LEVEL="3"
export NCCL_P2P_LEVEL="NVL"
## TRAINING CONFIG
export root_dir="./dataset/webvision1k/tfrecord_webvision_train"
export pathlist="./dataset/webvision1k/filelist/train_filelist_webvision_1k_usable_tf.txt"
export root_dir_test_web="./dataset/webvision1k/tfrecord_webvision_val"
export pathlist_test_web="./dataset/webvision1k/filelist/val_webvision_1k_usable_tf.txt"
export root_dir_test_target="./dataset/webvision1k/tfrecord_imgnet_val"
export pathlist_test_target="./dataset/webvision1k/filelist/val_webvision_1k_usable_tf.txt"
export root_dir_t="./dataset/webvision1k/tfrecord_imgnet_train"
export pathlist_t="./dataset/imgnet_webvision1k/fewshot_1_shot.txt"
export N_CLASSES=1000
export N_BATCHSIZE=256
## SAVE CONFIG
export SAVE_DIR_ROOT="./results"
export SAVE_DIR_NAME=web-webv1k
export SAVE_DIR=${SAVE_DIR_ROOT}/${SAVE_DIR_NAME}
if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

export SAVE_DIR1=${SAVE_DIR}/stage1
if [ ! -d ${SAVE_DIR1} ]; then
    mkdir -p ${SAVE_DIR1}
fi

export resume_path=ckpt_mopro/MoPro_V1_epoch90.tar

python3 -W ignore -u train.py --seed 0 --use_fewshot --webvision --rebalance \
--margin 0.5 --relation_nobp --use_temperature_density --use_soft_label \
--root_dir ${root_dir} --pathlist_web ${pathlist} \
--root_dir_test_web ${root_dir_test_web} --pathlist_test_web ${pathlist_test_web} \
--root_dir_test_target ${root_dir_test_target} --pathlist_test_target ${pathlist_test_target} \
--root_dir_t ${root_dir_t} --pathlist_t ${pathlist_t} \
--arch "resnet50" --workers 8 --epochs 50 --frozen_encoder_epoch 0 \
--init-proto-epoch 15 --batch-size ${N_BATCHSIZE} --relation_clean_epoch 20 \
--lr 1e-2 --cls_relation --pre_relation --cos --resume ${resume_path} \
--exp-dir ${SAVE_DIR1} --start-epoch 0 \
--num-class ${N_CLASSES} --low-dim 128 \
--moco_queue 8192 --start_clean_epoch 19 --pseudo_th 0.6 --dist_th 20 \
--w-inst 1 --w-recn 1 --w-proto 1 --w-relation 1 --w-cls-lowdim 1 \
--dist-url 'tcp://localhost:10213' --multiprocessing-distributed --world-size 1 --rank 0


export SAVE_DIR2=${SAVE_DIR}/stage2
if [ ! -d ${SAVE_DIR2} ]; then
    mkdir -p ${SAVE_DIR2}
fi

export resume_path=${SAVE_DIR1}/checkpoint_latest.tar

python3 -W ignore -u train.py --seed 0 --use_fewshot --webvision --rebalance \
--margin 0.5 --relation_nobp --use_temperature_density --use_soft_label \
--root_dir ${root_dir} --pathlist_web ${pathlist} \
--root_dir_test_web ${root_dir_test_web} --pathlist_test_web ${pathlist_test_web} \
--root_dir_test_target ${root_dir_test_target} --pathlist_test_target ${pathlist_test_target} \
--root_dir_t ${root_dir_t} --pathlist_t ${pathlist_t} \
--arch "resnet50" --workers 8 --epochs 10 --frozen_encoder_epoch 0 \
--init-proto-epoch 0 --batch-size ${N_BATCHSIZE} --relation_clean_epoch 20 \
--lr 1e-4 --cls_relation --ft_relation --cos \
--exp-dir ${SAVE_DIR2} --start-epoch 1 --warmup_epoch 0 \
--num-class ${N_CLASSES} --low-dim 128 --resume ${resume_path} \
--moco_queue 8192 --start_clean_epoch 0 --pseudo_th 0.6 --dist_th 20 \
--w-inst 1 --w-recn 1 --w-proto 1 --w-relation 1 --w-cls-lowdim 1 \
--dist-url 'tcp://localhost:10214' --multiprocessing-distributed --world-size 1 --rank 0

export SAVE_DIR3=${SAVE_DIR}/stage3
if [ ! -d ${SAVE_DIR3} ]; then
    mkdir -p ${SAVE_DIR3}
fi

export resume_path=${SAVE_DIR2}/checkpoint_latest.tar

python3 -W ignore -u train.py --seed 0 --use_fewshot --webvision --rebalance \
--margin 0.5 --relation_nobp --use_temperature_density --use_soft_label \
--root_dir ${root_dir} --pathlist_web ${pathlist} --clean_fusion \
--root_dir_test_web ${root_dir_test_web} --pathlist_test_web ${pathlist_test_web} \
--root_dir_test_target ${root_dir_test_target} --pathlist_test_target ${pathlist_test_target} \
--root_dir_t ${root_dir_t} --pathlist_t ${pathlist_t} \
--arch "resnet50" --workers 8 --epochs 50 --frozen_encoder_epoch 0 \
--init-proto-epoch 15 --batch-size ${N_BATCHSIZE} --relation_clean_epoch 20 \
--lr 1e-2 --cls_relation --update-relation-freq 1 --cos \
--exp-dir ${SAVE_DIR3} --start-epoch 21 --warmup_epoch 0 \
--num-class ${N_CLASSES} --low-dim 128 --resume ${resume_path} \
--moco_queue 8192 --start_clean_epoch 19 --pseudo_th 0.6 --dist_th 20 \
--w-inst 1 --w-recn 1 --w-proto 1 --w-relation 1 --w-cls-lowdim 1 \
--dist-url 'tcp://localhost:10215' --multiprocessing-distributed --world-size 1 --rank 0
