#!/usr/bin/env bash
set -e
set -x

## GPU CONFIG
export HOST_GPU_NUM=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
export root_dir=''
export root_dir_t=''
export pathlist_t=''
export N_CLASSES=200
export N_BATCHSIZE=64
## SAVE CONFIG
export SAVE_DIR_ROOT=''
export SAVE_DIR_NAME=web-air
export SAVE_DIR=${SAVE_DIR_ROOT}/${SAVE_DIR_NAME}
if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

export SAVE_DIR1=${SAVE_DIR}/stage1
if [ ! -d ${SAVE_DIR1} ]; then
    mkdir -p ${SAVE_DIR1}
fi

python3 -W ignore -u train.py --seed 0 --use_fewshot --margin 0.5 --relation_nobp --use_temperature_density \
--root_dir ${root_dir} --use_soft_label \
--root_dir_t ${root_dir_t} --pathlist_t ${pathlist_t} \
--arch "bcnn" --workers 8 --epochs 200 --pretrained --frozen_encoder_epoch 5 \
--init-proto-epoch 20 --batch-size ${N_BATCHSIZE} --relation_clean_epoch 26 \
--lr 1e-4 --weight-decay 1e-8 --pre_relation \
--exp-dir ${SAVE_DIR1} \
--num-class ${N_CLASSES} --low-dim 128 --alpha 0.5 \
--moco_queue 2048 --start_clean_epoch 25 --pseudo_th 0.6 --dist_th 20 \
--w-inst 1 --w-recn 1 --w-proto 1 --w-relation 1 --w-cls-lowdim 1 \
--dist-url 'tcp://localhost:12222' --multiprocessing-distributed --world-size 1 --rank 0

export SAVE_DIR2=${SAVE_DIR}/stage2
if [ ! -d ${SAVE_DIR2} ]; then
    mkdir -p ${SAVE_DIR2}
fi

export resume_path=${SAVE_DIR1}/checkpoint_latest.tar

python3 -W ignore -u train.py --seed 0 --use_fewshot --margin 0.5 --relation_nobp --use_temperature_density \
--root_dir ${root_dir} --use_soft_label \
--root_dir_t ${root_dir_t} --pathlist_t ${pathlist_t} \
--arch "bcnn" --workers 8 --epochs 20 --pretrained --frozen_encoder_epoch 0 \
--init-proto-epoch 0 --batch-size ${N_BATCHSIZE} --relation_clean_epoch 20 \
--lr 1e-4 --weight-decay 1e-8 --warmup_epoch 0 \
--exp-dir ${SAVE_DIR2} --ft_relation --start-epoch 1 \
--num-class ${N_CLASSES} --low-dim 128 --resume ${resume_path} \
--moco_queue 2048 --start_clean_epoch 0 --pseudo_th 0.6 --dist_th 20 \
--w-inst 1 --w-recn 1 --w-proto 1 --w-relation 1 --w-cls-lowdim 1 --alpha 0.5 \
--dist-url 'tcp://localhost:12223' --multiprocessing-distributed --world-size 1 --rank 0

export SAVE_DIR3=${SAVE_DIR}/stage3
if [ ! -d ${SAVE_DIR3} ]; then
    mkdir -p ${SAVE_DIR3}
fi

export resume_path=${SAVE_DIR2}/checkpoint_latest.tar

python3 -W ignore -u train.py --seed 0 --use_fewshot --margin 0.5 --relation_nobp --use_temperature_density \
--root_dir ${root_dir} --use_soft_label --clean_fusion \
--root_dir_t ${root_dir_t} --pathlist_t ${pathlist_t} \
--arch "bcnn" --workers 8 --epochs 200 --pretrained --frozen_encoder_epoch 5 \
--init-proto-epoch 20 --batch-size ${N_BATCHSIZE} --relation_clean_epoch 26 \
--lr 1e-4 --weight-decay 1e-8 --start-epoch 27 \
--exp-dir ${SAVE_DIR3} --resume ${resume_path} --update-relation-freq 1 \
--num-class ${N_CLASSES} --low-dim 128 \
--moco_queue 2048 --start_clean_epoch 25 --pseudo_th 0.6 --dist_th 20 --alpha 0.5 \
--w-inst 1 --w-recn 1 --w-proto 1 --w-relation 1 --w-cls-lowdim 1 \
--dist-url 'tcp://localhost:12224' --multiprocessing-distributed --world-size 1 --rank 0
