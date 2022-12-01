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
export root_dir='./dataset/WebFG496/web-car'
export root_dir_t='./dataset/FGVC/stanford_cars/cars_train'
export pathlist_t='./dataset/FGVC/stanford_cars/fewshot_1_shot.txt'
export N_CLASSES=196
export N_BATCHSIZE=64
## SAVE CONFIG
export SAVE_DIR_ROOT='./results'
export SAVE_DIR_NAME=web-car
export SAVE_DIR=${SAVE_DIR_ROOT}/${SAVE_DIR_NAME}
if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

export SAVE_DIR_EVAL=${SAVE_DIR}/eval
if [ ! -d ${SAVE_DIR_EVAL} ]; then
    mkdir -p ${SAVE_DIR_EVAL}
fi

export MODEL_DIR=${SAVE_DIR}/stage3/checkpoint_best.tar

python3 -W ignore -u eval.py --seed 0 --use_fewshot \
--root_dir ${root_dir} \
--root_dir_t ${root_dir_t} --pathlist_t ${pathlist_t} \
--arch "bcnn" --workers 8 --pretrained \
--batch-size ${N_BATCHSIZE} \
--exp-dir ${SAVE_DIR_EVAL} \
--num-class ${N_CLASSES} --low-dim 128 \
--moco_queue 2048 --pseudo_th 0.6 --dist_th 20 --resume ${MODEL_DIR}
