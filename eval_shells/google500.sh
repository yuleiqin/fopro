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
export pathlist="./dataset/webvision1k/filelist/train_filelist_google_500_usable_tf.txt"
export root_dir_test_web="./dataset/webvision1k/tfrecord_webvision_val"
export pathlist_test_web="./dataset/webvision1k/filelist/val_webvision_500_usable_tf.txt"
export root_dir_test_target="./dataset/webvision1k/tfrecord_imgnet_val"
export pathlist_test_target="./dataset/webvision1k/filelist/val_imagenet_500_usable_tf.txt"
export root_dir_t="./dataset/webvision1k/tfrecord_webvision_train"
export pathlist_t="./dataset/imgnet_g500/fewshot_1_shot.txt"
export N_CLASSES=500
export N_BATCHSIZE=256
## SAVE CONFIG
export SAVE_DIR_ROOT="./results"
export SAVE_DIR_NAME=web-g500
export SAVE_DIR=${SAVE_DIR_ROOT}/${SAVE_DIR_NAME}
if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

export SAVE_DIR_EVAL=${SAVE_DIR}/eval
if [ ! -d ${SAVE_DIR_EVAL} ]; then
    mkdir -p ${SAVE_DIR_EVAL}
fi

export MODEL_DIR=${SAVE_DIR}/stage3/checkpoint_best.tar

python3 -W ignore -u eval_imgnet.py --seed 0 --webvision --fast_eval 1 \
--root_dir ${root_dir} --pathlist_web ${pathlist} \
--root_dir_test_web ${root_dir_test_web} --pathlist_test_web ${pathlist_test_web} \
--root_dir_test_target ${root_dir_test_target} --pathlist_test_target ${pathlist_test_target} \
--root_dir_t ${root_dir_t} --pathlist_t ${pathlist_t} \
--arch "resnet50" --workers 8 --pretrained \
--batch-size ${N_BATCHSIZE} \
--exp-dir ${SAVE_DIR} --resume ${MODEL_DIR} \
--num-class ${N_CLASSES} --low-dim 128 \
--moco_queue 8192 --pseudo_th 0.6
