#!/bin/bash

# Default Settings
N_NODE=1
N_GPU_PER_NODE=1
MASTER_PORT=25680
EXP_NAME="pt"

while (( "$#" )); do
  case "$1" in
    -n|--nnodes)
      N_NODE=$2
      shift 2
      ;;
    -g|--nproc_per_node)
      N_GPU_PER_NODE=$2
      shift 2
      ;;
    -p|--master_port)
      MASTER_PORT=$2
      shift 2
      ;;
    -e|--exp_name)
      EXP_NAME=$2
      shift 2
      ;;
    *)
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
  esac
done

torchrun --nnodes=${N_NODE} --nproc_per_node=${N_GPU_PER_NODE} --master_port=${MASTER_PORT} main.py \
    task=train/pretrain task/train/exp=${EXP_NAME}
