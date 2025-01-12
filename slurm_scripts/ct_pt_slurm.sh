#!/bin/bash
#SBATCH --job-name=camt5-ctpt
#SBATCH --output=//home/osikjs/CAMT5/.caslurm-logs/camt5-ctpt-%j.out
#SBATCH --error=//home/osikjs/CAMT5/.slurm-logs/camt5-ctpt-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --exclude=node1,node2,node3

PORT=25682
LOG_INTERVAL=15
GPU_LOG_FILE="/home/osikjs/CAMT5/.slurm-gpu-logs/gpu-usage-${SLURM_JOB_ID}.log"

log_gpu_usage() {
    while true; do
        {
            echo "Timestamp: $(date)"
            nvidia-smi
            echo "----------------------------------------"
        } > "$GPU_LOG_FILE"
        sleep $LOG_INTERVAL
    done
}

log_gpu_usage &

source /opt/miniconda3/bin/activate /home/osikjs/miniconda3/envs/camt5
bash ./scripts/ct_pretrain.sh -e ct_pt_camt5 -p $PORT -g 1

kill %1