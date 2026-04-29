#!/bin/bash
set -e  # 오류 발생 시 스크립트 즉시 중단

echo "Initializing environment..."
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # nvidia-smi 번호와 CUDA 번호 일치시킴

eval "$(mamba shell hook --shell bash)"
mamba activate vittt  # ViTTT 가상환경 활성화

echo "Running validation script..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# -- Evaluate ViTTT-T on ImageNet:

#torchrun --nproc_per_node=1 "${SCRIPT_DIR}/main.py" \
#    --cfg "${SCRIPT_DIR}/cfgs/vittt_t.yaml" \
#    --data-path /mnt/dataset/ImageNet2012 \
#    --output "${SCRIPT_DIR}/output" \
#    --eval \
#    --resume "${SCRIPT_DIR}/ckpt/ViTTT-T.pth"

# Evaluate H-ViT3 on ImageNet:
export CUDA_VISIBLE_DEVICES="0"  # GPU IDs to use

torchrun --nproc_per_node=1 --master_port=29500 "${SCRIPT_DIR}/main_ema.py" \
    --cfg "${SCRIPT_DIR}/cfgs/h_vittt_t_mesa.yaml" \
    --data-path /dev/shm/ImageNet2012 \
    --output "${SCRIPT_DIR}/output" \
    --eval \
    --resume "${SCRIPT_DIR}/output/h_vittt_tiny/grad_accum_bs4096/max_ema_acc.pth"
#    --resume "${SCRIPT_DIR}/ckpt/max_ema_acc.pth"
