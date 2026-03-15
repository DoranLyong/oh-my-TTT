#!/bin/bash
set -e  # 오류 발생 시 스크립트 즉시 중단

echo "Initializing environment..."
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제 

eval "$(mamba shell hook --shell bash)"
mamba activate vittt  # ViTTT 가상환경 활성화

echo "Running training script..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# -- Train ViTTT-T on ImageNet:

#torchrun --nproc_per_node=1 "${SCRIPT_DIR}/main.py" \
#    --cfg "${SCRIPT_DIR}/cfgs/vittt_t.yaml" \
#    --data-path /mnt/dataset/ImageNet2012 \
#    --output "${SCRIPT_DIR}/output" \
#    --amp 

# Train H-ViT3 on ImageNet:
torchrun --nproc_per_node=1 "${SCRIPT_DIR}/main_ema.py" \
    --cfg "${SCRIPT_DIR}/cfgs/h_vittt_t_mesa.yaml" \
    --data-path /mnt/dataset/ImageNet2012 \
    --output "${SCRIPT_DIR}/output" \
    --amp


