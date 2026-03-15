#!/bin/bash
set -e  # 오류 발생 시 스크립트 즉시 중단

echo "Initializing environment..."
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제 
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # nvidia-smi 번호와 CUDA 번호 일치시킴

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
export CUDA_VISIBLE_DEVICES="5"  # GPU IDs to use
ALL_BATCH_SIZE=512  # Target effective batch size (matches paper)
NUM_GPU=1
GRAD_ACCUM_STEPS=1  # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

torchrun --nproc_per_node=$NUM_GPU "${SCRIPT_DIR}/main_ema.py" \
    --cfg "${SCRIPT_DIR}/cfgs/h_vittt_t_mesa.yaml" \
    --data-path /dev/shm/ImageNet2012 \
    --output "${SCRIPT_DIR}/output" \
    --batch-size $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
    --amp \
    --tag grad_accum_bs512_on_A100


