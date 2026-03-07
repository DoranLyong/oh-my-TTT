#!/bin/bash
set -e  # 오류 발생 시 스크립트 즉시 중단

echo "Initializing environment..."
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제

eval "$(mamba shell hook --shell bash)"
mamba activate vittt  # ViTTT 가상환경 활성화

echo "Running model analysis..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# -- H-ViT³-Tiny 모델 분석 (파라미터, FLOPs, 속도):
python "${SCRIPT_DIR}/temp.py" --cfg "${SCRIPT_DIR}/cfgs/h_vittt_t_mesa.yaml"
