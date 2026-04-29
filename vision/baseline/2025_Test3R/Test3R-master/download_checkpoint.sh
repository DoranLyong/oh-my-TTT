#!/bin/bash

# 1. 변수 설정
REPO_ID="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
TAG="dust3r_vit_large"

# 2. 모델 다운로드
# --local-dir-use-symlinks False: 캐시 심볼릭 링크 대신 '실제 파일'을 다운로드합니다.
hf download $REPO_ID \
  --repo-type model \
  --local-dir checkpoints/${TAG} \
  --max-workers 4