#!/bin/bash
set -e  # 오류 발생 시 스크립트 즉시 중단

echo "Initializing environment..."
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제 
eval "$(mamba shell hook --shell bash)"
mamba activate test3r  


python demo_ttt.py --images demo/data/1.png demo/data/2.png demo/data/3.png demo/data/4.png demo/data/5.png demo/data/6.png demo/data/7.png demo/data/8.png