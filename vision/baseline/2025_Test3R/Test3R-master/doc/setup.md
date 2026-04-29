# Setup 

## Prerequisites
* A linux 64-bits architecture (i.e. linux-64 platform in mamba info).
* A NVIDIA GPU with at least 32 Gb of VRAM.


## 1. Setup Python Environment 

```bash
#-- create Test3R environment
mamba env create -f environments/default.yml
mamba activate test3r

pip install uv 

#-- install Test3R and core dependencies
uv pip install -r requirements.txt


#-- Optional, compile the cuda kernels for RoPE (as in CroCo v2).
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```


## 2. Getting Checkpoints 

* Run `download_checkpoint.sh`

```bash
# 1. 변수 설정
REPO_ID="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
TAG="dust3r_vit_large"

# 2. 모델 다운로드
# --local-dir-use-symlinks False: 캐시 심볼릭 링크 대신 '실제 파일'을 다운로드합니다.
hf download $REPO_ID \
  --repo-type model \
  --local-dir checkpoints/${TAG} \
  --max-workers 4
```