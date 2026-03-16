# ViTTT 프로젝트 작업 로그

- 프로젝트: ViT^3 (ViTTT) - Test-Time Training for Vision Transformers
- 논문: https://arxiv.org/abs/2512.01643
- 저자: Dongchen Han
- 작업 기간: 2026-02-25 ~ 2026-03-15 (진행 중)

---

## 1. 프로젝트 개요

ViT^3는 Vision Transformer에 Test-Time Training(TTT)을 적용한 모델이다.
추론 시 각 샘플에 대해 self-supervised reconstruction loss로 모델 가중치를 업데이트하는 방식.

### 핵심 구조
- **TTT Block** (`vittt/models/ttt_block.py`): 두 개의 병렬 브랜치로 구성
  - Branch 1: Simplified SwiGLU (fc → gate → fc)
  - Branch 2: 3x3 Depthwise Convolution
  - Self-supervised task: Key(k)로부터 Value(v)를 예측하는 reconstruction loss
  - Hand-derived gradients: torch.autograd.backward 대신 closed-form gradient 사용 (per-head non-scalar loss 처리)
- **H-ViTTT** (`vittt/models/h_vittt.py`): Hierarchical 구조 (4-stage pyramid)
  - dim=[64, 128, 320, 512], depths=[1, 3, 9, 4], num_heads=[2, 4, 10, 16]
- **MESA** (Model EMA Self-distillation): EMA 모델을 teacher로 사용한 self-distillation
  - Loss = CE(output, label) + CE(output, ema_soft_label) * mesa_ratio
  - epoch 75(전체의 25%) 이후부터 활성화

---

## 2. 환경 설정

### 가상환경
- Conda 환경명: `vittt`
- Python: 3.x (mamba 사용)
- 활성화: `mamba activate vittt`

### 패키지 마이그레이션 (이전 세션에서 완료)
| 패키지 | 원본 버전 | 마이그레이션 버전 |
|--------|-----------|-------------------|
| PyTorch | < 2.0 | **2.8.0** |
| timm | 0.4.12 | **1.0.25** |

### 추가 설치 패키지
```bash
pip install termcolor  # logger.py에서 사용
```

---

## 3. 코드 수정 내역

### 3.1 timm 마이그레이션 (이전 세션)

| 파일 | 변경 내용 |
|------|-----------|
| `vittt/models/vittt.py` | timm imports 업데이트, `**kwargs` 추가 to VisionTransformer.__init__ |
| `vittt/models/h_vittt.py` | `timm.models.layers` → `timm.layers` |
| `vittt/models/ttt_block.py` | `timm.models.layers` → `timm.layers` |
| `vittt/data/build.py` | `_pil_interp` → `str_to_interp_mode` |
| `vittt/lr_scheduler.py` | import 경로 업데이트 |

### 3.2 PyTorch 2.8 마이그레이션

| 파일 | 변경 내용 |
|------|-----------|
| `vittt/main.py` | `torch.cuda.amp` → `torch.amp` (autocast, GradScaler에 device 인자 추가) |
| `vittt/main_ema.py` | 동일한 torch.amp 변경 |
| `vittt/utils.py` | `torch.load`에 `weights_only=False` 추가 (line 19, 40) |
| `vittt/utils_ema.py` | `torch.load`에 `weights_only=False` 추가 (line 20, 43) |

### 3.3 스크립트 경로 수정

| 파일 | 변경 내용 | 이유 |
|------|-----------|------|
| `vittt/run_val.sh` | `${ROOT_DIR}/main.py` → `${SCRIPT_DIR}/main.py` | main.py가 vittt/ 안에 있음 |
| `vittt/run_val.sh` | `${ROOT_DIR}/main_ema.py` → `${SCRIPT_DIR}/main_ema.py` | 동일 |
| `vittt/run_train.sh` | `${ROOT_DIR}/main_ema.py` → `${SCRIPT_DIR}/main_ema.py` | 동일 |

### 3.4 EMA 로딩 문제 해결 (`vittt/main_ema.py`)

**문제**: timm의 `ModelEma` 내부에서 `torch.load`를 `weights_only=True`(PyTorch 2.6+ 기본값)로 호출하여 yacs CfgNode 포함 체크포인트 로드 실패

**해결**:
```python
# 변경 전
model_ema = ModelEma(model, decay=args.model_ema_decay,
    device='cpu' if args.model_ema_force_cpu else '',
    resume=config.MODEL.RESUME)

# 변경 후
from timm.utils import accuracy, AverageMeter, ModelEma, unwrap_model

model_ema = ModelEma(model, decay=args.model_ema_decay,
    device='cpu' if args.model_ema_force_cpu else '',
    resume='')  # timm 내부 로드 우회
if config.MODEL.RESUME:
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)
    if 'state_dict_ema' in checkpoint:
        ema_model = unwrap_model(model_ema.ema)  # DDP wrapper 제거
        ema_model.load_state_dict(checkpoint['state_dict_ema'])
        logger.info(f"=> loaded EMA state from '{config.MODEL.RESUME}'")
    del checkpoint
    torch.cuda.empty_cache()
```

**핵심 포인트**: `unwrap_model`로 DDP wrapper를 벗겨야 state_dict 키가 매칭됨.

### 3.5 ttt_block.py 문서화 및 테스트 (이전 세션)
- TTT 클래스에 상세한 docstring 추가
- 플로우차트 마크다운 생성
- 유닛 테스트 추가 (`vittt/temp.py`)

---

## 4. Validation 결과

### 4.1 ViTTT-T (기본 모델)
```
모델: ViTTT-T
체크포인트: vittt/ckpt/ViTTT-T.pth
스크립트: main.py + cfgs/vittt_t.yaml
결과: Acc@1 76.530%, Acc@5 93.284% (50,000 images)
```

### 4.2 H-ViTTT-T-mesa (Hierarchical + MESA)
```
모델: H-ViTTT-T-mesa
체크포인트: vittt/ckpt/H-ViTTT-T-mesa.pth
스크립트: main_ema.py + cfgs/h_vittt_t_mesa.yaml
결과:
  - Base model: Acc@1 84.054%, Acc@5 96.794%
  - EMA model:  Acc@1 84.058%, Acc@5 96.886%
```

---

## 5. 현재 진행 중: H-ViTTT-T-mesa 학습 (from scratch)

### 5.1 1차 학습 설정 (BS=128, gradient accumulation 없음)
```
GPU: 1대 (nproc_per_node=1)
Batch size: 128 (per GPU = total)
Epochs: 300
LR: 1.25e-4 (= 5e-4 * 128/512, linear scaling)
EMA decay: 0.999987
MESA: epoch 75부터 활성화 (ratio=1.0)
AMP: enabled
VRAM 사용량: 16,308 MB
Epoch당 시간: ~37분
```

**1차 학습 최종 결과**: Network Best: 82.81%, EMA Best: 83.13% (논문 84.06% 대비 -0.93%p)

### 5.2 학습 추이 (2026-02-26 기준, epoch 23 완료)

| Epoch | Base Acc@1 | Base Acc@5 | EMA Acc@1 | 비고 |
|-------|-----------|-----------|----------|------|
| 1 | 1.10% | 4.36% | 0.10% | Warmup 시작 |
| 2 | 3.07% | 9.78% | 0.10% | |
| 3 | 5.90% | 17.07% | 0.10% | |
| 4 | 10.43% | 26.54% | 0.10% | |
| 5 | 15.82% | 35.78% | 0.10% | |
| 6 | 22.14% | 45.57% | 0.10% | |
| 7 | 28.20% | 53.33% | 0.10% | |
| 8 | 33.80% | 59.76% | 0.10% | |
| 9 | 39.07% | 65.51% | 0.10% | |
| 10 | 44.07% | 70.22% | 0.10% | |
| 11 | 47.07% | 72.92% | 0.10% | |
| 12 | 51.13% | 76.40% | 0.10% | |
| 13 | 53.98% | 78.98% | 0.11% | |
| 14 | 56.72% | 81.04% | 0.17% | |
| 15 | 58.49% | 82.40% | 0.11% | |
| 16 | 60.82% | 84.02% | 0.13% | |
| 17 | 61.56% | 84.43% | 0.15% | |
| 18 | 63.07% | 85.52% | 0.17% | |
| 19 | 64.16% | 86.34% | 0.20% | |
| 20 | 65.43% | 87.26% | 0.24% | Warmup 종료 |
| 21 | 66.26% | 87.77% | 0.35% | |
| 22 | 66.92% | 88.19% | 0.52% | |
| 23 | 68.13% | 88.87% | 0.98% | |

### 5.3 1차 학습 분석
- 1 GPU, BS=128 → effective batch size가 논문의 1/4 → LR도 1/4로 scaling
- **0.93%p 차이의 주된 원인**: effective batch size 차이 (128 vs 512)
- Normalization 영향은 미미 (주로 LayerNorm 사용, BatchNorm은 Stem과 출력부만, 원본도 SyncBN 미사용)

---

## 5-2. 2차 학습: Gradient Accumulation으로 논문 재현 (2026-03-15~)

### 5-2.1 Gradient Accumulation 구현

**목적**: 논문의 effective batch size를 재현하여 논문 결과(84.06%) 달성
**참조 구현**: SPANetV2 (`SPANetV2-official-main/image_classification/`)

#### 수정 파일 요약

| 파일 | 변경 내용 |
|------|-----------|
| `vittt/config.py` (L80-81) | `_C.TRAIN.GRAD_ACCUM_STEPS = 1` 기본값 추가 |
| `vittt/config.py` (L204-206) | `getattr(args, 'grad_accum_steps', None)`으로 CLI→config 반영 (`main.py` 호환 유지) |
| `vittt/main_ema.py` (L56-57) | `--grad-accum-steps` argparse 인자 추가 |
| `vittt/main_ema.py` (L92) | LR scaling에 `effective_batch_size = BS * world_size * GRAD_ACCUM_STEPS` 반영 |
| `vittt/main_ema.py` (L105) | EMA decay scaling에 effective BS 반영 |
| `vittt/main_ema.py` (L131-136) | LR scheduler를 optimizer step 기준으로 변경 + 로깅 |
| `vittt/main_ema.py` (L235-330) | `train_one_epoch` 전면 재작성 (accumulation 로직) |
| `vittt/run_train.sh` (L24-36) | SPANetV2 패턴 변수 + `--grad-accum-steps`, `--tag` CLI 전달 |

#### 핵심 구현 패턴 (`train_one_epoch`)
```python
# 매 micro-batch
loss = criterion(outputs, targets)
scaler.scale(loss / grad_accum_steps).backward()  # loss 나눠서 누적

# accumulation boundary에서만
if (idx + 1) % grad_accum_steps == 0:
    scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm_(model.parameters(), CLIP_GRAD)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    lr_scheduler.step_update(...)
    model_ema.update(model)  # EMA도 optimizer step 후에만
```

#### 설계 포인트
- `config.py`에서 `getattr` 사용: `main.py`는 `--grad-accum-steps` 인자가 없으므로 호환성 유지
- MESA soft label은 매 micro-batch마다 계산 (다른 샘플), EMA 가중치 갱신은 accumulation boundary에서만
- `n_iter_per_epoch = len(data_loader) // grad_accum_steps`: LR scheduler는 optimizer step 기준
- 마지막 불완전 accumulation group은 표준적으로 버림

### 5-2.2 논문 원본 학습 설정 (참고)
논문 원문 (Sec 5.1): "The total batch size is 4096 and initial learning rate is set to 4×10⁻³."
```
Effective batch size: 4096
LR: 4e-3
Optimizer: AdamW, weight decay 0.05
Epochs: 300, warmup 20 epochs, cosine decay
Augmentation: RandAugment, Mixup, CutMix, random erasing
MESA: 적용 (H-ViTTT-T‡)
```

### 5-2.3 LR Linear Scaling 관계
```python
# main_ema.py의 LR scaling 로직
linear_scaled_lr = BASE_LR * effective_batch_size / 512.0

# config.py 기본값: BASE_LR = 5e-4
# BS=512  → LR = 5e-4 × 512/512  = 5e-4
# BS=4096 → LR = 5e-4 × 4096/512 = 4e-3  ← 논문과 일치
```

### 5-2.4 2차 학습 설정 (BS=512, 진행 중)
```
GPU: A6000 × 2 (DDP) + A100 (초기)
Micro batch size: 256
Grad accumulation steps: 1 (DDP 2GPU)
Effective batch size: 512 (논문의 1/8)
LR: 5e-4 (linear scaling 적용)
Optimizer steps/epoch: ~2502
MESA: epoch 75부터 활성화
AMP: enabled
Tag: grad_accum_bs512_on_A100
출력 경로: output/h_vittt_tiny/grad_accum_bs512_on_A100/
```

**BS=512 vs 논문 BS=4096 차이 분석**:
- LR scaling은 올바르게 적용됨 (5e-4 @ BS=512 = 4e-3 @ BS=4096)
- 다만 batch size 자체가 학습 dynamics에 영향: 큰 배치는 gradient noise가 적어 안정적
- 1차 학습(BS=128) 대비 개선 기대, 논문 대비 -0.5~1.0%p 차이 예상

### 5-2.5 3차 학습 설정 (BS=4096, 예정)
```
GPU: A6000 × 2 (DDP)
Micro batch size: 256
Grad accumulation steps: 8
Effective batch size: 4096 (= 256 × 2 × 8, 논문과 동일)
LR: 4e-3 (= 5e-4 × 4096/512, 논문과 동일)
MESA: epoch 75부터 활성화
AMP: enabled
Tag: grad_accum_bs4096_on_A6000
예상 epoch 시간: ~130분
예상 총 학습 시간: ~27일
```

### 5-2.6 논문 재현 신뢰도
| 설정 | 재현 신뢰도 | 예상 정확도 |
|------|-----------|-----------|
| BS=128 (1차, 완료) | ~90% | 82.81% (실측) |
| BS=512 (2차, 진행 중) | ~93-95% | 83.0~83.5% 예상 |
| BS=4096 (3차, 예정) | ~97-99% | 83.7~84.3% 예상 |

### 5-2.4 서버 환경 이슈 및 최적화 (2026-03-15)

#### GPU 드라이버 호환성
- 서버 드라이버: **525.147.05** → CUDA 12.0까지만 지원
- PyTorch 2.8.0+cu128 설치 불가 → **PyTorch 2.7.0+cu118**로 다운그레이드
- `requirements.txt` 업데이트 완료

#### CUDA_DEVICE_ORDER 문제
- `CUDA_VISIBLE_DEVICES=5`로 설정해도 nvidia-smi 기준 다른 GPU에서 실행되는 현상 발생
- **원인**: CUDA 내부 GPU 열거 순서가 nvidia-smi(PCI bus 순서)와 다름
- **해결**: `run_train.sh`에 `export CUDA_DEVICE_ORDER=PCI_BUS_ID` 추가

#### GPU 선택: A100 80GB (GPU 5)
서버에 RTX A6000(48GB) 7대 + A100 80GB(GPU 5) 1대 존재

| 항목 | A100 80GB | RTX A6000 |
|------|-----------|-----------|
| FP16 Tensor | 312 TFLOPS | 155 TFLOPS |
| 메모리 대역폭 | 2,039 GB/s (HBM2e) | 768 GB/s (GDDR6) |
| VRAM | 80GB | 48GB |

AMP(FP16) 학습이므로 A100이 연산 약 2배 빠름. 단, 실제 속도는 I/O 병목에 지배됨.

#### HDD I/O 병목 발견 및 해결
- **문제**: ImageNet이 HDD(`/data1`, `rotational=1`)에 저장되어 있어 DataLoader 워커가 전부 `D`(disk sleep) 상태로 멈춤
- **증상**: "Start training" 이후 10분 이상 GPU 사용률 0%, 로그 없음
- **BS=512**: HDD 랜덤 IOPS 한계(~100-200)로 512장 동시 로딩 불가 → hang
- **BS=128**: HDD에서도 동작하지만 느림 (iteration 0.6초, epoch ~104분 → 300 epochs ≈ 22일)

**해결: tmpfs (RAM disk)에 ImageNet 복사**
```bash
# /dev/shm은 RAM 기반 파일시스템 (읽기 속도 수십 GB/s)
cp -r /data1/members/yg/ImageNet2012 /dev/shm/ImageNet2012
# run_train.sh에서 --data-path /dev/shm/ImageNet2012 로 변경
```

| 항목 | 값 |
|------|-----|
| 서버 RAM | 503GB (가용 483GB) |
| ImageNet 크기 | 152GB (train 140GB + val 13GB) |
| /dev/shm 용량 | 252GB |
| 복사 후 남는 RAM disk | ~84GB |
| 서버 재부팅 시 | 사라짐 (원본은 HDD에 유지) |

**주의사항**:
- 다른 사용자가 RAM을 대량 사용하면 tmpfs가 swap으로 밀릴 수 있음
- GPU 5(A100)는 선점 보호가 없으므로 팀에 사용 중임을 공유 필요
- 체크포인트는 HDD(`output/`)에 자동 저장되므로 크래시 시 `--resume`으로 복구 가능

---

## 6. 주요 파일 구조

```
vittt/
├── main.py              # ViTTT 학습/평가 (기본 모델용)
├── main_ema.py           # H-ViTTT + EMA + MESA 학습/평가
├── config.py             # 기본 설정 (yacs CfgNode)
├── optimizer.py          # AdamW optimizer + weight decay/lr 분리
├── lr_scheduler.py       # Cosine LR scheduler
├── data/build.py         # ImageNet 데이터로더
├── logger.py             # termcolor 기반 로거
├── utils.py              # 체크포인트 저장/로드 (기본 모델)
├── utils_ema.py          # 체크포인트 저장/로드 (EMA 모델 포함)
├── run_train.sh          # 학습 실행 스크립트
├── run_val.sh            # 검증 실행 스크립트
├── cfgs/
│   ├── vittt_t.yaml      # ViTTT-Tiny 설정
│   ├── h_vittt_t.yaml    # H-ViTTT-Tiny 설정 (MESA 없음)
│   └── h_vittt_t_mesa.yaml  # H-ViTTT-Tiny + MESA 설정
├── models/
│   ├── __init__.py       # build_model() 함수
│   ├── vittt.py          # ViTTT 모델 정의
│   ├── h_vittt.py        # Hierarchical ViTTT 모델 정의
│   └── ttt_block.py      # TTT Block 핵심 구현
├── ckpt/
│   ├── ViTTT-T.pth       # 사전 학습 체크포인트
│   └── H-ViTTT-T-mesa.pth
└── output/
    └── h_vittt_tiny/default/
        └── log_rank0.txt  # 학습 로그
```

---

## 7. Config 시스템

4단계 우선순위:
1. `config.py` (기본값) → BATCH_SIZE=128, BASE_LR=5e-4, EPOCHS=300 등
2. YAML 파일 (모델별 설정) → MODEL.TYPE, DROP_PATH_RATE, MESA 등
3. CLI 인자 (실행 시) → --batch-size, --data-path, --amp 등
4. `main_ema.py` 내 자동 계산 → LR linear scaling: `lr = BASE_LR * BS * world_size / 512`

---

## 8. 해결한 에러 목록

| # | 에러 | 원인 | 해결 |
|---|------|------|------|
| 1 | `main.py not found at ${ROOT_DIR}/main.py` | main.py가 vittt/ 안에 있는데 ROOT_DIR(상위)에서 찾음 | `${ROOT_DIR}` → `${SCRIPT_DIR}` |
| 2 | `ModuleNotFoundError: No module named 'termcolor'` | logger.py 의존성 누락 | `pip install termcolor` |
| 3 | `torch.load` weights_only 에러 | PyTorch 2.6+ 기본값 변경 (weights_only=True) | `weights_only=False` 추가 |
| 4 | timm ModelEma 내부 torch.load 실패 | timm 내부에서 weights_only 제어 불가 | `resume=''` + 수동 로드 |
| 5 | EMA state_dict 키 불일치 | DDP wrapper가 `module.` prefix 추가 | `unwrap_model()` 사용 |
| 6 | `_pil_interp` 제거됨 (timm 1.0) | timm API 변경 | `str_to_interp_mode` 사용 |
| 7 | `torch.cuda.amp` deprecated | PyTorch 2.x에서 변경 | `torch.amp` + device 인자 |
| 8 | CUDA_VISIBLE_DEVICES와 nvidia-smi GPU 번호 불일치 | CUDA 내부 열거 순서가 PCI bus 순서와 다름 | `export CUDA_DEVICE_ORDER=PCI_BUS_ID` 추가 |
| 9 | 학습 시작 후 hang (GPU 0%, 로그 없음) | HDD 랜덤 I/O 병목 (BS=512일 때 DataLoader 전원 D state) | BS=128 + grad_accum=4, tmpfs로 ImageNet 복사 |
| 10 | PyTorch 2.8.0+cu128 설치 불가 | 서버 드라이버 525 → CUDA 12.0까지만 지원 | PyTorch 2.7.0+cu118로 다운그레이드 |

---

## 9. 다음 할 일 (TODO)

- [x] 1차 학습 완료 (BS=128, 300 epochs) → Network 82.81%, EMA 83.13%
- [x] Gradient accumulation 구현 (effective BS=512)
- [x] 서버 환경 최적화 (드라이버 호환, CUDA_DEVICE_ORDER, tmpfs)
- [x] 논문 원본 BS=4096 확인 (기존 코드 기본값 BS=512는 논문의 1/8)
- [ ] 2차 학습 진행 중 (BS=512, A6000×2 DDP, epoch 32/300)
- [ ] 2차 학습 완료 후 정확도 확인 (논문 84.06% 대비)
- [ ] 3차 학습 검토 (BS=4096, 논문 동일 조건, ~27일 소요)
- [ ] EMA 모델 성능 확인 (MESA epoch 75 이후 추이)

---

## 10. 참고: shell hook 필요성

`bash script.sh`로 실행하면 `.bashrc`가 로드되지 않아 `mamba activate`가 정의되지 않음.
따라서 스크립트 상단에 아래를 추가해야 함:
```bash
eval "$(mamba shell hook --shell bash)"
mamba activate vittt
```
