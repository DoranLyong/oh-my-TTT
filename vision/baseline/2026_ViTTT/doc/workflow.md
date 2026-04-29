# H-ViTTT Tiny (MESA) Training Workflow

> `run_train.sh` → `main_ema.py` → `h_vittt_tiny` 모델의 학습 과정을 PyTorch 코드 레벨에서 상세히 분석한 문서

---

## 1. 전체 학습 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────┐
│  run_train.sh                                               │
│  torchrun --nproc_per_node=1 main_ema.py                    │
│    --cfg cfgs/h_vittt_t_mesa.yaml                           │
│    --data-path /mnt/dataset/ImageNet2012                    │
│    --amp                                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  main_ema.py::main()                                        │
│                                                             │
│  1. DDP 초기화 (NCCL backend)                                │
│  2. Config 로드 (h_vittt_t_mesa.yaml)                        │
│  3. DataLoader 구성 (ImageNet, batch=128)                     │
│  4. Model 생성 (h_vittt_tiny, drop_path=0.2)                 │
│  5. Optimizer 구성 (AdamW)                                   │
│  6. LR Scheduler (Cosine)                                   │
│  7. Loss 함수 (SoftTargetCrossEntropy)                       │
│  8. Model EMA (decay=0.9996)                                │
│  9. Training Loop (300 epochs + MESA from 25%)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 핵심 Config 값 (h_vittt_t_mesa.yaml + defaults)

| 항목 | 값 |
|---|---|
| `MODEL.TYPE` | `h_vittt_tiny` |
| `MODEL.DROP_PATH_RATE` | `0.2` |
| `TRAIN.MESA` | `1.0` |
| `DATA.IMG_SIZE` | `224` |
| `DATA.BATCH_SIZE` | `128` |
| `TRAIN.EPOCHS` | `300` |
| `TRAIN.BASE_LR` | `5e-4` (linear scaled) |
| `TRAIN.OPTIMIZER.NAME` | `adamw` |
| `AUG.MIXUP` | `0.8` |
| `AUG.CUTMIX` | `1.0` |

---

## 3. 모델 아키텍처: `h_vittt_tiny`

### 3.1 모델 생성 (`models/build.py`)

```python
# h_vittt_tiny() 호출 시 생성되는 ViTTT 인스턴스:
model = ViTTT(
    dim     = [64, 128, 320, 512],   # 각 stage의 채널 수
    depths  = [1, 3, 9, 4],          # 각 stage의 TTTBlock 수
    num_heads = [2, 4, 10, 16],      # 각 stage의 head 수
    drop_path_rate = 0.2
)
```

### 3.2 전체 모델 구조 & Tensor Shape Flow

```
입력 이미지
  [B, 3, 224, 224]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Stem (Patch Embedding)                                  │
│                                                          │
│  Conv2d(3→32, k3, s2, p1) + BN + ReLU                   │
│    → [B, 32, 112, 112]                                   │
│  Conv2d(32→32, k3, s1, p1) + BN + ReLU                  │
│    → [B, 32, 112, 112]                                   │
│  Conv2d(32→32, k3, s1, p1) + BN + ReLU                  │
│    → [B, 32, 112, 112]                                   │
│  Conv2d(32→256, k3, s2, p1) + BN + ReLU                 │
│    → [B, 256, 56, 56]                                    │
│  Conv2d(256→64, k1) + BN                                 │
│    → [B, 64, 56, 56]                                     │
│  flatten(2).transpose(1,2)                               │
│    → [B, 3136, 64]                                       │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: BasicLayer                                     │
│  resolution=56×56, dim=64, depth=1, heads=2              │
│  head_dim = 64 // 2 = 32                                 │
│                                                          │
│  ┌── TTTBlock ×1 ──────────────────────────────────┐     │
│  │  입력: [B, 3136, 64]                             │     │
│  │                                                  │     │
│  │  1. CPE: DWConv(64, k3, p1)                      │     │
│  │     reshape → [B,64,56,56] → conv → flatten      │     │
│  │     + residual → [B, 3136, 64]                   │     │
│  │                                                  │     │
│  │  2. LayerNorm → TTT Block (상세: §4)              │     │
│  │     + DropPath + residual → [B, 3136, 64]        │     │
│  │                                                  │     │
│  │  3. LayerNorm → MLP (상세: §5)                    │     │
│  │     + DropPath + residual → [B, 3136, 64]        │     │
│  └──────────────────────────────────────────────────┘     │
│                                                          │
│  PatchMerging (downsample):                              │
│    reshape → [B, 64, 56, 56]                             │
│    Conv1x1(64→512) + BN + ReLU → [B, 512, 56, 56]       │
│    DWConv3x3(512, s2) + BN + ReLU → [B, 512, 28, 28]    │
│    Conv1x1(512→128) + BN → [B, 128, 28, 28]             │
│    flatten → [B, 784, 128]                               │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2: BasicLayer                                     │
│  resolution=28×28, dim=128, depth=3, heads=4             │
│  head_dim = 128 // 4 = 32                                │
│                                                          │
│  TTTBlock ×3: [B, 784, 128] → [B, 784, 128]             │
│                                                          │
│  PatchMerging:                                           │
│    Conv1x1(128→1280) → DWConv3x3(s2) → Conv1x1(→320)   │
│    → [B, 196, 320]                                       │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3: BasicLayer                                     │
│  resolution=14×14, dim=320, depth=9, heads=10            │
│  head_dim = 320 // 10 = 32                               │
│                                                          │
│  TTTBlock ×9: [B, 196, 320] → [B, 196, 320]             │
│                                                          │
│  PatchMerging:                                           │
│    Conv1x1(320→2048) → DWConv3x3(s2) → Conv1x1(→512)   │
│    → [B, 49, 512]                                        │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4: BasicLayer                                     │
│  resolution=7×7, dim=512, depth=4, heads=16              │
│  head_dim = 512 // 16 = 32                               │
│                                                          │
│  TTTBlock ×4: [B, 49, 512] → [B, 49, 512]               │
│  (downsample 없음 — 마지막 stage)                          │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Classification Head                                     │
│                                                          │
│  BatchNorm1d: [B, 49, 512] → transpose → [B, 512, 49]   │
│  AdaptiveAvgPool1d(1): → [B, 512, 1]                    │
│  Flatten: → [B, 512]                                     │
│  Linear(512 → 1000): → [B, 1000]                        │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Stage별 요약 테이블

| Stage | Resolution | Tokens (N) | Dim (C) | Heads | head_dim | Depth | Output Shape |
|-------|-----------|------------|---------|-------|----------|-------|--------------|
| Stem | 224→56 | 3136 | 64 | - | - | - | `[B, 3136, 64]` |
| 1 | 56×56 | 3136 | 64 | 2 | 32 | 1 | `[B, 784, 128]` |
| 2 | 28×28 | 784 | 128 | 4 | 32 | 3 | `[B, 196, 320]` |
| 3 | 14×14 | 196 | 320 | 10 | 32 | 9 | `[B, 49, 512]` |
| 4 | 7×7 | 49 | 512 | 16 | 32 | 4 | `[B, 49, 512]` |
| Head | - | 1 | 512→1000 | - | - | - | `[B, 1000]` |

---

## 4. TTT Block 상세 (핵심: Test-Time Training)

TTT Block은 ViTTT의 핵심 혁신으로, **forward pass 안에서 inner model을 학습(inner training)**시킨 뒤 그 업데이트된 weight로 query를 처리한다.

### 4.1 TTT Block Forward Flow (Stage 3 기준: dim=320, heads=10, head_dim=32)

```
입력: x [B, 196, 320]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  QKV Projection                                          │
│  Linear(320 → 320×3 + 32×3 = 1056)                      │
│  → split into 6 텐서:                                    │
│                                                          │
│  ┌─ SwiGLU Branch ─────────────────────────────────┐     │
│  │  q1: [B,196,320] → RoPE → [B,10,196,32]        │     │
│  │  k1: [B,196,320] → RoPE → [B,10,196,32]        │     │
│  │  v1: [B,196,320]        → [B,10,196,32]        │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  ┌─ 3×3 DWConv Branch ────────────────────────────┐     │
│  │  q2: [B,196,32] → [B,32,14,14]                 │     │
│  │  k2: [B,196,32] → [B,32,14,14]                 │     │
│  │  v2: [B,196,32] → [B,32,14,14]                 │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Inner Training (1 step gradient descent)                │
│                                                          │
│  ┌─ Branch 1: Simplified SwiGLU Inner Training ───┐     │
│  │                                                 │     │
│  │  초기 weights:                                   │     │
│  │    w1: [1, 10, 32, 32]  (학습 가능 파라미터)      │     │
│  │    w2: [1, 10, 32, 32]  (학습 가능 파라미터)      │     │
│  │                                                 │     │
│  │  Forward:                                       │     │
│  │    z1 = k1 @ w1     → [B, 10, 196, 32]         │     │
│  │    z2 = k1 @ w2     → [B, 10, 196, 32]         │     │
│  │    sig = sigmoid(z2)→ [B, 10, 196, 32]         │     │
│  │    a = z2 * sig     → [B, 10, 196, 32]         │     │
│  │                                                 │     │
│  │  Backward (hand-derived, 수식 직접 계산):         │     │
│  │    e = -v1 / N * scale                          │     │
│  │    g1 = k1ᵀ @ (e * a)           → [B,10,32,32] │     │
│  │    g2 = k1ᵀ @ (e*z1*(sig*(1+z2*(1-sig))))      │     │
│  │                                  → [B,10,32,32] │     │
│  │    g1 = g1 / (‖g1‖ + 1)  (gradient clipping)   │     │
│  │    g2 = g2 / (‖g2‖ + 1)                        │     │
│  │                                                 │     │
│  │  Step (lr=1.0):                                 │     │
│  │    w1' = w1 - g1    → [B, 10, 32, 32]          │     │
│  │    w2' = w2 - g2    → [B, 10, 32, 32]          │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  ┌─ Branch 2: 3×3 DWConv Inner Training ──────────┐     │
│  │                                                 │     │
│  │  초기 weight:                                    │     │
│  │    w3: [32, 1, 3, 3]  (학습 가능 파라미터)        │     │
│  │                                                 │     │
│  │  Backward (prod implementation):                │     │
│  │    e = -v2 / (H*W) * scale                      │     │
│  │    k_pad = pad(k2, 1)  → [B, 32, 16, 16]       │     │
│  │    for (dy, dx) in 3×3 offsets:                 │     │
│  │      dot = (k_shifted * e).sum(H,W)             │     │
│  │    g = stack(dots)     → [B*32, 1, 3, 3]        │     │
│  │    g = g / (‖g‖ + 1)                            │     │
│  │                                                 │     │
│  │  Step:                                          │     │
│  │    w3' = w3.repeat(B) - g → [B*32, 1, 3, 3]    │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Apply Updated Inner Model to Query                      │
│                                                          │
│  Branch 1 (SwiGLU):                                      │
│    x1 = (q1 @ w1') * SiLU(q1 @ w2')                     │
│    → [B, 10, 196, 32]                                    │
│    → transpose & reshape → [B, 196, 320]                 │
│                                                          │
│  Branch 2 (3×3 DWConv):                                  │
│    x2 = conv2d(q2, w3', padding=1, groups=B*32)          │
│    → [1, B*32, 14, 14]                                   │
│    → reshape → [B, 32, 196] → transpose → [B, 196, 32]  │
│                                                          │
│  Concat & Project:                                       │
│    x = cat([x1, x2], dim=-1) → [B, 196, 352]            │
│    x = Linear(352 → 320)     → [B, 196, 320]            │
└──────────────────────────────────────────────────────────┘
```

### 4.2 TTT Inner Training의 직관적 이해

```
┌────────────────────────────────────────────────────┐
│            전통 Attention vs TTT                    │
│                                                    │
│  Self-Attention:                                   │
│    output = softmax(Q·Kᵀ/√d) · V                  │
│    → K,V를 "메모리"로 사용, Q로 조회               │
│                                                    │
│  TTT:                                              │
│    1. K,V 쌍으로 inner model W를 학습 (1-step GD)  │
│       loss = reconstruction(W(K), V)               │
│    2. 학습된 W를 Q에 적용                           │
│       output = W'(Q)                               │
│    → inner model W가 "메모리", K→V 매핑을 학습      │
│    → 학습된 매핑을 Q에 적용하여 출력 생성            │
└────────────────────────────────────────────────────┘
```

---

## 5. MLP Block (with DWConv)

```
입력: x [B, 196, 320]
       │
       ▼
  Linear(320 → 1280)       → [B, 196, 1280]
  GELU                      → [B, 196, 1280]
  Dropout
       │
       ├──── reshape → [B, 1280, 14, 14]
       │     DWConv(1280, k3, p1, groups=1280)
       │     → [B, 1280, 14, 14]
       │     flatten → [B, 196, 1280]
       │
       └──── + (residual)  → [B, 196, 1280]
  GELU                      → [B, 196, 1280]
  Linear(1280 → 320)        → [B, 196, 320]
  Dropout                   → [B, 196, 320]
```

---

## 6. Training Loop (MESA 포함)

### 6.1 한 Epoch의 학습 흐름

```
for epoch in range(0, 300):
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  MESA 활성화 조건 판단                                     │
│                                                          │
│  if epoch >= 75 (= 25% of 300):                          │
│      mesa = 1.0   ← MESA 활성화                          │
│  else:                                                   │
│      mesa = -1.0  ← MESA 비활성화                         │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
  for (samples, targets) in data_loader:
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  1. Mixup/CutMix Augmentation                            │
│     samples: [128, 3, 224, 224]                          │
│     targets: [128] → (soft labels) [128, 1000]           │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  2. AMP autocast('cuda') context                         │
│                                                          │
│  ┌─ MESA 활성화 시 (epoch ≥ 75) ────────────────────┐    │
│  │                                                   │    │
│  │  (a) EMA 모델로 soft target 생성 (no grad):       │    │
│  │      ema_output = model_ema(samples)              │    │
│  │        → [128, 1000]                              │    │
│  │      ema_output = softmax(ema_output)             │    │
│  │        → [128, 1000]  (확률 분포)                  │    │
│  │                                                   │    │
│  │  (b) Student 모델 forward:                        │    │
│  │      outputs = model(samples)                     │    │
│  │        → [128, 1000]                              │    │
│  │                                                   │    │
│  │  (c) 복합 Loss 계산:                               │    │
│  │      loss = CE(outputs, targets)                  │    │
│  │           + CE(outputs, ema_output) × 1.0         │    │
│  │             ↑ ground truth    ↑ EMA distillation  │    │
│  │                                                   │    │
│  └───────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─ MESA 비활성화 시 (epoch < 75) ──────────────────┐    │
│  │  outputs = model(samples) → [128, 1000]           │    │
│  │  loss = CE(outputs, targets)                      │    │
│  └───────────────────────────────────────────────────┘    │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  3. Backward & Optimizer Step                            │
│                                                          │
│  scaler.scale(loss).backward()                           │
│  scaler.unscale_(optimizer)                              │
│  clip_grad_norm_(model.parameters(), max_norm=5.0)       │
│  scaler.step(optimizer)  ← AdamW                         │
│  scaler.update()                                         │
│  lr_scheduler.step_update(global_step)                   │
│                                                          │
│  4. EMA Update                                           │
│  model_ema.update(model)                                 │
│    ema_params = decay * ema_params + (1-decay) * params  │
└──────────────────────────────────────────────────────────┘
```

### 6.2 MESA (Model EMA as Soft Augmentation) 시각화

```
                    ┌──────────────┐
                    │  Input Image │
                    │[128,3,224,224]│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │                         │
              ▼                         ▼
     ┌────────────────┐       ┌────────────────┐
     │  Student Model │       │  EMA Teacher   │
     │  (학습 대상)     │       │  (inference_mode)│
     │  w/ gradient    │       │  no gradient   │
     └───────┬────────┘       └───────┬────────┘
             │                        │
             ▼                        ▼
      outputs [128,1000]       ema_output [128,1000]
             │                        │
             │                   softmax()
             │                        │
             ▼                        ▼
     ┌───────────────────────────────────────┐
     │  Loss = CE(outputs, targets)          │
     │       + CE(outputs, soft_ema) × mesa  │
     │                                       │
     │  targets: ground truth (w/ mixup)     │
     │  soft_ema: EMA teacher의 soft label   │
     └───────────────────────────────────────┘
             │
             ▼
         backward()
             │
             ▼
     ┌───────────────┐
     │  Update Model │──── EMA update ──→ EMA Teacher
     │  (AdamW step) │    (moving average)
     └───────────────┘
```

---

## 7. 전체 Forward Pass: Tensor Shape 추적표

아래는 `B=128`일 때 한 장의 이미지가 모델을 통과하며 겪는 모든 shape 변환이다.

```
[128, 3, 224, 224]  ← 입력 이미지

── Stem ──────────────────────────────────────────────────
[128, 32, 112, 112]  Conv2d(3→32, k3, s2, p1) + BN + ReLU
[128, 32, 112, 112]  Conv2d(32→32, k3, p1) + BN + ReLU
[128, 32, 112, 112]  Conv2d(32→32, k3, p1) + BN + ReLU
[128, 256, 56, 56]   Conv2d(32→256, k3, s2, p1) + BN + ReLU
[128, 64, 56, 56]    Conv2d(256→64, k1) + BN
[128, 3136, 64]      flatten + transpose

── Stage 1 (56×56, dim=64, 1 block) ─────────────────────
[128, 3136, 64]      TTTBlock (CPE → TTT → MLP)
[128, 784, 128]      PatchMerging (2× downsample)

── Stage 2 (28×28, dim=128, 3 blocks) ───────────────────
[128, 784, 128]      TTTBlock ×3
[128, 196, 320]      PatchMerging

── Stage 3 (14×14, dim=320, 9 blocks) ───────────────────
[128, 196, 320]      TTTBlock ×9
[128, 49, 512]       PatchMerging

── Stage 4 (7×7, dim=512, 4 blocks) ─────────────────────
[128, 49, 512]       TTTBlock ×4

── Classification Head ──────────────────────────────────
[128, 512, 49]       transpose
[128, 512, 49]       BatchNorm1d
[128, 512, 1]        AdaptiveAvgPool1d
[128, 512]           flatten
[128, 1000]          Linear → logits
```

---

## 8. RoPE (Rotary Position Embedding)

TTT Block 내에서 SwiGLU branch의 q1, k1에 적용된다.

```
입력: q1 [B, N, C] (예: [128, 196, 320])
       │
       ▼
  reshape → [B, H, W, C] (예: [128, 14, 14, 320])
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  RoPE(shape=(14, 14, 320), base=10000)               │
│                                                      │
│  k_max = 320 // (2 × 2) = 80                        │
│  (2 channel_dims: H, W)                              │
│                                                      │
│  θ_k = 1 / (10000^(k/80))  for k=0..79              │
│  angles = meshgrid(H,W) × θ_k → [14, 14, 160]      │
│  rotations = cos(angles) + i·sin(angles)             │
│                                                      │
│  x_complex = view_as_complex(x) → [..., 160]        │
│  pe_x = rotations * x_complex                        │
│  output = view_as_real(pe_x) → flatten → [B,H,W,320]│
└──────────────────────────────────────────────────────┘
       │
       ▼
  reshape → [B, N, heads, head_dim] → transpose → [B, heads, N, head_dim]
```

---

## 9. PatchMerging 상세

각 stage 경계에서 해상도를 2배 줄이고 채널을 증가시키는 역할을 한다.

```
예: Stage 2 → Stage 3 전환 (dim=128 → 320, ratio=4.0)

입력: [B, 784, 128]
       │
  reshape → [B, 128, 28, 28]
       │
       ▼
  Conv1x1(128 → 1280) + BN + ReLU    → [B, 1280, 28, 28]
  DWConv3x3(1280, s2, p1) + BN + ReLU → [B, 1280, 14, 14]
  Conv1x1(1280 → 320) + BN            → [B, 320, 14, 14]
       │
  flatten + permute → [B, 196, 320]
```

---

## 10. Weight Initialization & Optimizer 설정

### 10.1 Weight 초기화

```python
# Linear: truncated normal (std=0.02), bias=0
# LayerNorm: weight=1.0, bias=0
# TTT inner weights (w1, w2, w3): truncated normal (std=0.02)
```

### 10.2 AdamW Optimizer 파라미터 그룹

```
┌────────────────────────────────────────────────────┐
│  Group 1: has_decay                                │
│    - 2D+ weight tensors (Linear.weight, Conv.weight)│
│    - weight_decay = 0.05                           │
│                                                    │
│  Group 2: no_decay                                 │
│    - 1D tensors (bias, LayerNorm, BatchNorm)       │
│    - weight_decay = 0.0                            │
└────────────────────────────────────────────────────┘
```

### 10.3 Learning Rate Schedule

```
LR
 ↑
 │     ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │    ╱                                     ╲
 │   ╱  cosine decay                         ╲
 │  ╱                                         ╲
 │ ╱                                           ╲__
 │╱ warmup                                      min_lr
 └─────────────────────────────────────────────────→ epoch
   0    20                                   300
   ↑    ↑
 warmup  base_lr
 5e-7   (linear scaled)
```

---

## 11. 학습 과정 전체 타임라인

```
Epoch:  0 ──────── 20 ──────── 75 ──────────────── 300
        │           │           │                    │
        │  Warmup   │  Normal   │  MESA 활성화        │
        │  LR ramp  │  Training │  loss = CE + CE_ema│
        │  5e-7→LR  │  CE only  │                    │
        │           │           │                    │
        │           │           │  EMA Teacher의     │
        │           │           │  soft label을      │
        │           │           │  추가 supervision으로│
        │           │           │  활용              │
        │           │           │                    │
        ├───────────┴───────────┴────────────────────┤
        │  매 epoch: EMA model weights 업데이트        │
        │  ema = 0.9996 * ema + 0.0004 * model       │
        │  매 epoch: validation (model + ema)         │
        │  best checkpoint 저장                       │
        └────────────────────────────────────────────┘
```

---

## 12. 핵심 코드 파일 참조

| 파일 | 역할 |
|---|---|
| `vittt/main_ema.py` | 학습 진입점, training loop, MESA loss |
| `vittt/config.py` | 전체 config 기본값 정의 |
| `vittt/cfgs/h_vittt_t_mesa.yaml` | h_vittt_tiny + MESA=1.0 설정 |
| `vittt/models/build.py` | 모델 팩토리 |
| `vittt/models/h_vittt.py` | ViTTT 모델 (Stem, TTTBlock, PatchMerging, Head) |
| `vittt/models/ttt_block.py` | TTT 핵심 구현 (inner training + inner inference) |
| `vittt/data/build.py` | ImageNet DataLoader + Mixup/CutMix |
| `vittt/optimizer.py` | AdamW + weight decay 분리 |
| `vittt/lr_scheduler.py` | Cosine LR scheduler |
