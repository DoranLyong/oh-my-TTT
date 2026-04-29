# Test3R — TTT (Test-Time Training) 메커니즘 심층 분석

> 논문 "Test3R: Learning to Reconstruct 3D at Test Time" 과 실제 코드 구현 기반 분석

---

## 목차

1. [TTT 핵심 개념](#1-ttt-핵심-개념)
2. [왜 TTT가 필요한가 — Cross-pair Inconsistency 문제](#2-왜-ttt가-필요한가--cross-pair-inconsistency-문제)
3. [TTT 자기지도 목적함수 — Triplet Consistency](#3-ttt-자기지도-목적함수--triplet-consistency)
4. [Visual Prompt Tuning (VPT) — 무엇을 학습하는가](#4-visual-prompt-tuning-vpt--무엇을-학습하는가)
5. [전체 TTT 연산 흐름 (텐서 Shape 추적)](#5-전체-ttt-연산-흐름-텐서-shape-추적)
6. [단계별 PyTorch 구현과 수학적 해석](#6-단계별-pytorch-구현과-수학적-해석)
7. [Gradient Flow 분석](#7-gradient-flow-분석)
8. [요약 및 직관](#8-요약-및-직관)

---

## 1. TTT 핵심 개념

### 1.1 일반적인 TTT 정의

Test-Time Training (TTT)은 **학습 시에 대규모 데이터로 사전학습된 모델을, 추론 시점에 테스트 입력에 맞게 미세조정**하는 기법이다.

```
┌─────────────────────┐     ┌────────────────────────────────┐
│  Training Phase     │     │  Test-Time Training Phase      │
│  (사전학습, 오프라 인)  │     │  (추론 직전, 온라인)              │
│                     │     │                                │
│  대규모 데이터셋       │     │  테스트 장면의 이미지만 사용         │
│  GT pointmap 사용    │     │  GT 없음 (자기지도 학습)           │
│  전체 파라미터 학습     │     │  prompt 토큰만 학습 (0.14%)      │
│                     │     │                                │
│  f_s: I_s → X_s     │ ──▶ │  f_t: I_t → X_t                │
│  (범용 모델)          │     │  (장면 특화 모델)                 │
└─────────────────────┘     └────────────────────────────────┘
```

### 1.2 Test3R의 TTT

Test3R에서는 다음과 같이 적용된다:

| 항목 | 내용 |
|------|------|
| **사전학습 모델** | DUSt3R (ViT-Large encoder + dual decoder) |
| **학습 대상** | `prompt_embeddings` (0.79M / 571.96M, 전체의 0.14%) |
| **자기지도 신호** | 같은 reference 이미지에 대한 **cross-pair consistency** |
| **학습 비용** | ~40초 (RTX 4090), 메모리 +3MB |

---

## 2. 왜 TTT가 필요한가 — Cross-pair Inconsistency 문제

### 2.1 DUSt3R의 Pairwise 한계

DUSt3R은 **한 번에 2장의 이미지**만 보고 3D 포인트를 예측한다:

```
f(I_ref, I_src1) → X_1^{ref,ref}   # I_ref의 3D 포인트 (src1 관점에서 추정)
f(I_ref, I_src2) → X_2^{ref,ref}   # I_ref의 3D 포인트 (src2 관점에서 추정)
```

**이상적으로는** $X_1^{ref,ref} = X_2^{ref,ref}$ 이어야 한다 — 같은 reference 이미지의 3D 좌표이므로.

**그러나 실제로는 불일치 발생:**

```
원인 1: 다른 source 이미지 → 다른 visual correspondence → 다른 3D 예측
원인 2: short baseline → 부족한 삼각측량 → 부정확한 기하 추정
원인 3: 학습 데이터와 다른 분포의 테스트 장면 → 일반화 실패
```

### 2.2 문제의 수학적 표현

이미지 $I^{ref}$의 pointmap을 $X^{ref}$라 하면, DUSt3R 모델 $f_\Theta$에 대해:

$$X_1^{ref} = f_\Theta(I^{ref}, I^{src_1}), \quad X_2^{ref} = f_\Theta(I^{ref}, I^{src_2})$$

$$\text{Inconsistency} = \|X_1^{ref} - X_2^{ref}\| \neq 0$$

이 불일치가 이후 Global Alignment 단계에서도 해소되지 않고, 전체 3D 재구성 품질을 저하시킨다.

---

## 3. TTT 자기지도 목적함수 — Triplet Consistency

### 3.1 Triplet 구성

$N$개의 이미지 집합 $\{I^i\}_{i=1}^{N}$에서 triplet $(I^{ref}, I^{src_1}, I^{src_2})$를 구성한다.

**코드 구현** (`image_pairs.py:make_pairs_tri`):

```python
# N=8 이미지에 대해 complete graph → 8³ = 512 triplets
for i in range(len(imgs)):          # ref
    for j in range(len(imgs)):      # src1
        for k in range(len(imgs)):  # src2
            pairs.append((imgs[i], imgs[j], imgs[k]))
random.shuffle(pairs)  # 랜덤 셔플
```

> **논문 Appendix A:** triplet 수가 165를 초과하면 165개를 랜덤 샘플링 (현재 코드에서는 전체 사용)

### 3.2 목적함수 정의

각 triplet $(I^{ref}, I^{src_1}, I^{src_2})$에 대해:

$$\mathcal{L} = \left\| X_1^{ref,ref} - X_2^{ref,ref} \right\|_1$$

구체적으로:

$$\mathcal{L} = \frac{1}{H \times W \times 3} \sum_{h=1}^{H} \sum_{w=1}^{W} \sum_{c=1}^{3} \left| X_1^{ref,ref}[h,w,c] - X_2^{ref,ref}[h,w,c] \right|$$

여기서:
- $X_1^{ref,ref} = f_\Theta(I^{ref}, I^{src_1})[\text{pts3d}]$ — (ref, src1) 쌍에서 ref의 3D 좌표
- $X_2^{ref,ref} = f_\Theta(I^{ref}, I^{src_2})[\text{pts3d}]$ — (ref, src2) 쌍에서 ref의 3D 좌표

### 3.3 핵심 직관

```
  "같은 reference 이미지의 3D 구조는 어떤 source 이미지와 쌍을 이루든
   동일하게 예측되어야 한다"
```

이 간단한 원리가 **라벨 없이** 사용할 수 있는 강력한 자기지도 신호가 된다:

1. **일관성(Consistency)**: 서로 다른 source 뷰에 대한 예측을 정렬
2. **정밀도(Precision)**: short-baseline으로 부정확한 예측을 다른 쌍의 결과로 보정
3. **적응(Adaptation)**: 특정 테스트 장면의 분포에 모델을 적응

**PyTorch 구현** (`inference.py:train_criterion`):

```python
def train_criterion(res1, res2):
    """
    res1['pts3d']: [1, H, W, 3]  ← f(I_ref, I_src1)의 ref pointmap
    res2['pts3d']: [1, H, W, 3]  ← f(I_ref, I_src2)의 ref pointmap
    """
    loss = torch.abs((res1['pts3d'] - res2['pts3d'])).mean()
    return loss
```

> **L1 Loss를 사용하는 이유:** L2에 비해 outlier에 덜 민감하여,
> 초기 불일치가 큰 상태에서도 안정적인 학습이 가능하다.

---

## 4. Visual Prompt Tuning (VPT) — 무엇을 학습하는가

### 4.1 왜 전체 파라미터가 아닌 Prompt만 학습하는가?

| 전략 | 장점 | 단점 |
|------|------|------|
| Full fine-tuning | 표현력 최대 | 노이즈성 자기지도 목적함수로 과적합/붕괴 위험 |
| Prompt tuning | 안정적, 사전학습 지식 보존 | 표현력 제한적 (충분히 효과적임) |

논문 Section 4.3:
> "These objectives are often noisy and unreliable, which makes the model prone to
> overfitting and may lead to training collapse."

**Prompt tuning은 backbone을 고정**하여 대규모 학습으로 얻은 3D 복원 지식을 보존하면서,
소수의 prompt 파라미터만 조정하여 특정 장면에 적응한다.

### 4.2 Prompt 임베딩의 구조

**코드 구현** (`model.py:AsymmetricCroCo3DStereo.__init__`):

```python
if use_prompt:
    self.prompt_size = prompt_size  # 32
    self.is_shallow = is_shallow    # False (deep prompting)

    if self.is_shallow:
        # Shallow: 첫 레이어에만 prompt 삽입
        prompt_embeddings = torch.zeros(1, 1, prompt_size, 1024)
        #                               │  │  │          └── enc_embed_dim (ViT-Large)
        #                               │  │  └────────── 32개 prompt 토큰
        #                               │  └───────────── batch (expand로 확장)
        #                               └──────────────── 1개 레이어분
    else:
        # Deep: 모든 encoder 레이어에 독립적 prompt
        prompt_embeddings = torch.zeros(24, 1, prompt_size, 1024)
        #                               │   │  │          └── enc_embed_dim
        #                               │   │  └────────── 32개 prompt 토큰
        #                               │   └───────────── batch
        #                               └──────────────── 24개 encoder 레이어

    self.prompt_embeddings = nn.Parameter(prompt_embeddings)
    self.patch_start_idx = prompt_size  # = 32
```

### 4.3 파라미터 수 비교

```
prompt_embeddings shape: [24, 1, 32, 1024]
총 prompt 파라미터: 24 × 1 × 32 × 1024 = 786,432 (≈ 0.79M)
전체 모델 파라미터: ~571.96M
비율: 0.79M / 571.96M ≈ 0.14%
```

### 4.4 Deep vs Shallow Prompting

```
Shallow Prompting (Test3R-S):
  Layer 0:  [P₀ ; E₀] → L₀ → [P₀' ; E₁]
  Layer 1:  [P₀'; E₁]  → L₁ → [P₀''; E₂]    ← P₀가 계속 전파
  ...
  Layer 23: [P₀^(23); E₂₃] → L₂₃ → 출력

  특징: 하나의 prompt이 모든 레이어를 통과 → 레이어별 특화 불가

Deep Prompting (Test3R):
  Layer 0:  [P₀ ; E₀] → L₀ → [_ ; E₁]
  Layer 1:  [P₁ ; E₁] → L₁ → [_ ; E₂]        ← 매 레이어마다 새 prompt 삽입
  ...
  Layer 23: [P₂₃; E₂₃] → L₂₃ → 출력

  특징: 각 레이어의 feature distribution에 맞춘 독립적 prompt
```

논문 Table 4에서 Deep prompting이 더 나은 성능을 보이지만,
파라미터가 많아 수렴이 어려운 경우도 있음:
- prompt_length=32: Deep(0.122/0.155) vs Shallow(0.125/0.150) → Deep 우세
- prompt_length=64: Deep(0.105/0.136) vs Shallow(0.120/0.158) → Deep 우세

---

## 5. 전체 TTT 연산 흐름 (텐서 Shape 추적)

**전제:** 이미지 8장, 크기 H=288, W=512, patch_size=16

### 5.1 Phase 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 0: Triplet 생성                                                │
│   8³ = 512 triplets, 각각 (I_ref, I_src1, I_src2)                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: 파라미터 고정 및 optimizer 설정                               │
│   prompt_embeddings만 requires_grad=True                            │
│   AdamW(lr=1e-5, betas=(0.9, 0.95))                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: TTT 학습 루프 (1 epoch × 512 triplets)                      │
│                                                                     │
│   for each triplet (I_ref, I_src1, I_src2):                        │
│     ┌─────────────────────────────────────────────────────┐        │
│     │ pair1 = (I_ref, I_src1) → model → pred1['pts3d']   │        │
│     │ pair2 = (I_ref, I_src2) → model → pred2['pts3d']   │        │
│     │ loss = |pred1['pts3d'] - pred2['pts3d']|.mean()     │        │
│     │ loss /= accum_iter                                   │        │
│     │ loss.backward() → prompt_embeddings.grad 업데이트     │        │
│     │ every accum_iter steps: optimizer.step()             │        │
│     └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 단일 Forward Pass 텐서 추적

하나의 pair `(I_ref, I_src)` 에 대한 상세 흐름:

```
────────── Patch Embedding ──────────

I_ref['img']:  [1, 3, 288, 512]    I_src['img']:  [1, 3, 288, 512]
         │                                   │
         └──── cat(dim=0) ──────────────────┘
                    │
                    ▼
            image: [2, 3, 288, 512]

Conv2d(3→1024, k=16, s=16):  [2, 1024, 18, 32]
flatten + transpose:          [2, 576, 1024]      ← 576 = 18×32 patches
pos:                          [2, 576, 2]          ← (y, x) grid 좌표

────────── Prompt 삽입 ──────────

prompt_embeddings[0]: [1, 32, 1024] → expand → [2, 32, 1024]

x = cat([prompt, patches], dim=1)
      [2, 32, 1024] ⊕ [2, 576, 1024] → [2, 608, 1024]
                                          │    └── dim
                                          └── 32 prompt + 576 patch tokens

pos_final = cat([zeros(2,32,2), pos+1], dim=1)
                                      → [2, 608, 2]

────────── Encoder (24 × Block) ──────────

Layer i (i=0..23):
  ┌──────────────────────────────────────────────┐
  │ x: [2, 608, 1024]                            │
  │                                              │
  │ ── Self-Attention with RoPE2D ──             │
  │ Q,K,V = Linear(1024→3072)                    │
  │ reshape → [2, 16, 608, 64]                   │
  │                  │    │    └── head_dim       │
  │                  │    └── N_tokens (prompt+patch)│
  │                  └── N_heads                  │
  │                                              │
  │ RoPE2D(Q, pos_final), RoPE2D(K, pos_final)  │
  │ Attn = softmax(Q·K^T/√64) · V               │
  │ → [2, 608, 1024]                             │
  │                                              │
  │ x = x + Attn(LayerNorm(x))                  │
  │                                              │
  │ ── FFN ──                                    │
  │ Linear(1024→4096) → GELU → Linear(4096→1024)│
  │ x = x + FFN(LayerNorm(x))                   │
  │                                              │
  │ ── Deep Prompt 교체 (i=1..22) ──              │
  │ x[:, :32, :] = prompt_embeddings[i]          │
  │ (prompt 토큰 위치에 새 값 주입)                  │
  └──────────────────────────────────────────────┘

최종:
  x = LayerNorm(x)            → [2, 608, 1024]
  x = x[:, 32:, :]            → [2, 576, 1024]  (prompt 토큰 제거)

────────── Encoder 출력 분리 ──────────

feat_ref, feat_src = x.chunk(2, dim=0)
  feat_ref: [1, 576, 1024]
  feat_src: [1, 576, 1024]
pos_ref:    [1, 576, 2]
pos_src:    [1, 576, 2]

────────── Decoder (8 × DecoderBlock) ──────────

projection: Linear(1024→768)
  f_ref: [1, 576, 768]
  f_src: [1, 576, 768]

Layer ℓ (ℓ=1..8):
  ┌────────────────────────────────────────────────────────────┐
  │ Decoder 1 (dec_blocks, ref 쪽):                             │
  │   f_ref = f_ref + SelfAttn(LN(f_ref), pos_ref)            │
  │   f_ref = f_ref + CrossAttn(LN(f_ref), LN(f_src), pos)    │
  │   f_ref = f_ref + FFN(LN(f_ref))                          │
  │                                                            │
  │ Decoder 2 (dec_blocks2, src 쪽):                            │
  │   f_src = f_src + SelfAttn(LN(f_src), pos_src)            │
  │   f_src = f_src + CrossAttn(LN(f_src), LN(f_ref), pos)    │
  │   f_src = f_src + FFN(LN(f_src))                          │
  └────────────────────────────────────────────────────────────┘

최종:
  dec_ref: list of [1, 576, 768]  (encoder출력 + 8 decoder 레이어 = 9개)
  dec_src: list of [1, 576, 768]

────────── Prediction Head (LinearPts3d) ──────────

tokens = dec_ref[-1]: [1, 576, 768]

Linear(768 → 1024):   [1, 576, 1024]    # 1024 = 4 × 16²
                                          # 4 = 3(xyz) + 1(conf)

reshape:  → [1, 1024, 18, 32]  (B, C, H/16, W/16)

PixelShuffle(16):
  [1, 1024, 18, 32] → [1, 4, 288, 512]
  # 1024 채널이 16×16 공간으로 재배치

permute(0,2,3,1):  → [1, 288, 512, 4]

────────── Postprocess ──────────

raw_xyz = out[:,:,:,0:3]:  [1, 288, 512, 3]

depth_mode = ('exp', -inf, inf):
  d = ‖raw_xyz‖₂:           [1, 288, 512, 1]  (원점까지 거리)
  n̂ = raw_xyz / d:          [1, 288, 512, 3]  (단위 방향 벡터)
  pts3d = n̂ · (eᵈ - 1):    [1, 288, 512, 3]  (최종 3D 좌표)

raw_conf = out[:,:,:,3]:   [1, 288, 512]
  conf = 1 + exp(raw_conf): [1, 288, 512]     (confidence ∈ [1, ∞))
```

---

## 6. 단계별 PyTorch 구현과 수학적 해석

### 6.1 TTT 루프 전체 구현

**파일:** `dust3r/inference.py:inference_ttt` (line 81-123)

```python
def inference_ttt(pairs, idx, model, device,
                  batch_size=1, epoches=2, lr=0.00001, accum_iter=4, verbose=True):

    # ─── Step 1: 학습 대상 설정 ───
    assert model.use_prompt == True
    model.train(True)

    # prompt_embeddings만 학습 가능하게 설정
    for name, param in model.named_parameters():
        if "prompt_embeddings" in name:
            param.requires_grad = True      # ← 이것만 학습
        else:
            param.requires_grad = False     # ← 나머지 전부 고정

    # ─── Step 2: Optimizer 설정 ───
    param_groups = misc.get_parameter_groups(model, weight_decay=0)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()  # Mixed precision 학습용
    optimizer.zero_grad()

    # ─── Step 3: TTT 학습 루프 ───
    for epoch in range(epoches):          # default: 1 epoch
        for i in range(len(pairs)):       # 512 triplets

            # triplet 분해: (anchor, src1, src2) → 2개의 pair
            pair1 = [(pairs[i][0], pairs[i][1])]  # (I_ref, I_src1)
            pair2 = [(pairs[i][0], pairs[i][2])]  # (I_ref, I_src2)

            # 각 pair에 대해 forward pass
            res1 = loss_of_one_batch(collate_with_cat(pair1), model, None, device)
            res2 = loss_of_one_batch(collate_with_cat(pair2), model, None, device)

            # TTT Loss 계산
            loss = train_criterion(res1["pred1"], res2["pred1"])
            #                          │                │
            #                          └── ref의 예측    └── ref의 예측
            #                          (I_ref,I_src1)    (I_ref,I_src2)

            # Gradient accumulation
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(i + 1) % accum_iter == 0)

            if (i + 1) % accum_iter == 0:
                optimizer.zero_grad()

    return model  # prompt가 최적화된 모델 반환
```

### 6.2 TTT Loss의 수학적 의미

#### L1 Consistency Loss

$$\mathcal{L}_{TTT} = \frac{1}{|\Omega|} \sum_{(h,w) \in \Omega} \left| \mathbf{P}_1(h,w) - \mathbf{P}_2(h,w) \right|$$

여기서:
- $\Omega = \{1,...,H\} \times \{1,...,W\}$ — 모든 픽셀 좌표
- $\mathbf{P}_1(h,w) = f_\Theta(I^{ref}, I^{src_1})^{ref}_{h,w} \in \mathbb{R}^3$ — pair1에서 ref의 3D 좌표
- $\mathbf{P}_2(h,w) = f_\Theta(I^{ref}, I^{src_2})^{ref}_{h,w} \in \mathbb{R}^3$ — pair2에서 ref의 3D 좌표
- $|\Omega| = H \times W \times 3$ (`.mean()` 으로 H, W, 3차원 모두 평균)

```python
# 텐서 연산 추적
res1["pred1"]['pts3d'].shape  # [1, 288, 512, 3]
res2["pred1"]['pts3d'].shape  # [1, 288, 512, 3]

diff = res1['pts3d'] - res2['pts3d']  # [1, 288, 512, 3]  원소별 차이
abs_diff = torch.abs(diff)             # [1, 288, 512, 3]  절댓값
loss = abs_diff.mean()                 # scalar            전체 평균
```

#### 대안: Confidence-weighted Loss

코드에는 구현되어 있으나 기본값으로는 사용하지 않는 변형:

```python
def train_criterion_with_conf(res1, res2):
    """confidence로 가중평균한 loss — 더 확신 있는 예측에 더 큰 가중치"""
    conf1 = min_max_normalize(res1['conf']).squeeze(0).unsqueeze(-1)  # [H,W,1]
    conf2 = min_max_normalize(res2['conf']).squeeze(0).unsqueeze(-1)  # [H,W,1]
    loss = torch.abs(conf1 * res1['pts3d'] - conf2 * res2['pts3d']).mean()
    return loss
```

$$\mathcal{L}_{conf} = \frac{1}{|\Omega|} \sum_{(h,w)} \left| \tilde{C}_1(h,w) \cdot \mathbf{P}_1(h,w) - \tilde{C}_2(h,w) \cdot \mathbf{P}_2(h,w) \right|$$

여기서 $\tilde{C} = \text{MinMaxNorm}(C)$는 정규화된 confidence.

### 6.3 Encoder에서 Prompt가 작동하는 방식

**파일:** `dust3r/model.py:_encode_image` (line 150-180)

```python
def _encode_image(self, image, true_shape):
    """
    image: [2, 3, 288, 512]  (ref와 src를 batch로 합침)
    """
    # ─── Patch Embedding ───
    x, pos = self.patch_embed(image, None, true_shape=true_shape)
    # x:   [2, 576, 1024]   ← 576 = (288/16) × (512/16) = 18 × 32
    # pos: [2, 576, 2]      ← 각 패치의 (y, x) 좌표

    # ─── Layer 0: 초기 Prompt 삽입 ───
    x = torch.cat([self.prompt_embeddings[0].expand(2, -1, -1), x], dim=1)
    # prompt_embeddings[0]: [1, 32, 1024] → expand → [2, 32, 1024]
    # x: [2, 32+576, 1024] = [2, 608, 1024]

    # prompt 토큰의 위치 인코딩은 0으로 설정 (특별한 위치 없음)
    pos_special = torch.zeros(2, 32, 2).to(x.device)       # [2, 32, 2]
    pos_final = torch.cat([pos_special, pos + 1], dim=1)    # [2, 608, 2]
    # patch 토큰의 pos에 +1을 해서 prompt(pos=0)와 구분

    # ─── 24개 Encoder Block 순회 ───
    for i, blk in enumerate(self.enc_blocks):
        x = blk(x, pos_final)
        # x: [2, 608, 1024]

        # Deep Prompting: 레이어 1~22에서 prompt 토큰 교체
        if i != 23 and self.use_prompt and not self.is_shallow and i != 0:
            x[:, :32, :] = self.prompt_embeddings[i].expand(2, -1, -1)
            # 레이어 i의 전용 prompt로 덮어쓰기
            # ← 이전 레이어의 attention 결과가 반영된 prompt 토큰을 버리고
            #    새로운 학습 가능 prompt로 교체

    # ─── 정규화 및 Prompt 제거 ───
    x = self.enc_norm(x)           # [2, 608, 1024]
    x = x[:, 32:, :]              # [2, 576, 1024]  prompt 토큰 제거

    return x, pos, None
```

### 6.4 Deep Prompting의 수학적 해석

Layer $i$ ($i = 0, 1, ..., 23$)에서의 연산:

$$\mathbf{x}^{(i)} = \text{Block}_i\left([\mathbf{P}^{(i-1)}; \mathbf{E}^{(i-1)}]\right) \in \mathbb{R}^{(N_p + N_t) \times D}$$

여기서:
- $\mathbf{P}^{(i)} \in \mathbb{R}^{N_p \times D}$ — layer $i$의 prompt 토큰 ($N_p=32$, $D=1024$)
- $\mathbf{E}^{(i)} \in \mathbb{R}^{N_t \times D}$ — layer $i$의 patch 토큰 ($N_t=576$)

**Deep prompting의 핵심 연산:**

```
Layer i의 Self-Attention에서:

  입력: [P^(i); E^(i-1)]    (608 토큰)

  Q = W_Q · [P^(i); E^(i-1)]   ← prompt 토큰도 query 생성
  K = W_K · [P^(i); E^(i-1)]   ← prompt 토큰도 key 생성
  V = W_V · [P^(i); E^(i-1)]   ← prompt 토큰도 value 생성

  Attention 행렬: [608 × 608]
  ┌──────────────┬───────────────┐
  │ P→P (32×32)  │ P→E (32×576) │  ← prompt가 patch에 attend
  ├──────────────┼───────────────┤
  │ E→P (576×32) │ E→E (576×576)│  ← patch가 prompt에 attend ★
  └──────────────┴───────────────┘

  ★ 핵심: patch 토큰이 prompt 토큰에 attend하면서
    prompt의 정보가 patch 표현에 주입됨
```

이후:
$$\mathbf{x}^{(i)}_{\text{prompt}} \leftarrow \mathbf{P}^{(i)} \quad \text{(새 prompt로 교체, } i = 1,...,22\text{)}$$
$$\mathbf{x}^{(i)}_{\text{patch}} = \mathbf{E}^{(i)} \quad \text{(attention 결과 유지)}$$

**각 레이어의 prompt가 독립적인 이유:**

논문 Section 5.3.2:
> "the feature distributions vary across each layer of the encoder,
> making layer-specific prompts more effective for fine-tuning."

Encoder의 얕은 레이어(low-level features: 엣지, 텍스처)와
깊은 레이어(high-level features: 의미, 구조)에서 필요한 보정이 다르기 때문.

### 6.5 Self-Attention with RoPE2D

**파일:** `croco/models/blocks.py:Attention.forward`

```python
def forward(self, x, xpos):
    B, N, C = x.shape  # [2, 608, 1024]

    # Q, K, V 생성
    qkv = self.qkv(x)  # Linear(1024 → 3072)
    # qkv: [2, 608, 3072]

    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
    # → [2, 16, 608, 3, 64] → transpose → [2, 3, 16, 608, 64]...
    # 실제: [2, 16, 3, 608, 64] 에서 q,k,v 분리

    q, k, v = [qkv[:, :, i] for i in range(3)]
    # q, k, v: 각각 [2, 16, 608, 64]

    # RoPE2D 적용 — 2D 회전 위치 인코딩
    if self.rope is not None:
        q = self.rope(q, xpos)  # xpos: [2, 608, 2] → (y, x) 좌표
        k = self.rope(k, xpos)

    # Scaled Dot-Product Attention
    attn = (q @ k.transpose(-2, -1)) * self.scale
    # [2, 16, 608, 64] @ [2, 16, 64, 608] → [2, 16, 608, 608]
    # scale = 1/√64 = 0.125

    attn = attn.softmax(dim=-1)  # 행 단위 softmax

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    # [2, 16, 608, 64] → transpose → [2, 608, 16, 64] → reshape → [2, 608, 1024]

    x = self.proj(x)  # Linear(1024 → 1024)
    return x
```

**RoPE2D의 수학:**

2D Rotary Position Embedding은 각 head dimension을 2D 좌표에 따라 회전:

$$q'_{h,d} = R(\theta_d, y_h, x_h) \cdot q_{h,d}$$

여기서:
- $(y_h, x_h)$ = 토큰 $h$의 2D 그리드 위치
- $\theta_d$ = 주파수 파라미터
- $R$ = 2D 회전 행렬

이렇게 하면 $\langle q'_i, k'_j \rangle$에 상대 위치 $(y_i - y_j, x_i - x_j)$가 자연스럽게 인코딩되어,
패치 간 공간 관계를 attention에서 활용할 수 있다.

**Prompt 토큰의 position:** `pos_special = zeros(2, 32, 2)` → RoPE에서 원점(0,0)으로 처리됨.
이는 prompt가 특정 공간 위치에 종속되지 않는 "global context" 역할을 하게 한다.

### 6.6 Cross-Attention in Decoder

**파일:** `croco/models/blocks.py:DecoderBlock.forward`

```python
def forward(self, x, y, xpos, ypos):
    """
    x:    [1, 576, 768]  ← 현재 branch (ref 또는 src)
    y:    [1, 576, 768]  ← 다른 branch (src 또는 ref)
    xpos: [1, 576, 2]
    ypos: [1, 576, 2]
    """
    # Self-Attention (자기 branch 내부)
    x = x + self.attn(self.norm1(x), xpos)

    # Cross-Attention (다른 branch를 memory로 사용)
    y_ = self.norm_y(y)
    x = x + self.cross_attn(self.norm2(x), y_, y_, xpos, ypos)
    #                        Q=x (query)   K=y  V=y (key, value from other view)

    # FFN
    x = x + self.mlp(self.norm3(x))

    return x, y  # y는 변경 없이 반환 (다른 decoder에서 사용)
```

**Cross-Attention의 수학:**

$$\text{CrossAttn}(Q_{ref}, K_{src}, V_{src}) = \text{softmax}\left(\frac{Q_{ref} \cdot K_{src}^T}{\sqrt{d_k}}\right) \cdot V_{src}$$

```
  Q (from ref): [1, 16, 576, 48]     ← ref의 각 패치가 query
  K (from src): [1, 16, 576, 48]     ← src의 각 패치가 key
  V (from src): [1, 16, 576, 48]     ← src의 각 패치가 value

  Attention: [1, 16, 576, 576]
  → ref의 패치 i가 src의 패치 j에 얼마나 attend하는지

  Output: [1, 576, 768]
  → ref의 각 패치에 src의 정보가 가중합으로 주입됨
```

**의미:** ref 이미지의 각 패치가 src 이미지에서 대응하는 영역을 찾아 정보를 가져옴.
이것이 두 뷰 사이의 **dense correspondence**를 암시적으로 학습하는 메커니즘이다.

### 6.7 Pointmap 복원: exp Depth Mode

**파일:** `dust3r/heads/postprocess.py:reg_dense_depth`

```python
def reg_dense_depth(xyz, mode):
    """
    xyz: [1, 288, 512, 3]  ← 네트워크 raw 출력
    mode: ('exp', -inf, inf)
    """
    mode, vmin, vmax = mode  # 'exp', -inf, inf

    # 원점까지의 거리 (L2 norm)
    d = xyz.norm(dim=-1, keepdim=True)    # [1, 288, 512, 1]

    # 단위 방향 벡터
    xyz_hat = xyz / d.clip(min=1e-8)      # [1, 288, 512, 3]

    # exp 변환으로 최종 3D 좌표 생성
    return xyz_hat * torch.expm1(d)       # [1, 288, 512, 3]
    #               └── expm1(d) = e^d - 1
```

**수학적 해석:**

네트워크 출력 $\mathbf{r} = (r_x, r_y, r_z) \in \mathbb{R}^3$를 극좌표 분해:

$$d = \|\mathbf{r}\|_2 \quad \text{(pseudo-depth, 네트워크가 예측한 "깊이" 신호)}$$
$$\hat{\mathbf{n}} = \frac{\mathbf{r}}{d} \quad \text{(ray direction, 단위 방향 벡터)}$$

최종 3D 포인트:

$$\mathbf{P}_{3D} = \hat{\mathbf{n}} \cdot (e^d - 1)$$

**왜 $e^d - 1$을 쓰는가?**

1. **양수 보장:** $e^d - 1 > 0$ for $d > 0$ → 깊이가 항상 양수
2. **가까운 물체 정밀도:** $d \approx 0$일 때 $e^d - 1 \approx d$ (선형 근사) → 가까운 물체에서 높은 해상도
3. **먼 물체 범위:** $d$가 클 때 $e^d - 1$이 급격히 증가 → 넓은 depth range 커버
4. **미분 가능:** 모든 지점에서 smooth하게 미분 가능 → gradient flow 원활

---

## 7. Gradient Flow 분석

### 7.1 역전파 경로

TTT에서 gradient가 흐르는 경로:

```
loss = |pts3d_1 - pts3d_2|.mean()
  │
  ▼
postprocess (exp depth) ← 미분 가능
  │
  ▼
LinearPts3d.proj ← head 파라미터 (고정, grad 안 흐름)
  │
  ▼
Decoder (8 layers) ← decoder 파라미터 (고정, grad 안 흐름)
  │                    그러나 activation은 통과
  ▼
enc_norm ← 고정
  │
  ▼
Encoder Block[23]
  │
  ▼
  ... (역순으로)
  │
  ▼
Encoder Block[i]
  │
  ├── MLP 출력에서 역전파 (MLP 파라미터 고정, activation 통과)
  │
  ├── Self-Attention 출력에서 역전파
  │     │
  │     └── Q·K^T에서 → prompt 토큰이 K, V에 기여한 부분으로 흐름
  │                     → patch 토큰이 prompt에 attend한 부분으로 흐름
  │
  └── prompt_embeddings[i]  ★ 여기서 grad 축적!
      x[:, :32, :] = prompt_embeddings[i]  ← in-place 할당이므로
      이 레이어의 prompt에 대한 gradient가 직접 계산됨
```

### 7.2 Gradient가 Prompt에 도달하는 메커니즘

Layer $i$에서 prompt $\mathbf{P}^{(i)}$가 patch 토큰 $\mathbf{E}$에 미치는 영향:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{P}^{(i)}} = \sum_{j \in \text{patches}} \frac{\partial \mathcal{L}}{\partial \mathbf{E}^{(i)}_j} \cdot \frac{\partial \mathbf{E}^{(i)}_j}{\partial \mathbf{P}^{(i)}}$$

Self-Attention에서 patch 토큰 $j$가 prompt 토큰에 attend하는 부분:

$$\mathbf{E}^{(i)}_j = \sum_{k=1}^{N_p} \alpha_{j,k} \cdot V(\mathbf{P}^{(i)}_k) + \sum_{k=1}^{N_t} \alpha_{j,N_p+k} \cdot V(\mathbf{E}^{(i-1)}_k)$$

여기서 $\alpha_{j,k} = \text{softmax}(Q_j \cdot K_k^T / \sqrt{d})$

따라서 prompt의 gradient는 **attention weight를 통해** patch 토큰에 전파된 영향에 비례한다.

### 7.3 왜 2번의 Forward Pass가 필요한가

```python
# pair1: (I_ref, I_src1) → forward → res1
res1 = loss_of_one_batch(pair1, model, None, device)  # grad 계산 O (train mode)

# pair2: (I_ref, I_src2) → forward → res2
res2 = loss_of_one_batch(pair2, model, None, device)  # grad 계산 O (train mode)

# 두 결과를 비교
loss = |res1['pred1']['pts3d'] - res2['pred1']['pts3d']|.mean()
```

**주의:** `@torch.no_grad()`가 적용되지 않음 (일반 `inference()`와 다른 점).
따라서 두 forward pass 모두에서 computation graph가 유지되어,
`loss.backward()`시 **두 경로 모두를 통해** prompt_embeddings로 gradient가 흐른다.

$$\frac{\partial \mathcal{L}}{\partial \mathbf{P}} = \frac{\partial \mathcal{L}}{\partial \mathbf{P}_1} \cdot \frac{\partial \mathbf{P}_1}{\partial \mathbf{P}} + \frac{\partial \mathcal{L}}{\partial \mathbf{P}_2} \cdot \frac{\partial \mathbf{P}_2}{\partial \mathbf{P}}$$

즉, 같은 prompt가 두 pair 모두에서 사용되므로, 두 경로의 gradient가 합산된다.

---

## 8. 요약 및 직관

### 8.1 한 줄 요약

> **Test3R는 "같은 reference 이미지의 3D 좌표는 source 이미지에 관계없이 일관되어야 한다"는
> 자명한 원리를 자기지도 신호로 활용하여, 소수의 prompt 파라미터만 테스트 시점에 최적화한다.**

### 8.2 핵심 수치

```
학습 대상:     prompt_embeddings [24, 1, 32, 1024] = 0.79M 파라미터 (0.14%)
학습 신호:     L_TTT = |pts3d_pair1 - pts3d_pair2|.mean()  (라벨 불필요)
학습 비용:     ~40초 / RTX 4090, +3MB 메모리
성능 향상:     DTU rel: 3.3 → 2.0 (-39%), τ: 69.9 → 84.1 (+20%)
```

### 8.3 직관적 이해

```
사전학습된 DUSt3R의 문제:

  f(🏠, 📷₁) → 🗺️₁  ← "집"의 3D 구조 예측 A
  f(🏠, 📷₂) → 🗺️₂  ← "집"의 3D 구조 예측 B

  🗺️₁ ≠ 🗺️₂  😱  ← 같은 집인데 다른 3D 예측!

Test3R의 해결:

  "🗺️₁과 🗺️₂가 같아지도록 prompt를 조정하자"

  prompt 최적화 후:
  f'(🏠, 📷₁) → 🗺️'₁ ≈ 🗺️'₂ ← f'(🏠, 📷₂)  ✅

  bonus: 이 과정에서 각 예측의 정확도도 함께 향상됨
```

### 8.4 주요 설계 결정 정리

| 설계 선택 | 선택지 | 이유 |
|----------|--------|------|
| Loss 함수 | L1 (MAE) vs L2 (MSE) | L1이 outlier에 강건 → 초기 불일치가 큰 상태에서 안정적 |
| 학습 대상 | Prompt only vs Full fine-tune | 노이즈성 자기지도에서 과적합/붕괴 방지 |
| Prompt 방식 | Deep vs Shallow | Deep이 레이어별 feature 분포에 맞춤 가능 → 더 효과적 |
| Prompt 위치 | Encoder only | Encoder가 feature 추출의 핵심, decoder는 pair-specific |
| Triplet 구성 | All combinations | 더 많은 제약조건 → 더 강한 일관성 보장 |
| Optimizer | AdamW(lr=1e-5) | 작은 lr로 사전학습 지식 보존하면서 미세조정 |
| Epoch | 1 | 과적합 방지 + 비용 최소화 |

---

## 부록: 코드 파일 참조

| 개념 | 파일 | 라인 |
|------|------|------|
| TTT 학습 루프 | `dust3r/inference.py` | 81-123 |
| Triplet 쌍 생성 | `dust3r/image_pairs.py` | 71-134 |
| Prompt 임베딩 정의 | `dust3r/model.py` | 84-97 |
| Prompt 삽입 (Encoder) | `dust3r/model.py` | 150-180 |
| TTT Loss (L1) | `dust3r/inference.py` | 138-140 |
| Confidence-weighted Loss | `dust3r/inference.py` | 130-136 |
| Self-Attention + RoPE | `croco/models/blocks.py` | 81-112 |
| Cross-Attention (Decoder) | `croco/models/blocks.py` | 132-191 |
| exp Depth 변환 | `dust3r/heads/postprocess.py` | 22-46 |
| PixelShuffle Head | `dust3r/heads/linear_head.py` | 30-41 |
| 데모 스크립트 | `demo_ttt.py` | 1-204 |
