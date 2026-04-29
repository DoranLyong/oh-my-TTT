# Test3R (TTT) Code Flow Analysis

`quick_demo.sh` 실행 시 전체 파이프라인 분석

---

## 0. 실행 진입점

```bash
# quick_demo.sh
python demo_ttt.py \
  --images demo/data/1.png demo/data/2.png ... demo/data/8.png
```

**기본 하이퍼파라미터:**
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `image_size` | 512 | 이미지 긴 변 크기 |
| `epoches` | 1 | TTT 학습 에폭 |
| `lr` | 1e-5 | TTT 학습률 |
| `accum_iter` | 2 | Gradient accumulation 스텝 |
| `prompt_size` | 32 | VPT prompt 토큰 수 |

---

## 1. 입력 데이터의 형태

### 1.1 원본 이미지 로딩 (`dust3r/utils/image.py:load_images`)

```
원본 이미지 (PNG, 임의 해상도)
    ↓  _resize_pil_image(): 긴 변을 512px로 리사이즈
    ↓  center crop: 16의 배수로 정렬 (halfw, halfh 계산)
    ↓  ImgNorm(): ToTensor + Normalize(mean=0.5, std=0.5) → [-1, 1] 범위
    ↓  [None] 으로 batch 차원 추가
```

### 1.2 개별 이미지 딕셔너리 구조

```python
img_dict = {
    'img':        Tensor [1, 3, H, W],    # 예: [1, 3, 288, 512] (정규화된 RGB)
    'true_shape': np.int32 [H, W],         # 예: [288, 512]
    'idx':        int,                      # 이미지 인덱스 (0~7)
    'instance':   str,                      # 문자열 인덱스
}
```

> **이미지 크기 예시:** 512x384 원본 → 긴 변 512 유지, crop 후 `[3, 288, 512]` 또는 `[3, 384, 512]`
> (정확한 H는 원본 비율에 따라 결정, 16의 배수로 정렬)

---

## 2. 전체 파이프라인 Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  8장의 이미지 로딩                                                 │
│  imgs: List[dict] (len=8)                                        │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 1: TTT (Test-Time Training)                               │
│  ─────────────────────────────────                               │
│  triplet 쌍 생성 → prompt_embeddings만 학습                       │
│  self-supervised loss로 모델 적응                                  │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 2: Inference (Pairwise)                                   │
│  ─────────────────────────────                                   │
│  모든 이미지 쌍에 대해 3D 포인트 예측                                │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 3: Global Alignment                                       │
│  ─────────────────────────                                       │
│  모든 pairwise 예측을 하나의 일관된 3D 장면으로 정합                   │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 4: Point Cloud 저장 (.ply)                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1 — TTT (Test-Time Training) 상세

### 3.1 Triplet 쌍 생성 (`image_pairs.py:make_pairs_tri`)

8장의 이미지에 대해 **complete graph**로 모든 (i, j, k) 조합 생성:

```
N = 8일 때: 8 × 8 × 8 = 512개의 triplet
각 triplet = (img_anchor, img_ref1, img_ref2)
```

이 triplet은 랜덤 셔플된다.

### 3.2 TTT 학습 루프 (`inference.py:inference_ttt`)

**핵심 아이디어:** 모델 파라미터 중 `prompt_embeddings`만 학습하고, 나머지는 고정

```python
# 학습 가능 파라미터
prompt_embeddings: [24, 1, 32, 1024]  # deep prompting (각 encoder 레이어별)
                    │   │   │    └── 임베딩 차원 (enc_embed_dim)
                    │   │   └────── prompt 토큰 수 (prompt_size)
                    │   └────────── batch 차원 (expand로 확장됨)
                    └────────────── encoder 레이어 수 (24개)

# 나머지 모델 파라미터: requires_grad = False (고정)
```

#### TTT 1 iteration 과정:

```
triplet[i] = (anchor, ref1, ref2)

pair1 = (anchor, ref1)  → model(pair1) → pred1_pair1 → pred1_pair1['pts3d']  [1, H/16*W/16, 3]
pair2 = (anchor, ref2)  → model(pair2) → pred1_pair2 → pred1_pair2['pts3d']  [1, H/16*W/16, 3]
                                                         (pixel-shuffle 후 [1,H,W,3])

loss = |pred1_pair1['pts3d'] - pred1_pair2['pts3d']|.mean()
```

**수학적 해석 — TTT 자기 지도 손실:**

$$\mathcal{L}_{TTT} = \frac{1}{N} \sum_{h,w} \left| \hat{P}^{(1)}_{anchor \to ref_1}(h,w) - \hat{P}^{(1)}_{anchor \to ref_2}(h,w) \right|$$

여기서 $\hat{P}^{(1)}$는 anchor 이미지의 카메라 좌표계에서의 예측된 3D 포인트.
**같은 anchor를 사용하므로 anchor 시점의 3D 좌표는 ref를 바꿔도 일관되어야** 한다는 자기지도 신호.

```
loss /= accum_iter (=2)
gradient accumulation: 2 step마다 optimizer.step()
```

---

## 4. Phase 2 — Model Forward Pass 상세

### 4.1 모델 아키텍처 개요

```
AsymmetricCroCo3DStereo (← CroCoNet)
├── patch_embed: PatchEmbedDust3R
│     └── Conv2d(3, 1024, kernel=16, stride=16)
├── enc_blocks: 24 × Block (ViT-Large Encoder)
│     └── Self-Attention + MLP, with RoPE2D
├── dec_blocks:  8 × DecoderBlock (Decoder 1 — view1)
│     └── Self-Attn + Cross-Attn + MLP
├── dec_blocks2: 8 × DecoderBlock (Decoder 2 — view2)
│     └── Self-Attn + Cross-Attn + MLP
├── head1: LinearPts3d (→ view1의 pts3d)
├── head2: LinearPts3d (→ view2의 pts3d, in view1 frame)
└── prompt_embeddings: [24, 1, 32, 1024] (TTT용)
```

### 4.2 입력 쌍 생성 (`image_pairs.py:make_pairs`)

```
8장 → complete graph → C(8,2) = 28 쌍
symmetrize=True → 28 × 2 = 56 쌍
각 쌍 = (img_i, img_j), 둘 다 dict
```

### 4.3 Forward Pass 텐서 흐름

가정: 이미지 크기 `H=288, W=512`, `patch_size=16`

#### Step 1: Patch Embedding (`model.py:_encode_image`)

```
입력:  image [2, 3, 288, 512]     (symmetrize 시 view1, view2를 concat하여 batch=2)

Conv2d(3→1024, k=16, s=16):
       [2, 3, 288, 512]
    →  [2, 1024, 18, 32]          # H/16=18, W/16=32

flatten + transpose:
    →  [2, 576, 1024]             # 576 = 18×32 패치 토큰, 각 1024차원

pos:   [2, 576, 2]                # 각 토큰의 (y, x) 좌표 (0~17, 0~31)
```

#### Step 2: Prompt 삽입 (TTT 모드, `use_prompt=True`)

```
prompt_embeddings[0]: [1, 32, 1024] → expand → [2, 32, 1024]

x = cat([prompt, patch_tokens], dim=1)
    [2, 32, 1024] + [2, 576, 1024] → [2, 608, 1024]

pos_final = cat([zeros(2,32,2), pos+1], dim=1)
    [2, 32, 2] + [2, 576, 2] → [2, 608, 2]
```

#### Step 3: Encoder (24 × Block)

```
각 Block: Self-Attention + MLP (with RoPE2D)

x: [2, 608, 1024]  (32 prompt + 576 patch tokens)

Self-Attention:
  Q, K, V ← Linear(1024 → 3072), reshape → [2, 16, 608, 64]
  Q, K에 RoPE2D 적용 (pos_final 기반 회전 위치 임베딩)
  Attn = softmax(Q·K^T / √64) · V     ← Scaled Dot-Product Attention
  출력: [2, 608, 1024]

MLP:
  Linear(1024 → 4096) → GELU → Linear(4096 → 1024)

Deep Prompting (레이어 i=1~22):
  x[:, :32, :] = prompt_embeddings[i]   # 매 레이어마다 prompt 토큰 교체

최종 enc_norm 후:
  x = x[:, 32:, :]                      # prompt 토큰 제거
  → [2, 576, 1024]
```

#### Step 4: Encoder 출력 분리

```
feat1, feat2 = x.chunk(2, dim=0)
  feat1: [1, 576, 1024]   (view1 encoder 특징)
  feat2: [1, 576, 1024]   (view2 encoder 특징)

pos1: [1, 576, 2]
pos2: [1, 576, 2]
```

#### Step 5: Decoder (`model.py:_decoder`)

```
Linear projection (1024 → 768):
  f1: [1, 576, 768]
  f2: [1, 576, 768]

8 × DecoderBlock (비대칭 — dec_blocks / dec_blocks2):

  dec_blocks (view1 decoder):
    Self-Attn(f1, pos1)           # f1 내부 self-attention
    Cross-Attn(f1, f2, pos1, pos2) # f2를 memory로 사용
    MLP

  dec_blocks2 (view2 decoder):
    Self-Attn(f2, pos2)           # f2 내부 self-attention
    Cross-Attn(f2, f1, pos2, pos1) # f1을 memory로 사용
    MLP

최종 dec_norm:
  dec1: list of [1, 576, 768] (각 레이어 출력 + encoder 출력)
  dec2: list of [1, 576, 768]
```

**Cross-Attention의 수학적 의미:**
$$\text{CrossAttn}(Q_1, K_2, V_2) = \text{softmax}\left(\frac{Q_1 \cdot K_2^T}{\sqrt{d_k}}\right) V_2$$

view1의 각 패치가 view2의 모든 패치와 attention하여, 두 뷰 사이의 대응관계(correspondence)를 학습.

#### Step 6: Prediction Head (`heads/linear_head.py:LinearPts3d`)

```
tokens = dec_out[-1]: [1, 576, 768]   (마지막 decoder 레이어 출력)

Linear(768 → 4×16²) = Linear(768 → 1024):
  feat: [1, 576, 1024]

Reshape:
  transpose(-1,-2) → [1, 1024, 576]
  view → [1, 1024, 18, 32]        # (B, 4*16², H/16, W/16)
                                    # 4 = 3(xyz) + 1(conf)

PixelShuffle(16):
  [1, 1024, 18, 32] → [1, 4, 288, 512]  # 원본 해상도 복원
                        │
                        ├── [:3] → pts3d 채널 (x, y, z)
                        └── [3:] → confidence 채널
```

#### Step 7: Postprocess (`heads/postprocess.py`)

```
feat: [1, 4, 288, 512]
  → permute(0,2,3,1) → [1, 288, 512, 4]

pts3d_raw = fmap[:,:,:,0:3]: [1, 288, 512, 3]

depth_mode = ('exp', -inf, inf):
  d = ||xyz||₂                    # 원점까지 거리
  direction = xyz / d             # 단위 방향 벡터
  pts3d = direction × (e^d - 1)   # exp 변환으로 깊이를 양수로 보장

conf_raw = fmap[:,:,:,3]: [1, 288, 512]

conf_mode = ('exp', 1, inf):
  conf = 1 + exp(conf_raw)        # confidence ∈ [1, ∞)
```

### 4.4 Model 출력 정리

```python
# forward() 반환값
res1 = {
    'pts3d': Tensor [B, H, W, 3],  # view1 좌표계에서 view1의 3D 포인트
    'conf':  Tensor [B, H, W],     # view1의 confidence
}

res2 = {
    'pts3d_in_other_view': Tensor [B, H, W, 3],  # view1 좌표계에서 view2의 3D 포인트
    'conf': Tensor [B, H, W],                     # view2의 confidence
}
```

**비대칭(Asymmetric) 구조의 핵심:**
- `res1['pts3d']`: view1의 각 픽셀이 view1 카메라 좌표계에서 어디에 있는지
- `res2['pts3d_in_other_view']`: view2의 각 픽셀이 **view1 카메라 좌표계**에서 어디에 있는지
- 두 포인트 클라우드가 같은 좌표계에 있으므로 직접 비교/결합 가능

---

## 5. Phase 3 — Global Alignment 상세

### 5.1 입력

56개 pairwise inference 결과를 하나로 collate:

```python
output = {
    'view1': {img, true_shape, idx, ...}   # collated
    'view2': {img, true_shape, idx, ...}
    'pred1': {pts3d, conf}                 # [56, H, W, 3], [56, H, W]
    'pred2': {pts3d_in_other_view, conf}   # [56, H, W, 3], [56, H, W]
}
```

### 5.2 PointCloudOptimizer

8장 (> 2장)이므로 `PointCloudOptimizer` 모드 사용.

**최적화 변수:**

| 변수 | Shape | 설명 |
|------|-------|------|
| `im_depthmaps` | `[8, H×W]` | 각 이미지의 픽셀별 log-depth |
| `im_poses` | `[8, 6]` | 카메라 포즈 (rotation + translation) |
| `im_focals` | `[8, 1]` | 카메라 초점거리 |
| `im_pp` | `[8, 2]` | 주점 (principal point) |

**최적화 목표:**

$$\min_{\theta} \sum_{(i,j) \in \text{edges}} w_{ij} \cdot \left\| \text{proj}(\text{depth}_i, \text{pose}_i, \text{focal}_i) - \hat{P}_{ij} \right\|$$

여기서:
- $\text{proj}(\cdot)$: depth + intrinsics + pose로 3D 포인트 생성
- $\hat{P}_{ij}$: 모델이 예측한 pairwise 3D 포인트
- $w_{ij}$: confidence 기반 가중치

```
init='mst':     Minimum Spanning Tree로 초기 포즈 추정
niter=300:      최적화 반복 횟수
schedule='linear': 학습률 선형 감소
lr=0.01:        최적화 학습률
```

### 5.3 출력

```python
scene.get_pts3d()   → List[Tensor [H, W, 3]] × 8    # 전역 좌표계 3D 포인트
scene.get_masks()   → List[Tensor [H, W]] × 8        # 유효 포인트 마스크
scene.get_im_poses() → Tensor [8, 4, 4]              # 전역 좌표계 카메라 포즈
scene.get_focals()   → Tensor [8]                     # 추정된 초점거리
scene.imgs           → List[ndarray [H, W, 3]] × 8   # RGB 이미지
```

---

## 6. Phase 4 — Point Cloud 저장

```python
# 각 이미지의 3D 포인트를 마스크 적용 후 결합
for (pts, img, mask) in zip(pts3d, imgs, masks):
    pts_masked  = pts[mask]      # [N_valid, 3]
    colors_masked = img[mask]    # [N_valid, 3]

# 전체 결합
pts_all:    [N_total, 3]   # 모든 유효 3D 좌표
colors_all: [N_total, 3]   # 대응하는 RGB 색상

# Open3D로 PLY 저장
```

---

## 7. 주요 텐서 Shape 요약 (H=288, W=512 기준)

```
단계                              텐서              Shape
─────────────────────────────── ──────────────── ────────────────────
입력 이미지                      img              [1, 3, 288, 512]
패치 임베딩 후                    x                [2, 576, 1024]
Prompt 삽입 후                   x                [2, 608, 1024]
Encoder 출력 (prompt 제거)       feat             [1, 576, 1024]
Decoder 입력                     f                [1, 576, 768]
Decoder 출력                     dec_out          [1, 576, 768]
Head proj 출력                   feat             [1, 576, 1024]
PixelShuffle 후                  feat             [1, 4, 288, 512]
최종 pts3d                       pts3d            [1, 288, 512, 3]
최종 confidence                  conf             [1, 288, 512]
TTT loss 입력                    pts3d            [1, 288, 512, 3]
Global align 입력 (collated)     pred_pts3d       [56, 288, 512, 3]
Global align 출력                pts3d_per_img    [288, 512, 3] × 8
최종 포인트 클라우드               pts_all          [N_total, 3]
```

---

## 8. 수학적 관점 종합 정리

### 8.1 Encoder: 이미지 → 패치 토큰 시퀀스

$$\mathbf{x} = \text{Conv2D}(\mathbf{I}) \in \mathbb{R}^{N \times D}$$

여기서 $N = \frac{H}{16} \times \frac{W}{16}$, $D = 1024$ (ViT-Large)

RoPE2D로 2D 위치 인코딩:

$$q'_i = R(\theta, y_i, x_i) \cdot q_i, \quad k'_j = R(\theta, y_j, x_j) \cdot k_j$$

→ $(q'_i)^T k'_j$에 상대 위치 $(y_i - y_j, x_i - x_j)$가 자연스럽게 인코딩됨

### 8.2 VPT (Visual Prompt Tuning) via Deep Prompting

각 encoder 레이어 $\ell$에서:

$$\mathbf{x}^{(\ell)} = \text{Block}^{(\ell)}\left([\mathbf{p}^{(\ell)}; \mathbf{x}^{(\ell-1)}_{patch}]\right)$$

여기서 $\mathbf{p}^{(\ell)} \in \mathbb{R}^{32 \times 1024}$는 학습 가능한 prompt 토큰.
TTT 시 이 $\mathbf{p}$만 역전파하여 테스트 시점에 입력 장면에 적응.

### 8.3 비대칭 Decoder: Stereo 매칭

$$f_1^{(\ell)} = \text{Self-Attn}(f_1^{(\ell-1)}) + \text{Cross-Attn}(f_1^{(\ell-1)}, f_2^{(\ell-1)})$$
$$f_2^{(\ell)} = \text{Self-Attn}(f_2^{(\ell-1)}) + \text{Cross-Attn}(f_2^{(\ell-1)}, f_1^{(\ell-1)})$$

두 개의 별도 디코더(`dec_blocks`, `dec_blocks2`)를 사용하여 비대칭 출력:
- head1: view1 좌표계의 view1 3D 포인트
- head2: view1 좌표계의 view2 3D 포인트

### 8.4 3D 포인트 복원 (exp depth mode)

네트워크 출력 $\mathbf{r} = (r_x, r_y, r_z)$에서:

$$d = \|\mathbf{r}\|_2, \quad \hat{\mathbf{n}} = \frac{\mathbf{r}}{d}$$
$$\mathbf{P}_{3D} = \hat{\mathbf{n}} \cdot (e^d - 1)$$

이 변환은:
1. 방향(ray direction)과 깊이(depth)를 분리
2. $e^d - 1$ 변환으로 깊이를 항상 양수로 보장하면서 가까운 물체의 정밀도 유지

### 8.5 TTT 자기지도 손실

Triplet $(I_a, I_1, I_2)$에 대해:

$$\mathcal{L} = \frac{1}{HW} \sum_{h,w} \left| f_\theta(I_a, I_1)^{(1)}_{h,w} - f_\theta(I_a, I_2)^{(1)}_{h,w} \right|$$

**직관:** 같은 anchor 이미지의 3D 구조는 어떤 reference 이미지를 사용하든 동일해야 함.
이 일관성(consistency) 조건이 라벨 없이 prompt를 학습시키는 자기지도 신호가 됨.

### 8.6 Global Alignment

모든 pairwise 예측을 전역 좌표계로 정합:

$$\min_{\{d_i, R_i, t_i, f_i\}} \sum_{(i,j)} \left\| \pi(d_i, K_i, [R_i|t_i]) - \hat{P}_{ij}^{(1)} \right\|^2 + \left\| \pi(d_j, K_j, [R_i|t_i]^{-1} \cdot [R_j|t_j]) - \hat{P}_{ij}^{(2)} \right\|^2$$

MST(Minimum Spanning Tree)로 초기 포즈를 추정한 후, 300 iterations의 gradient descent로 정제.

---

## 9. 핵심 파일 참조표

| 파일 | 역할 |
|------|------|
| `demo_ttt.py` | 메인 실행 스크립트 |
| `dust3r/model.py` | `AsymmetricCroCo3DStereo` 모델 정의 |
| `croco/models/croco.py` | `CroCoNet` 기반 클래스 |
| `croco/models/blocks.py` | Block, DecoderBlock, Attention, CrossAttention |
| `dust3r/patch_embed.py` | PatchEmbedDust3R, ManyAR_PatchEmbed |
| `dust3r/inference.py` | inference(), inference_ttt(), train_criterion() |
| `dust3r/image_pairs.py` | make_pairs(), make_pairs_tri() |
| `dust3r/utils/image.py` | load_images(), ImgNorm |
| `dust3r/heads/linear_head.py` | LinearPts3d 예측 헤드 |
| `dust3r/heads/postprocess.py` | exp depth 변환, confidence 추출 |
| `dust3r/cloud_opt/optimizer.py` | PointCloudOptimizer (전역 정합) |
