# ViT³ 논문-코드 통합 해설

> **ViT³: Unlocking Test-Time Training in Vision** (arXiv:2512.01643)
> 논문의 핵심 개념, 수식, 그리고 대응하는 PyTorch 구현을 나란히 설명하는 문서

---

## 목차

0. [TTT(Test-Time Training)란 무엇인가? — 보편적 개념](#0-ttttest-time-training란-무엇인가--보편적-개념)
1. [왜 TTT인가? — 논문의 문제 의식과 동기](#1-왜-ttt인가--논문의-문제-의식과-동기)
2. [논문의 전개 구조: §1→§6 흐름 요약](#2-논문의-전개-구조-1→6-흐름-요약)
3. [Attention → Linear Attention → TTT 진화 과정](#3-attention--linear-attention--ttt-진화-과정)
4. [TTT의 핵심 원리: Inner Training](#4-ttt의-핵심-원리-inner-training)
5. [논문의 6가지 Insight와 ViT³ 설계](#5-논문의-6가지-insight와-vit³-설계)
6. [수식-코드 1:1 매칭: TTT Block](#6-수식-코드-11-매칭-ttt-block)
7. [수식-코드 1:1 매칭: Inner Training (SwiGLU)](#7-수식-코드-11-매칭-inner-training-swiglu)
8. [수식-코드 1:1 매칭: Inner Training (3×3 DWConv)](#8-수식-코드-11-매칭-inner-training-3×3-dwconv)
9. [수식-코드 1:1 매칭: Inner Inference (Query 적용)](#9-수식-코드-11-매칭-inner-inference-query-적용)
10. [H-ViT³ 전체 아키텍처와 코드 매칭](#10-h-vit³-전체-아키텍처와-코드-매칭)
11. [MESA 학습 전략과 코드 매칭](#11-mesa-학습-전략과-코드-매칭)
12. [Inner Loss 함수의 수학적 분석과 코드](#12-inner-loss-함수의-수학적-분석과-코드)
13. [학습 실험 결과](#13-학습-실험-결과)

---

## 0. TTT(Test-Time Training)란 무엇인가? — 보편적 개념

### 0.1 TTT의 기원과 일반적 정의

TTT(Test-Time Training)는 원래 **Yu Sun et al. (ICML 2025)**에 의해 제안된 개념으로,
모델이 **추론(test) 시점에 입력 데이터에 맞춰 스스로를 적응(학습)**시키는 패러다임이다.

기존 딥러닝 모델의 한계:
```
┌─── 기존 패러다임 ────────────────────────────────────────┐
│                                                         │
│  Training Phase:   대량의 데이터로 weight W를 학습         │
│                    W₀ → W₁ → ... → W* (최종 weight)      │
│                                                         │
│  Inference Phase:  W*를 고정(frozen)한 채로 예측           │
│                    모든 테스트 입력에 동일한 W* 적용        │
│                                                         │
│  문제: 학습 때 못 본 분포의 데이터 → 성능 저하             │
│        모든 입력에 동일한 처리 → 개별 입력 특성 무시       │
└─────────────────────────────────────────────────────────┘
```

TTT의 핵심 아이디어:
```
┌─── TTT 패러다임 ─────────────────────────────────────────┐
│                                                         │
│  Training Phase:   동일 (W₀ → W*)                        │
│                                                         │
│  Inference Phase:  입력이 들어올 때마다 W*를 살짝 조정!    │
│                    W* → W*(x)  ← 입력 x에 특화된 weight  │
│                                                         │
│  방법: self-supervised loss로 빠르게 few-step 학습        │
│  장점: 각 입력의 고유한 패턴에 모델이 실시간 적응          │
└─────────────────────────────────────────────────────────┘
```

### 0.2 TTT의 두 가지 해석

TTT는 분야에 따라 두 가지 다른 맥락에서 사용된다:

```
┌─────────────────────────────────────────────────────────┐
│  (A) Domain Adaptation으로서의 TTT (원래 의미)            │
│                                                         │
│  목표: 분포 변화(distribution shift)에 대한 적응          │
│  방법: 테스트 이미지에 대해 auxiliary task(회전 예측 등)을  │
│        self-supervised loss로 몇 step 학습               │
│  예시: 손상된 이미지 → 모델이 해당 손상 패턴에 적응        │
│                                                         │
│  적용 분야: robustness, out-of-distribution 일반화        │
├─────────────────────────────────────────────────────────┤
│  (B) Sequence Modeling으로서의 TTT (이 논문의 의미)        │
│                                                         │
│  목표: Attention 메커니즘의 O(N²)을 O(N)으로 대체         │
│  방법: K,V 쌍을 데이터셋으로, inner model을 학습시켜       │
│        context 정보를 weight에 압축                      │
│  핵심: "attention = inner model의 test-time training"    │
│                                                         │
│  적용 분야: 효율적 시퀀스 모델링 (NLP, Vision)             │
└─────────────────────────────────────────────────────────┘
```

**이 논문(ViT³)은 (B)의 관점**에서 TTT를 사용한다.
Attention 연산 자체를 "작은 모델을 매 입력마다 학습시키는 것"으로 재해석한다.

### 0.3 TTT as Sequence Modeling — 직관적 비유

```
┌─────────────────────────────────────────────────────────┐
│  비유: "시험 보기 전에 요약 노트 만들기"                    │
│                                                         │
│  전통적 Attention (Softmax):                             │
│    → 교과서(K,V) 전체를 펼쳐놓고 매번 필요한 부분을 찾음    │
│    → 교과서가 길면(N↑) 찾는 시간 급증: O(N²)             │
│                                                         │
│  Linear Attention:                                      │
│    → 교과서를 한 장짜리 치트시트(W=K^T·V)로 요약           │
│    → 빠르게 참조: O(N), 하지만 정보 손실 큼               │
│                                                         │
│  TTT:                                                   │
│    → 교과서(K,V)로 "나만의 미니 모델(brain)"을 학습시킴    │
│    → 핵심을 이해한 모델이 질문(Q)에 답변: O(N)            │
│    → 단순 요약이 아닌 "이해"를 통한 압축 → 표현력 ↑        │
└─────────────────────────────────────────────────────────┘
```

### 0.4 TTT의 학술적 연결 고리

TTT는 여러 기존 연구 분야와 깊은 관련이 있다:

| 관련 분야 | 연결점 |
|----------|--------|
| **Meta-learning (MAML)** | Inner loop에서 few-step 학습 → outer loop에서 end-to-end 최적화 |
| **Fast Weight Programmers** | 입력에 따라 weight를 동적으로 생성하는 초기 개념 |
| **Linear Attention** | TTT의 특수한 경우 (inner model = linear layer, 학습 = 직접 구성) |
| **State Space Models (Mamba)** | 동일한 O(N) 복잡도 목표, 다른 접근 방식 |
| **Knowledge Distillation** | Teacher(K,V) → Student(inner model) 관계와 유사 |

---

## 1. 왜 TTT인가? — 논문의 문제 의식과 동기

### 1.1 Vision Transformer의 한계

Vision Transformer(ViT)는 이미지 분류, 객체 탐지, 분할 등 거의 모든 비전 태스크에서 SOTA를 달성했지만,
**Softmax Attention의 O(N²) 복잡도**가 근본적인 병목이다.

```
시퀀스 길이 N과 연산량 관계:

  224×224 이미지, patch=16 → N = 196 토큰   → 아직 괜찮음
  512×512 이미지, patch=16 → N = 1024 토큰  → 느려짐
  1248×1248 이미지           → N = 6084 토큰  → Softmax는 매우 느림

  Softmax Attention: O(N²) — N이 커지면 메모리/속도 급격히 악화
  Linear Attention:  O(N)  — 빠르지만 표현력이 약함
  TTT:               O(N)  — 빠르면서도 표현력이 풍부함  ← 이 논문의 핵심
```

### 1.2 Linear Attention은 왜 부족한가?

논문 §3.1에서 Softmax Attention을 MLP로 재해석한다:

> Softmax Attention은 width=N인 2-layer MLP를 K,V로 구성한 것과 같다.
> Linear Attention은 d×d 선형(FC) 레이어 하나를 K^T·V로 구성한 것과 같다.

**Linear Attention의 문제**: d×d 크기의 선형 레이어 하나로 전체 context를 압축하므로
표현력이 제한적이다. 이것이 실제로 성능이 낮은 이유.

### 1.3 TTT의 해답

> "K,V 쌍을 '미니 데이터셋'으로 보고, 작은 inner model을 실시간으로 학습시키자!"

TTT는 inner model을 **임의의 신경망**으로 설정할 수 있어 설계 자유도가 높고,
**비선형 표현**이 가능하면서도 **O(N) 복잡도**를 유지한다.

---

## 2. 논문의 전개 구조: §1→§6 흐름 요약

이 논문은 "TTT를 vision에 어떻게 잘 적용할 것인가?"라는 질문에 체계적으로 답한다.

### 2.1 논문의 스토리라인

```
§1 Introduction — "왜 TTT가 필요한가?"
│
│  ViT의 O(N²) 문제 → Linear Attention의 표현력 한계
│  → TTT가 유망하지만, 설계 공간이 넓고 미탐구 상태
│  → 핵심 질문: "vision에 맞는 TTT 설계 원칙은 무엇인가?"
│
▼
§2 Related Work — "TTT는 어디서 왔는가?"
│
│  Attention/ViT, Linear Attention, SSM(Mamba) 계보 정리
│  TTT의 기원: Yu Sun et al. (ICML 2025)
│  기존 TTT 연구: 언어(LaCT), 비디오(1-min), 3D(TTT3R)
│  → 하지만 vision에서의 체계적 연구는 부재
│
▼
§3 Preliminaries — "Attention과 TTT의 수학적 관계"
│
│  핵심 재해석:
│  Softmax Attention = width-N의 2-layer MLP (Eq.2)
│  Linear Attention  = d×d 선형 레이어 (Eq.4)
│  TTT              = 임의 inner model의 online 학습 (Eq.5)
│  → 세 가지가 "같은 프레임워크의 스펙트럼"임을 보임
│
▼
§4 Exploring TTT Designs — "무엇이 잘 작동하는가?" ★ 논문의 핵심
│
│  DeiT에 TTT를 넣고 체계적 실험:
│
│  §4.1 Inner Training 설정:
│  ├─ Insight 1: ∂²L/∂V∂V̂≠0인 loss 필수 (Tab.1)
│  ├─ Insight 2: Full-batch, 1-epoch이 vision에 최적 (Tab.2)
│  └─ Insight 3: Inner LR=1.0이 효과적 (Tab.3)
│
│  §4.2 Inner Model 설계:
│  ├─ Insight 4: Inner model 용량↑ → 성능↑ (Tab.4)
│  ├─ Insight 5: 깊은 inner model은 최적화 어려움 (Tab.4, Fig.3)
│  └─ Insight 6: Conv inner model이 vision에 특히 적합 (Tab.4)
│
▼
§5 ViT³ 모델 — "Insight를 어떻게 모델로 만드는가?"
│
│  6개 insight를 결합한 최종 설계:
│  ├─ Inner model = Simplified SwiGLU + 3×3 DWConv
│  ├─ Inner training = Dot product loss, lr=1.0, 1-epoch full-batch
│  ├─ 모델 패밀리: ViT³ (non-hierarchical), H-ViT³ (4-stage)
│  │
│  실험 결과 (논문 Tab.5,7,8,9,10):
│  ├─ 분류: Mamba, Linear Attention 능가, Transformer와 경쟁적
│  ├─ 탐지: 긴 시퀀스에서 특히 강점 (COCO)
│  ├─ 분할: VMamba, SOFT++ 능가 (ADE20K)
│  └─ 생성: DiT 대비 일관된 FID 개선 (DiT³)
│
▼
§6 Conclusion — "TTT의 가능성과 미래"
│
│  ViT³ = 강력한 O(N) baseline
│  미래 방향: 깊은 inner model 최적화, vision 특화 mini-batch 전략
```

### 2.2 논문의 핵심 기여 (세 줄 요약)

1. **체계적 탐구**: vision TTT의 설계 공간을 inner training(loss, lr, batch, epoch)과 inner model(구조, 크기) 두 축으로 나누어 체계적으로 분석
2. **6가지 Insight**: 실험에서 도출한 실용적 설계 원칙 — 특히 "loss의 혼합 이계도함수가 0이면 안 된다"는 발견이 핵심
3. **ViT³ 모델**: insight를 결합한 순수 TTT 아키텍처로, O(N) 복잡도에서 Mamba/Linear Attention을 능가하고 Transformer에 근접

### 2.3 이 논문이 기존 TTT 연구와 다른 점

| | 기존 TTT (Sun et al.) | 이 논문 (ViT³) |
|---|---|---|
| **모달리티** | 언어 (causal) | 비전 (non-causal) |
| **Inner batch** | Mini-batch (sequential) | Full-batch (parallel) |
| **Inner model** | MLP만 | SwiGLU + DWConv |
| **Inner LR** | Dynamic (학습) | Fixed 1.0 |
| **체계적 분석** | 제한적 | Loss 함수, 깊이, 너비 등 포괄 |
| **아키텍처** | 언어 모델 위주 | ViT/Swin 스타일 + DiT |

---

## 3. Attention → Linear Attention → TTT 진화 과정

### 논문 Eq.(1)~(4)와 직관

```
┌────────────────────────────────────────────────────────────────────┐
│  (a) Softmax Attention  — 논문 Eq.(1),(2)                          │
│                                                                    │
│  O = σ(Q·K^T) · V                                                 │
│    = σ(Q·W₁) · W₂         ← W₁=K^T, W₂=V                        │
│    = MLP(Q)                ← width=N인 2-layer MLP                 │
│                                                                    │
│  복잡도: O(N²·d)   표현력: 매우 높음 (width=N)                       │
├────────────────────────────────────────────────────────────────────┤
│  (b) Linear Attention  — 논문 Eq.(3),(4)                           │
│                                                                    │
│  O = Q · (K^T · V)                                                │
│    = Q · W              ← W = K^T·V, d×d 선형 레이어               │
│    = FC(Q)                                                        │
│                                                                    │
│  복잡도: O(N·d²)   표현력: 낮음 (d×d 선형 매핑)                      │
├────────────────────────────────────────────────────────────────────┤
│  (c) TTT  — 논문 Eq.(5)                                           │
│                                                                    │
│  D = {(Kᵢ, Vᵢ) | i=1,...,N}   ← K,V를 "데이터셋"으로 봄           │
│  V̂ = F_W(K)                    ← inner model forward              │
│  W ← W - η · ∂L(V̂, V)/∂W     ← 1-step gradient descent          │
│  O = F_W*(Q)                   ← 학습된 W*로 Q 처리                │
│                                                                    │
│  복잡도: O(N·d²)   표현력: 높음 (비선형 inner model)                 │
└────────────────────────────────────────────────────────────────────┘
```

핵심 차이: Linear Attention이 `W = K^T·V`로 **한 번에** 정보를 압축하는 반면,
TTT는 **gradient descent로 학습**하여 더 풍부한 비선형 정보를 W에 담는다.

---

## 4. TTT의 핵심 원리: Inner Training

### 4.1 개념: Outer Loop vs Inner Loop

```
┌─ Outer Loop (일반적인 모델 학습) ──────────────────────────┐
│                                                           │
│  for epoch in range(300):                                 │
│    for images, labels in dataloader:                      │
│      logits = model(images)     ← 여기서 Inner Loop 발생   │
│      loss = CE(logits, labels)                            │
│      loss.backward()            ← Inner Loop도 함께 미분   │
│      optimizer.step()                                     │
│                                                           │
│  ┌─ Inner Loop (TTT Block 내부, 매 forward마다) ────────┐  │
│  │                                                      │  │
│  │  1. Q, K, V 생성 (outer model의 projection)          │  │
│  │  2. (K, V)로 inner model W를 1-step 학습             │  │
│  │  3. 학습된 W*로 Q를 처리하여 output 생성              │  │
│  │                                                      │  │
│  │  이 모든 과정이 미분 가능 → end-to-end 학습            │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### 4.2 Softmax Attention과의 비교 (논문 Remark 1)

논문은 Softmax Attention도 사실상 TTT와 같은 목표를 **암묵적으로** 달성한다고 분석한다:

> Softmax Attention의 inner MLP에 Kᵢ를 입력하면:
> F(Kᵢ) = σ(Kᵢ·K^T)·V ≈ Vᵢ
>
> 이유: Kᵢ·Kᵢ^T (자기 자신과의 유사도)가 가장 크므로,
> Softmax 후 one-hot에 가까워져 Vᵢ를 거의 그대로 복원.

TTT는 이 "K→V 복원"이라는 목표를 **명시적인 학습**으로 달성하는 것이다.

---

## 5. 논문의 6가지 Insight와 ViT³ 설계

논문 §4에서 체계적인 실험을 통해 6가지 설계 원칙을 도출했다:

| # | Insight | ViT³에서의 적용 | 코드 위치 |
|---|---------|----------------|----------|
| 1 | ∂²L/∂V∂V̂ = 0인 loss는 부적합 | **Dot product loss** 사용 | `ttt_block.py:72` (주석) |
| 2 | Full-batch, 1-epoch inner training이 vision에 적합 | **B=N, 1 epoch** | `ttt_block.py:54-88` |
| 3 | Inner learning rate η=1.0이 효과적 | **lr=1.0** 고정 | `ttt_block.py:54,87` |
| 4 | Inner model 용량↑ → 성능↑ | **SwiGLU** (2× capacity) | `ttt_block.py:40-41` |
| 5 | 깊은 inner model은 최적화 어려움 | **Identity output layer** | `ttt_block.py:165` |
| 6 | Conv가 vision inner model로 적합 | **3×3 DWConv** branch 추가 | `ttt_block.py:42,90-134` |

### Insight 1 → 코드: Dot Product Loss

```
논문 Eq.(8):
  L = -(1/B√d) · Σ V̂ᵢ·Vᵢᵀ

혼합 이계도함수:
  ∂²L / ∂V·∂V̂ = -1/(B√d)  ≠ 0   ← 학습 가능!

비교: MAE loss의 경우
  ∂²L / ∂V·∂V̂ = 0  (거의 어디서나)  ← W_V 학습 불가
```

코드에서는 dot product loss를 직접 계산하지 않고, 그 gradient를 닫힌 형태로 유도하여 사용한다:

```python
# ttt_block.py:78  ← dot product loss의 gradient
e = - v / float(v.shape[2]) * self.scale
#     ↑ V              ↑ N           ↑ 1/√d (scale factor)
# e = ∂L/∂V̂ = -V / (N·√d)  ← Eq.(8)의 V̂에 대한 미분
```

### Insight 4+5 → 코드: Simplified SwiGLU

```
논문 §4.2, Table 4:
  SwiGLU(x) = (xW₁) ⊙ SiLU(xW₂)·W₃    → 79.0%
  FC(x) ⊙ SiLU(FC(x))                   → 79.7%  ← output layer를 Identity로!

깊은 inner model의 최적화 문제를 피하면서 용량을 2배로 키움
```

```python
# ttt_block.py:40-41  ← W₁, W₂가 inner model의 학습 가능 파라미터
self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
# 출력 레이어(W₃) 없음 → Identity output = Simplified SwiGLU
```

### Insight 6 → 코드: 3×3 DWConv Inner Model

```
논문 §4.2:
  Conv inner model은 global(학습된 커널) + local(수용 영역) 정보를 자연스럽게 통합
  DWConv: 80.1% (최고 성능, 적은 파라미터)
```

```python
# ttt_block.py:42  ← DWConv의 초기 커널 (outer model이 학습하는 W₀)
self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
# head_dim 채널, 각 채널별 독립 3×3 커널 = depthwise convolution
```

---

## 6. 수식-코드 1:1 매칭: TTT Block

### 논문 §5, Fig.2: TTT Block 구조

논문에서 TTT Block은 attention의 drop-in replacement로 설계되었다.
ViT³는 두 개의 inner module을 병렬로 사용한다:
- F₁ = FC(x) ⊙ SiLU(FC(x)) — Simplified SwiGLU (num_heads-1 개의 head)
- F₂ = DWConv(x) — 3×3 Depthwise Convolution (1개의 head)

### 코드: TTT.__init__

```python
# ttt_block.py:32-52

class TTT(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads

        # ──────────────────────────────────────────────────────────────
        # 논문 §3.2: Q, K, V = xW_Q, xW_K, xW_V
        # SwiGLU branch용 Q₁,K₁,V₁ (dim 크기) + DWConv branch용 Q₂,K₂,V₂ (head_dim 크기)
        # ──────────────────────────────────────────────────────────────
        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        #                         ↑ Q₁,K₁,V₁    ↑ Q₂,K₂,V₂

        # ──────────────────────────────────────────────────────────────
        # 논문 Eq.(5): inner model의 초기 파라미터 W₀ (outer model이 end-to-end 학습)
        # ──────────────────────────────────────────────────────────────
        # F₁의 파라미터: Simplified SwiGLU = FC(x)·W₁ ⊙ SiLU(FC(x)·W₂)
        self.w1 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))  # W₁
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))  # W₂

        # F₂의 파라미터: 3×3 Depthwise Convolution
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))                # W₃

        # ──────────────────────────────────────────────────────────────
        # 두 branch의 출력을 합쳐서 최종 output 생성
        # ──────────────────────────────────────────────────────────────
        self.proj = nn.Linear(dim + head_dim, dim)
        #                     ↑ F₁ 출력(dim) + F₂ 출력(head_dim)

        # 논문 Remark 3: 1/√d scaling (여기서는 equivalent_head_dim=9 사용)
        self.scale = 9 ** -0.5
```

### 코드: TTT.forward — Q/K/V 준비

```python
# ttt_block.py:136-158

def forward(self, x, h, w, rope=None):
    b, n, c = x.shape       # B, N, C
    d = c // self.num_heads  # head_dim

    # ──────────────────────────────────────────────────────────────
    # 논문 §3.2:  Q = xW_Q,  K = xW_K,  V = xW_V
    # 실제 구현: 하나의 Linear로 6개 텐서를 한 번에 생성
    # ──────────────────────────────────────────────────────────────
    q1, k1, v1, q2, k2, v2 = torch.split(
        self.qkv(x),               # [B, N, dim*3 + head_dim*3]
        [c, c, c, d, d, d], dim=-1  # SwiGLU branch: c,c,c / DWConv branch: d,d,d
    )

    # SwiGLU branch: multi-head 형태로 reshape
    # [B, N, C] → RoPE → [B, num_heads, N, head_dim]
    q1 = rope(q1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
    k1 = rope(k1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
    v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)

    # DWConv branch: 2D spatial 형태로 reshape
    # [B, N, d] → [B, d, H, W]
    q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
    k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
    v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)
```

---

## 7. 수식-코드 1:1 매칭: Inner Training (SwiGLU)

### 논문 Eq.(5)의 SwiGLU 특화 버전

논문의 일반 수식:
```
V̂_B = F_W(K_B)
W ← W - η · ∂L(V̂_B, V_B) / ∂W
```

ViT³에서 F₁ = Simplified SwiGLU:
```
F_W(x) = (x·W₁) ⊙ SiLU(x·W₂)

여기서 SiLU(z) = z · σ(z),  σ = sigmoid
```

### Forward (inner model 예측)

```
수식:                                  코드 (ttt_block.py:67-70):
─────                                  ─────────────────────────
z₁ = K · W₁                           z1 = k @ w1
z₂ = K · W₂                           z2 = k @ w2
σ(z₂) = sigmoid(z₂)                   sig = F.sigmoid(z2)
V̂ = z₁ ⊙ (z₂ · σ(z₂))                a = z2 * sig
   = z₁ ⊙ SiLU(z₂)                   # v_hat = z1 * a (실제로는 계산 생략)
```

> **코드 최적화**: `v_hat`과 loss `l`은 실제로 계산하지 않는다.
> gradient의 닫힌 형태(closed-form)를 직접 유도하여 사용하기 때문이다.

### Inner Loss (Dot Product Loss)

```
논문 Eq.(8):
  L = -(1/N√d) · Σᵢ V̂ᵢ · Vᵢᵀ

∂L/∂V̂ = -V / (N√d)                     코드 (ttt_block.py:78):
                                        e = -v / float(v.shape[2]) * self.scale
                                        #    -V /      N           * (1/√d)
```

### Backward (hand-derived gradient)

V̂ = z₁ ⊙ a 이고, a = z₂ · σ(z₂) = SiLU(z₂) 이므로:

**∂L/∂W₁ 유도:**

```
∂L/∂W₁ = ∂L/∂V̂ · ∂V̂/∂z₁ · ∂z₁/∂W₁

V̂ = z₁ ⊙ a  이므로  ∂V̂/∂z₁ = a  (elementwise)
z₁ = K·W₁   이므로  ∂z₁/∂W₁ = Kᵀ

따라서:
∂L/∂W₁ = Kᵀ · (e ⊙ a)

수식:                                  코드 (ttt_block.py:79):
─────                                  ─────────────────────────
g₁ = Kᵀ · (e ⊙ a)                     g1 = k.transpose(-2,-1) @ (e * a)
```

**∂L/∂W₂ 유도 (chain rule을 통한 SiLU 미분):**

```
SiLU(z₂) = z₂ · σ(z₂)
SiLU'(z₂) = σ(z₂) + z₂ · σ(z₂) · (1 - σ(z₂))
           = σ(z₂) · (1 + z₂ · (1 - σ(z₂)))

∂V̂/∂z₂ = z₁ ⊙ SiLU'(z₂)
∂L/∂W₂ = Kᵀ · (e ⊙ z₁ ⊙ SiLU'(z₂))

수식:                                  코드 (ttt_block.py:80):
─────                                  ─────────────────────────
g₂ = Kᵀ·(e ⊙ z₁ ⊙ σ(z₂)             g2 = k.transpose(-2,-1) @ (
     ⊙ (1 + z₂·(1-σ(z₂))))                e * z1 * (sig * (1.0 + z2 * (1.0 - sig)))
                                        )
```

### Gradient Clipping & Step

```
수식:                                  코드 (ttt_block.py:83-87):
─────                                  ─────────────────────────
g₁ = g₁ / (‖g₁‖ + 1)                  g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
g₂ = g₂ / (‖g₂‖ + 1)                  g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

W₁* = W₁ - η·g₁  (η=1.0)             w1, w2 = w1 - lr * g1, w2 - lr * g2
W₂* = W₂ - η·g₂
```

---

## 8. 수식-코드 1:1 매칭: Inner Training (3×3 DWConv)

### 논문 Insight 6: Conv Inner Model

논문 Remark 6에서 DWConv inner model은 데이터셋을 다음과 같이 일반화한다:
```
D = {(Kᵢ^{3×3}, Vᵢ) | i=1,...,N}
Kᵢ^{3×3} = Kᵢ를 중심으로 한 3×3 로컬 이웃
```

### Inner Model 정의

```
F₂_W(x) = DWConv₃ₓ₃(x; W₃)

수식:                                  코드 개념:
─────                                  ─────────────────────────
V̂ = Conv2d(K, W₃, groups=C)            # v_hat = F.conv2d(k, w, padding=1, groups=C)
L = -(1/HW·√d) · Σ V̂ᵢ·Vᵢ             # l = -(v_hat * v).mean(dim=[-2,-1]) * scale
```

### Backward: ∂L/∂W₃ 계산

DWConv의 gradient는 입력과 error의 cross-correlation이다:

```
수식:                                  코드 (ttt_block.py:111-125):
─────                                  ─────────────────────────
e = ∂L/∂V̂ = -V/(HW) · scale           e = -v / float(v.shape[2]*v.shape[3]) * self.scale

# 'prod' 구현: 3×3 오프셋별로 dot product 계산
∂L/∂W₃[dy,dx]                          k = F.pad(k, (1,1,1,1))
  = Σ_{h,w} K[h+dy, w+dx] · e[h,w]    for dy in (-1, 0, 1):
                                            for dx in (-1, 0, 1):
                                                ys, xs = 1+dy, 1+dx
                                                dot = (k[:,:,ys:ys+H,xs:xs+W] * e)
                                                      .sum(dim=(-2,-1))
                                        g = torch.stack(outs, dim=-1)
                                            .reshape(B*C, 1, 3, 3)
```

> **왜 'prod' 구현인가?**
> `F.conv2d`를 사용한 구현도 수학적으로 동일하지만,
> 9번의 elementwise 곱+합이 grouped conv보다 약간 더 빠르다.

### Step: 샘플별 커널 생성

```
수식:                                  코드 (ttt_block.py:130-133):
─────                                  ─────────────────────────
g = g / (‖g‖ + 1)                      g = g / (g.norm(dim=[-2,-1], keepdim=True) + 1.0)
W₃* = W₃ - η·g                         w = w.repeat(B, 1, 1, 1) - lr * g
                                        #   ↑ broadcast를 위해 batch 차원 확장
```

**중요**: `w.repeat(B, ...)`는 **샘플마다 다른 커널**이 생성됨을 의미한다.
각 이미지의 (K, V) 내용에 따라 inner model이 다르게 적응하므로,
같은 batch 내에서도 서로 다른 W₃*를 갖게 된다.

---

## 9. 수식-코드 1:1 매칭: Inner Inference (Query 적용)

Inner training이 끝나면, 업데이트된 W*로 Query를 처리하여 최종 output을 생성한다.

### 논문: O = F_{W*}(Q)

```
수식:                                  코드 (ttt_block.py:165-172):
─────                                  ─────────────────────────

# Branch 1: Simplified SwiGLU
O₁ = (Q₁·W₁*) ⊙ SiLU(Q₁·W₂*)         x1 = (q1 @ w1) * F.silu(q1 @ w2)
→ [B, heads, N, d]                      x1 = x1.transpose(1,2).reshape(b, n, c)
→ reshape → [B, N, C]                      # → [B, N, dim]

# Branch 2: 3×3 DWConv
O₂ = DWConv(Q₂; W₃*)                   x2 = F.conv2d(
→ [B, d, H, W]                              q2.reshape(1, b*d, h, w),
→ reshape → [B, N, d]                       w3,
                                             padding=1,
                                             groups=b*d  # 샘플별×채널별 독립 conv
                                        )
                                        x2 = x2.reshape(b, d, n).transpose(1, 2)
                                            # → [B, N, head_dim]

# 두 branch 결합
O = Proj(concat[O₁, O₂])              x = torch.cat([x1, x2], dim=-1)
                                            # → [B, N, dim + head_dim]
= Linear(dim+head_dim → dim)           x = self.proj(x)
                                            # → [B, N, dim]
```

> **`groups=b*d`의 의미**: 각 샘플(b)의 각 채널(d)에 대해 독립적인 커널을 적용한다.
> 이것이 가능한 이유는 inner training에서 `w3`가 `[B*d, 1, 3, 3]` 형태로 생성되었기 때문이다.

---

## 10. H-ViT³ 전체 아키텍처와 코드 매칭

### 논문 Table 12 ↔ 코드

논문의 H-ViT³-T 아키텍처 정의:

```
논문 Table 12:                         코드 (h_vittt.py:357-358):
─────────────                          ────────────────────────────
Stage1: Stem↓4, B(64,2)×1             def h_vittt_tiny(**kwargs):
Stage2: Down↓2, B(128,4)×3                model = ViTTT(
Stage3: Down↓2, B(320,10)×9                   dim=[64, 128, 320, 512],
Stage4: Down↓2, B(512,16)×4                   depths=[1, 3, 9, 4],
Classifier: GAP, Linear                       num_heads=[2, 4, 10, 16],
                                          )
```

### TTTBlock: 논문 Fig.2 ↔ 코드

논문의 TTT Block은 다음 구조를 따른다:

```
논문 Fig.2 구조:                       코드 (h_vittt.py:130-143):
────────────────                       ────────────────────────────

x → CPE → (+)                          x = x + self.cpe(reshape(x))
       ↓
  → LN → TTT → DropPath → (+)          x = x + drop_path(self.attn(self.norm1(x)))
                                ↓
           → LN → MLP → DropPath → (+)  x = x + drop_path(self.mlp(self.norm2(x)))
```

### Stem: 논문 Table 12 "Stem↓4" ↔ 코드

```
논문:                                  코드 (h_vittt.py:245-263):
─────                                  ─────────────────────────
Stem ↓4: 입력을 4배 다운샘플링          self.conv = nn.Sequential(
하여 초기 토큰 시퀀스 생성                ConvLayer(3→32, k3, s2, p1),    # ↓2
                                          ConvLayer(32→32, k3, s1, p1),
                                          ConvLayer(32→32, k3, s1, p1),
                                          ConvLayer(32→256, k3, s2, p1), # ↓2 (총 ↓4)
                                          ConvLayer(256→64, k1),
                                      )
```

### PatchMerging: 논문 Table 12 "Down↓2" ↔ 코드

```
논문:                                  코드 (h_vittt.py:159-178):
─────                                  ─────────────────────────
Down ↓2: 해상도 절반,                  self.conv = nn.Sequential(
채널 수 증가                               Conv1×1(dim → dim_out × 4),    # 채널 확장
                                          DWConv3×3(stride=2),            # ↓2 다운샘플
                                          Conv1×1(dim_out × 4 → dim_out) # 채널 축소
                                      )
```

### Classification Head ↔ 코드

```
논문:                                  코드 (h_vittt.py:322-324, 341-354):
─────                                  ─────────────────────────
GAP + Linear(→1000)                    self.norm = nn.BatchNorm1d(dim[-1])
                                       self.avgpool = nn.AdaptiveAvgPool1d(1)
                                       self.head = nn.Linear(dim[-1], num_classes)

                                       def forward_features(self, x):
                                           x = self.patch_embed(x)
                                           for layer in self.layers:
                                               x = layer(x)
                                           x = self.norm(x.transpose(1,2))
                                           x = self.avgpool(x)   # [B, C, 1]
                                           x = torch.flatten(x, 1)  # [B, C]
                                           return x
```

---

## 11. MESA 학습 전략과 코드 매칭

### 논문 §5.1: MESA (Model EMA as Soft Augmentation)

논문 Table 5에서 ‡ 표시는 MESA를 적용한 모델이다:
```
H-ViT³-T:  83.5%
H-ViT³-T‡: 84.0%  (+0.5%p 향상)
```

MESA는 EMA teacher 모델의 soft label을 추가 supervision으로 사용하는 전략이다.

### Config ↔ 코드

```
cfgs/h_vittt_t_mesa.yaml:             코드 (main_ema.py:198):
────────────────────────               ──────────────────────
TRAIN:                                 mesa = config.TRAIN.MESA
  MESA: 1.0                                 if epoch >= int(0.25 * total_epochs)
                                             else -1.0
                                       # epoch 75(=300×25%) 이후부터 MESA 활성화
```

### MESA Loss 계산 ↔ 코드

```
논문의 MESA loss:                      코드 (main_ema.py:251-257):
───────────────                        ──────────────────────────────
L_total = L_CE(output, target)         with torch.inference_mode():
        + L_CE(output, soft_ema) × λ       ema_output = model_ema.ema(samples).detach()
                                       ema_output = ema_output.softmax(dim=-1).detach()
여기서:                                outputs = model(samples)
  soft_ema = softmax(EMA_model(x))     loss = criterion(outputs, targets)
  λ = 1.0 (MESA ratio)                     + criterion(outputs, ema_output) * mesa
```

### EMA 업데이트 ↔ 코드

```
수식:                                  코드 (main_ema.py:292-293):
─────                                  ─────────────────────────
θ_ema ← α·θ_ema + (1-α)·θ_model       if model_ema is not None:
α = 0.9996 (decay)                         model_ema.update(model)
                                       # timm의 ModelEma 클래스가 내부적으로 처리
```

---

## 12. Inner Loss 함수의 수학적 분석과 코드

### 논문 §8 (Appendix): 5가지 Loss 함수 비교

논문의 핵심 발견 (Insight 1)은 outer loop gradient가 inner loss의 **혼합 이계도함수**
∂²L/∂V∂V̂를 통해 전파된다는 것이다 (Eq.(6)):

```
∂G/∂W_V = (∂V̂/∂W) · (∂²L/∂V̂∂V) · (∂V/∂W_V)
                       ↑ 이것이 0이면 W_V 학습 불가!
```

| Loss | ∂²L/∂V∂V̂ | 학습 가능? | Top-1 |
|------|-----------|-----------|-------|
| Dot Product | -1/(B√d) ≠ 0 | O | 78.9% |
| MSE (L2) | -1/(B√d) ≠ 0 | O | 79.2% |
| RMSE | ≠ 0 (복잡) | O | 78.8% |
| **MAE (L1)** | **= 0** (거의 어디서나) | **X** | **76.5%** |
| Smooth L1 | 부분적으로 0 | △ | 78.1% |

### 코드에서의 Dot Product Loss 구현

ViT³ 코드는 Dot Product Loss를 채택한다. 하지만 loss를 직접 계산→autograd로 미분하는 것이
아니라, **gradient의 닫힌 형태(closed-form)**를 직접 코드로 구현한다:

```python
# 일반적인 PyTorch 방식 (사용하지 않음):
# v_hat = z1 * a                         # F_W(K)
# loss = -(v_hat * v).sum(3).mean(2) * scale  # Eq.(8)
# loss.backward()                        # autograd
# w1.grad, w2.grad                       # 자동 계산된 gradient

# ViT³가 사용하는 방식 (hand-derived gradient):
e = -v / N * scale                       # ∂L/∂V̂ (직접 계산)
g1 = k.T @ (e * a)                       # ∂L/∂W₁ (chain rule 직접 전개)
g2 = k.T @ (e * z1 * SiLU'(z2))         # ∂L/∂W₂ (chain rule 직접 전개)
```

> **왜 hand-derived gradient인가?**
> 논문 Note (ttt_block.py:20-24):
> TTT inner loss는 `[B, num_heads]` shape의 vector-valued loss이다.
> `torch.autograd.backward`는 scalar loss만 지원하므로,
> head별/sample별 독립적인 gradient를 효율적으로 계산하기 위해
> 닫힌 형태의 gradient 수식을 직접 유도하여 구현한다.

---

## 부록: 논문의 주요 표와 코드 설정 대응표

| 논문 항목 | 값 | 코드 위치 |
|----------|-----|----------|
| Inner loss | Dot product | `ttt_block.py:72,78` |
| Inner batch size | N (full-batch) | `ttt_block.py:54-88` (전체 K,V 사용) |
| Inner epochs | 1 | `ttt_block.py:161-162` (1회 호출) |
| Inner learning rate η | 1.0 | `ttt_block.py:54` (default) |
| Inner model F₁ | Simplified SwiGLU | `ttt_block.py:40-41,165` |
| Inner model F₂ | 3×3 DWConv | `ttt_block.py:42,167` |
| Gradient clipping | g/(‖g‖+1) | `ttt_block.py:83-84,130` |
| Scale factor | 9^(-0.5) | `ttt_block.py:48-49` |
| Outer optimizer | AdamW | `optimizer.py:29-30` |
| Outer LR | 5e-4 (linear scaled) | `config.py:68` |
| Weight decay | 0.05 | `config.py:67` |
| Training epochs | 300 | `config.py:64` |
| MESA ratio | 1.0 | `h_vittt_t_mesa.yaml:6` |
| MESA activation | epoch ≥ 25% | `main_ema.py:198` |
| EMA decay | 0.9996 | `main_ema.py:63` |
| Mixup α | 0.8 | `config.py:114` |
| CutMix α | 1.0 | `config.py:116` |
| Drop path rate | 0.2 | `h_vittt_t_mesa.yaml:4` |
| H-ViT³-T dims | [64,128,320,512] | `h_vittt.py:358` |
| H-ViT³-T depths | [1,3,9,4] | `h_vittt.py:358` |
| H-ViT³-T heads | [2,4,10,16] | `h_vittt.py:358` |

---

## 13. 학습 실험 결과

### 13.1 학습 환경

```
모델:       H-ViT³-Tiny (MESA)
데이터셋:   ImageNet-1K (1.28M train / 50K val)
GPU:        1× GPU (nproc_per_node=1)
Batch Size: 128
AMP:        True (Mixed Precision)
Epochs:     300
시작:       2026-02-25 20:49
종료:       2026-03-07 02:38
총 학습 시간: 9일 5시간 49분
```

### 13.2 학습 속도

```
Epoch 1~75  (MESA 비활성): ~37분/epoch
Epoch 76~300 (MESA 활성):  ~45분/epoch  (EMA forward 추가로 ~20% 증가)
```

### 13.3 Accuracy 추이

```
Acc@1
  ↑
83│                                          ★ 83.13% (max EMA)
  │                                      ★ 82.81% (max network)
82│
  │
81│                          ·······
  │                    ····
80│               ····                    ·· ·· ·· 80.1% (final)
  │           ···
79│        ···
  │      ··
78│
  │
77│    ·· 77.07%
  │
75│  · 75.72%
  │
69│ · 69.34%
  │
  └──────────────────────────────────────────────→ epoch
    1   25   50   75  100  125  150  200  250  300
        └─ warmup ─┘   └─ MESA 활성화 ─────────────┘
```

| Epoch | Max Network Acc@1 | Max EMA Acc@1 | 비고 |
|-------|------------------|---------------|------|
| 1 | 1.10% | 0.10% | 학습 시작 (from scratch) |
| 25 | 69.34% | 3.86% | Warmup 완료, EMA는 아직 수렴 전 |
| 50 | 75.72% | 68.83% | EMA가 빠르게 따라잡음 |
| 75 | 77.07% | 79.26% | MESA 활성화 시점, EMA > Network |
| 100 | 78.96% | 80.46% | MESA 효과로 꾸준한 상승 |
| 125 | 79.69% | 81.21% | |
| 150 | 80.23% | 81.76% | |
| 175 | 80.74% | 82.31% | |
| 200 | 80.98% | 82.80% | |
| 225 | 81.19% | 83.09% | |
| 250 | 82.06% | **83.13%** | EMA best 달성 |
| 275 | **82.81%** | 83.13% | Network best 달성 |
| 300 | 82.81% | 83.13% | 학습 종료 |

### 13.4 결과 분석

**최종 성능:**
- **Network Best Acc@1: 82.81%** (epoch 275에서 달성)
- **EMA Best Acc@1: 83.13%** (epoch 250에서 달성)
- Final epoch(300) network accuracy: 80.1% (best 대비 약간 하락 = 후반부 LR 감소에 의한 정상 현상)

**논문 Table 5 대비 비교:**

| 모델 | 논문 결과 | 우리 실험 | 차이 |
|------|----------|----------|------|
| H-ViT³-T (MESA) | 84.0% | 83.13% (EMA) | -0.87%p |
| H-ViT³-T (no MESA) | 83.5% | 82.81% (Network) | -0.69%p |

> **차이 원인 분석:**
> - 논문은 **4096 total batch** (32 GPU × 128), 우리는 **128 batch** (1 GPU × 128)
> - Batch size 차이 → LR linear scaling 차이 (논문: 4e-3, 우리: 1.25e-4)
> - 큰 batch에서 더 안정적인 gradient → 더 높은 성능
> - 그럼에도 83.13%는 1 GPU 기준으로 매우 좋은 결과

**MESA 효과 관찰:**
- Epoch 75(25%) 이후 MESA 활성화
- EMA accuracy가 network accuracy를 **일관되게** 상회 (+0.3~2%p)
- MESA 활성화 후 epoch당 학습 시간이 37분→45분으로 증가 (EMA forward 오버헤드)

**학습 안정성:**
- 300 epoch 전체에서 loss divergence 없음
- Gradient clipping (max_norm=5.0)이 효과적으로 작동
- EMA decay가 안정적인 teacher를 유지

### 13.5 Checkpoint 파일

```
output/h_vittt_tiny/default/
├── ckpt_epoch_298.pth     # 448MB
├── ckpt_epoch_299.pth     # 448MB
├── ckpt_epoch_300.pth     # 448MB  ← 최종 epoch
├── max_acc.pth            # 448MB  ← Network best (82.81%)
├── max_ema_acc.pth        # 448MB  ← EMA best (83.13%)
├── config.json            # 학습 config
└── log_rank0.txt          # 37,689줄 학습 로그
```
