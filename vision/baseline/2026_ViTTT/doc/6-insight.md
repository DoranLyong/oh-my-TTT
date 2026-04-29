# ViT³ — The 6 Practical Insights

> Paper: *ViT³: Unlocking Test-Time Training in Vision* (arXiv:2512.01643)
> Purpose: Presentation-ready reference for the 6 insights that shape TTT design for vision.

---

## Why Do We Need 6 Insights?

TTT replaces Softmax Attention with a learnable inner model $\mathcal{F}_W$.
This flexibility opens a **vast design space** with decisions that don't exist in standard Transformers:

1. **How** to train the inner model? (Loss, Batch/Epoch, Learning Rate)
2. **What** architecture for the inner model? (Width, Depth, Type)

The paper systematically explores these 6 axes on ImageNet classification (ViT³-Small, 224×224).

```
              Design Space of TTT for Vision
             ┌─────────────────────────────────┐
             │  Part A: Inner Training Config   │
             │    Insight 1 — Loss Function      │
             │    Insight 2 — Batch & Epoch      │
             │    Insight 3 — Learning Rate      │
             ├─────────────────────────────────┤
             │  Part B: Inner Model Design       │
             │    Insight 4 — Width (wider ↑)    │
             │    Insight 5 — Depth (deeper ↓)   │
             │    Insight 6 — Convolution (best)  │
             └─────────────────────────────────┘
                            ↓
                     ViT³ TTT Block
```

---

## Part A: Inner Training Configuration

### Insight 1 — Inner Loss Function

> **The mixed second derivative $\frac{\partial^2 \mathcal{L}}{\partial V \, \partial \hat{V}}$ must not vanish.**

#### Why It Matters

The outer-loop gradient to $W_V$ flows through this term (Eq.6):

$$
\frac{\partial G}{\partial W_V}
= \frac{\partial \hat{V}_B}{\partial W}
\cdot \underbrace{\frac{\partial^2 \mathcal{L}}{\partial \hat{V}_B \, \partial V_B}}_{\text{must} \neq 0}
\cdot \frac{\partial V_B}{\partial W_V}
$$

If the middle term equals zero, $W_V$ receives **no gradient** and learning collapses.

#### 5 Loss Candidates

| Loss | Formula | $\frac{\partial^2 \mathcal{L}}{\partial V \, \partial \hat{V}}$ | Top-1 |
|---|---|---|---|
| **Dot Product** | $-\frac{1}{B\sqrt{d}} \sum_i \hat{V}_i V_i^\top$ | $-\frac{1}{B\sqrt{d}}$ (constant) | **78.9** |
| MSE | $\frac{1}{2B\sqrt{d}} \sum_i \|\hat{V}_i - V_i\|^2$ | $-\frac{1}{B\sqrt{d}}$ (constant) | 79.2 |
| RMSE | $\sqrt{\text{MSE}}$ | data-dependent, non-zero | 78.8 |
| MAE | $\frac{1}{B\sqrt{d}} \sum_i \|\hat{V}_i - V_i\|_1$ | **= 0 (a.e.)** | 76.5 |
| Smooth L1 | $\frac{1}{B\sqrt{d}} \sum_i \text{smooth\_l1}(\hat{V}_i - V_i)$ | **= 0 in linear region** | 78.1 |

- MAE uses $\text{sign}(\cdot)$ which is piecewise constant — its derivative vanishes almost everywhere.
- Smooth L1 has a linear region ($|x| > 1$) where the same problem occurs.

#### ViT³ Choice

**Dot Product Loss** — simplest formula, constant non-zero derivative, fastest computation.

---

### Insight 2 — Inner Batch Size and Epochs

> **A single epoch of full-batch gradient descent works best for vision.**

#### Why Full-Batch?

Mini-batch gradient descent imposes a **causal bias**: earlier batches influence the model state seen by later batches. This sequential dependency is natural for *language* (tokens are ordered) but suboptimal for *vision* (spatial, non-causal).

#### Why 1 Epoch?

More epochs yield marginal accuracy gains but severely hurt throughput — and risk divergence.

| Epochs | Batch Size | FPS | Top-1 |
|---|---|---|---|
| **1** | **N** | **1315** | **78.9** |
| 1 | N/2 | 1201 | 78.6 |
| 1 | N/3 | 1131 | 78.3 |
| 2 | N | 971 | 79.1 |
| 3 | N | 787 | 79.2 |
| 4 | N | 659 | 57.0 (diverged) |

#### ViT³ Choice

**1 epoch, full-batch ($B = N$)** — all tokens processed at once, no sequential dependency.

---

### Insight 3 — Inner Learning Rate

> **A fixed learning rate $\eta = 1.0$ is effective.**

The inner-loop weight update:

$$
W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
$$

| $\eta$ | 0.1 | 0.2 | 0.5 | **1.0** | 2.0 | 5.0 | 10.0 | Dynamic |
|---|---|---|---|---|---|---|---|---|
| Top-1 | 77.5 | 78.1 | 78.7 | **78.9** | 78.9 | 76.7* | 76.9* | 78.7 |

- Too small ($\eta < 0.5$): inner model barely adapts, insufficient information compression.
- Too large ($\eta > 5.0$): training instability, outer optimization diverges.
- Dynamic per-token rate $\eta_i = \eta \cdot \sigma(x_i W_\eta)$ from prior work [55, 77] is less effective for vision.

#### ViT³ Choice

**$\eta = 1.0$, fixed** — simple, stable, and performant.

---

## Part B: Inner Model Design

### Insight 4 — Scaling Width (Wider = Better)

> **Increasing inner model capacity consistently improves performance.**

Using a two-layer MLP as the inner model, varying hidden dimension ratio $r$:

| Inner Model | Hidden Dim | FLOPs | Top-1 |
|---|---|---|---|
| MLP, $r=1$ | $d$ | 4.58G | 78.9 |
| MLP, $r=2$ | $2d$ | 4.92G | 79.2 |
| MLP, $r=3$ | $3d$ | 5.27G | 79.5 |
| MLP, $r=4$ | $4d$ | 5.62G | 79.6 |

This is a key advantage over Linear Attention, which is stuck with a fixed $d \times d$ linear state.

> **Remark 4**: An inner module costs ~4x the FLOPs of an equivalent outer module (1 forward on K + 2 backward + 1 forward on Q). So lightweight but expressive designs are critical.

---

### Insight 5 — Scaling Depth (Deeper = Worse)

> **Deep inner models suffer from optimization difficulties in current TTT settings.**

| Inner Model | Layers | Top-1 |
|---|---|---|
| FC | 1 ($d \times d$) | 79.1 |
| MLP | 2 ($d \to d \to d$) | 78.9 |
| MLP | 3 ($d \to d \to d \to d$) | 77.5 |

Deeper = more capacity in theory, but **worse** in practice. Two reasons:

1. **Outer-loop**: $W_0$ for deep inner modules is harder to learn end-to-end.
2. **Inner-loop**: deeper networks cause exploding/vanishing gradients in 1-step SGD.

#### Evidence: Removing the Output Layer Helps

| Inner Model | Formula | Top-1 |
|---|---|---|
| 2-layer MLP | $\text{SiLU}(xW_1) \cdot W_2$ | 78.9 |
| Constrained (no output layer) | $\text{SiLU}(xW_1)$ | **79.4** |
| Full SwiGLU | $(xW_1 \odot \text{SiLU}(xW_2)) \cdot W_3$ | 79.0 |
| Simplified SwiGLU (no output layer) | $xW_1 \odot \text{SiLU}(xW_2)$ | **79.7** |

Standard remedies (residual connections, identity init) provide only limited help:

| Strategy | Top-1 |
|---|---|
| $\text{SiLU}(xW_1)W_2 + x$ (residual) | 78.8 |
| $\text{SiLU}(xW_1)(W_2 + I)$ (implicit residual) | 79.1 |
| $\text{SiLU}(xW_1)W_2$, $W_2$ init as $I$ | 79.0 |

All underperform the constrained designs — the issue is **fundamental depth optimization difficulty**, not initialization.

#### ViT³ Solution

Remove the output layer entirely → **Simplified SwiGLU** (see below).

---

### Insight 6 — Convolution as Inner Model (Best Choice)

> **Convolutional architectures are particularly appropriate as inner models for vision.**

| Inner Model | Params | FLOPs | Top-1 |
|---|---|---|---|
| FC($x$) | 23.2M | 4.34G | 79.1 |
| $\text{FC}(x) \odot \text{SiLU}(\text{FC}(x))$ | 23.5M | 4.58G | 79.7 |
| Conv 3x3 | 25.5M | 5.27G | 79.9 |
| **DWConv 3x3** | **22.9M** | **4.25G** | **80.1** |

DWConv 3x3 achieves the **best accuracy** with the **fewest parameters and FLOPs**.

#### Why It Works: Global + Local Fusion

```
  Inner Training (gradient descent)     Convolution (3x3 kernel)
  ─────────────────────────────────     ────────────────────────
  Compresses GLOBAL context             Applies LOCAL spatial
  from all (K, V) pairs                 filtering (3x3 receptive field)
  into the kernel weights               on the query features
           │                                      │
           └──────────── COMBINED ────────────────┘
                          │
              Output captures BOTH
              global relationships AND
              local spatial structure
```

- Each channel has its own independent $3 \times 3$ kernel (weight shape: $[d, 1, 3, 3]$).
- After inner training, each image in the batch has its **own adapted kernel** — per-sample specialization.
- Only $9d$ parameters per inner model, vs. $d^2$ for a linear layer.

---

## Summary: From 6 Insights to the ViT³ Block

```
  Insight 1: Dot Product Loss    ─┐
  Insight 2: Full-batch, 1 epoch  ├──→  Inner Training Config
  Insight 3: η = 1.0, fixed      ─┘
                                          │
  Insight 4: Wider = better      ─┐      ↓
  Insight 5: Deeper = worse       ├──→  Inner Model Design
  Insight 6: DWConv is best      ─┘
                                          │
                                          ↓
                                   ViT³ TTT Block
                              ┌──────────┴──────────┐
                        Simplified SwiGLU       3x3 DWConv
                        (num_heads - 1)          (1 head)
```

### Final Design Decisions

| Decision | Choice | Insight |
|---|---|---|
| Inner loss | Dot Product | #1 |
| Batch / Epoch | Full-batch ($B=N$), 1 epoch | #2 |
| Inner LR | $\eta = 1.0$, fixed | #3 |
| Inner model (main) | Simplified SwiGLU: $xW_1 \odot \text{SiLU}(xW_2)$ | #4, #5 |
| Inner model (conv head) | 3x3 Depthwise Convolution | #6 |

---

## Presentation Flow Recommendation

```
  TTT Layer Explanation
         │
         ▼
  "TTT is flexible — but HOW to design it for vision?"
         │
         ▼
  ┌─ Part A: How to Train ───────────────────┐
  │  Slide 1: Insight 1 (Loss) ← most        │
  │           theoretical, show Eq.6 diagram  │
  │  Slide 2: Insight 2 (Batch/Epoch)         │
  │  Slide 3: Insight 3 (Learning Rate)       │
  └───────────────────────────────────────────┘
         │
         ▼
  ┌─ Part B: What to Use ────────────────────┐
  │  Slide 4: Insight 4 (Width ↑) — positive  │
  │  Slide 5: Insight 5 (Depth ↓) — problem   │
  │  Slide 6: Insight 6 (Conv) — solution      │
  └───────────────────────────────────────────┘
         │
         ▼
  Summary Slide: 6 Insights → ViT³ Block
```

**Tips**:
- Insight 1 is the most theoretical — prepare an Eq.6 gradient flow diagram.
- Insights 5 → 6 form a natural **problem → solution** arc that keeps the audience engaged.
- Show one row of experimental results (numbers) per insight for evidence.
