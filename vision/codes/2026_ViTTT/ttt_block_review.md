# TTT Block Code Review

**Source:** `ttt_block.py` — Test-Time Training block for the ViT^3 model
**Paper:** [ViT^3: Unlocking Test-Time Training in Vision](https://arxiv.org/abs/2512.01643)

---

## Architecture Overview

The `TTT` module replaces standard self-attention with two parallel inner-learnable sub-modules whose weights are updated at test time using each input's own (k, v) pairs, then applied to queries.

---

## What is TTT and Where Does It Live in This Code?

Traditional models **freeze all weights** after training. TTT **updates weights on-the-fly
for each input** during inference. The entire TTT mechanism is contained in the
**inner training + inner inference** pattern described below.

### TTT Components (the core novelty)

| Component | Location | What It Does |
|-----------|----------|--------------|
| `inner_train_simplified_swiglu()` | Lines 54-88 | Updates `w1, w2` using the current input's own `(k1, v1)` as a self-supervised signal |
| `inner_train_3x3dwc()` | Lines 90-134 | Updates `w3` using the current input's own `(k2, v2)` |
| Inner inference | Lines 165-168 | Applies the **updated** (per-sample adapted) weights to queries |

The inner training loop in both branches follows the same pattern:

```
1. Forward:  predict v_hat from k using current weights
2. Backward: compute gradient of reconstruction loss ‖v_hat - v‖  (hand-derived, closed-form)
3. Clip:     stabilize the gradient via g / (‖g‖ + 1)
4. Step:     w' = w - lr * gradient   (one-step SGD)
```

The **reconstruction loss** (predict `v` from `k`) is the self-supervised objective —
no labels needed, which is why it works at test time.

### Non-TTT Components (standard machinery)

| Component | Location | Role |
|-----------|----------|------|
| `self.qkv` | Line 39 | Standard linear projection (frozen at test time) |
| `self.proj` | Line 46 | Standard output projection (frozen at test time) |
| RoPE application | Lines 149-154 | Positional encoding |
| Concatenation + projection | Lines 171-172 | Output fusion |

### Dual Role of `w1, w2, w3`

These parameters are **not** fixed weights — they are **initial conditions** for per-sample optimization:

- During **outer training** (normal backprop): they learn good initial values
- During **inner training** (TTT, every forward pass): they are adapted per-sample from those initial values

Each input gets its own personalized version of these weights, derived by one gradient
step on the self-supervised reconstruction loss.

---

## Flowchart

```
                        Input x [B, N, C]
                              │
                              ▼
                     ┌────────────────┐
                     │  Linear QKV    │  nn.Linear(dim, dim*3 + head_dim*3)
                     │  Projection    │
                     └────────┬───────┘
                              │
              ┌───────────────┴────────────────┐
              │ Split into 6 tensors           │
              │ q1, k1, v1  (dim each)         │
              │ q2, k2, v2  (head_dim each)    │
              └───────┬───────────────┬────────┘
                      │               │
            ┌─────────┴──────┐  ┌─────┴────────────┐
            │  Branch 1:     │  │  Branch 2:       │
            │  Multi-Head    │  │  Spatial         │
            │  Token-wise    │  │  Depth-wise Conv │
            │                │  │                  │
            │ Reshape to     │  │ Reshape to       │
            │ [B,H,N,d]      │  │ [B,d,H,W]        │
            │ + optional     │  │ (spatial layout) │
            │   RoPE on q,k  │  │                  │
            └────────┬───────┘  └────────┬─────────┘
                     │                   │
        ┌────────────┴───┐      ┌────────┴────────┐
        │  INNER TRAIN   │      │  INNER TRAIN    │
        │  Simplified    │      │  3x3 Depthwise  │
        │  SwiGLU        │      │  Convolution    │
        │                │      │                 │
        │  Input:        │      │  Input:         │
        │   k1, v1,      │      │   k2, v2,       │
        │   w1, w2       │      │   w3            │
        │                │      │                 │
        │  Forward:      │      │  Forward:       │
        │   z1 = k@w1    │      │   (implicit     │
        │   z2 = k@w2    │      │    conv2d)      │
        │   a = z2*σ(z2) │      │                 │
        │   (SwiGLU)     │      │  Backward:      │
        │                │      │   e = -v/HW*s   │
        │  Backward:     │      │   g = correlate │
        │   e = -v/N*s   │      │       (k, e)    │
        │   g1 = kᵀ(e*a) │      │                 │
        │   g2 = kᵀ(...) │      │  Clip & Step:   │
        │                │      │   w3' = w3-lr*g │
        │  Clip & Step:  │      └────────┬────────┘
        │   w1' = w1-g1  │               │
        │   w2' = w2-g2  │               │
        └───────┬────────┘               │
                │                        │
        ┌───────┴────────┐      ┌────────┴────────┐
        │  INNER INFER   │      │  INNER INFER    │
        │                │      │                 │
        │  x1 = (q@w1')  │      │  x2 = conv2d(   │
        │     * SiLU(    │      │    q2, w3')     │
        │       q@w2')   │      │                 │
        │                │      │  Reshape to     │
        │  Reshape to    │      │  [B, N, d]      │
        │  [B, N, C]     │      │                 │
        └───────┬────────┘      └────────┬────────┘
                │                        │
                └───────────┬────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  Concatenate   │  [B, N, C+d]
                   │  x1 and x2     │
                   └────────┬───────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  Output Proj   │  nn.Linear(dim + head_dim, dim)
                   └────────┬───────┘
                            │
                            ▼
                      Output [B, N, C]
```

---

## Detailed Walkthrough

### 1. QKV Projection

- A single `nn.Linear` projects input `x [B, N, C]` into **6 components**:
  - `q1, k1, v1` each of size `C` (for the SwiGLU branch, multi-head)
  - `q2, k2, v2` each of size `head_dim = C // num_heads` (for the conv branch, single-head spatial)

### 2. Branch 1 — Simplified SwiGLU Inner Module

**Learnable parameters:** `w1, w2` of shape `[1, num_heads, head_dim, head_dim]`

**Inner training** (`inner_train_simplified_swiglu`):

| Step | Operation | Purpose |
|------|-----------|---------|
| Forward | `z1 = k @ w1`, `z2 = k @ w2`, `a = z2 * sigmoid(z2)` | SwiGLU activation |
| Backward | Hand-derived gradients `g1`, `g2` from reconstruction loss | Avoid `torch.autograd` (loss is per-head, non-scalar) |
| Clip | `g = g / (‖g‖ + 1)` | Stability |
| Step | `w' = w - lr * g` | Single-step SGD update |

**Inner inference:**
```
x1 = (q1 @ w1') * SiLU(q1 @ w2')
```

### 3. Branch 2 — 3x3 Depthwise Convolution Inner Module

**Learnable parameter:** `w3` of shape `[head_dim, 1, 3, 3]`

**Inner training** (`inner_train_3x3dwc`):

| Step | Operation | Purpose |
|------|-----------|---------|
| Forward | Implicit `conv2d(k, w, padding=1, groups=C)` | Spatial convolution |
| Backward | Two implementations: `conv` (cross-correlation via conv2d) or `prod` (manual sliding-window dot products) | `prod` is default and slightly faster |
| Clip | `g = g / (‖g‖ + 1)` | Stability |
| Step | `w3' = w3 - lr * g` | Per-sample update (w3 is repeated B times) |

**Inner inference:**
```
x2 = conv2d(q2, w3', padding=1, groups=B*d)
```
Note: `w3` becomes per-sample after inner training (`[B*d, 1, 3, 3]`), so inference uses grouped convolution.

### 4. Output Fusion

- Concatenate `x1 [B, N, C]` and `x2 [B, N, d]` along the channel dimension
- Project back to `[B, N, C]` via `nn.Linear(dim + head_dim, dim)`

---

## Key Design Decisions

| Decision | Detail |
|----------|--------|
| **Hand-derived gradients** | The TTT loss is per-head/per-sample (shape `[B, num_heads]`), which `torch.autograd.backward` cannot handle directly. Closed-form gradient expressions bypass this. |
| **Gradient clipping** | `g / (‖g‖ + 1)` — a soft normalization that bounds gradient magnitude without hard thresholds. |
| **Single-step update** | Only one gradient step per forward pass (lr=1.0 by default). |
| **Scale factor** | `scale = 9^{-0.5}` based on the equivalent head dimension of the 3x3 conv branch (1 channel x 3x3 kernel = 9). Shared across both branches. |
| **Per-sample weights** | After inner training, `w3` is per-sample. `w1, w2` are also per-sample (broadcasted from shared init). This means the model adapts its weights to each individual input. |

---

## Parameter Summary

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `qkv.weight` | `[dim*3 + head_dim*3, dim]` | Joint QKV projection |
| `w1` | `[1, num_heads, head_dim, head_dim]` | SwiGLU branch weight 1 (init) |
| `w2` | `[1, num_heads, head_dim, head_dim]` | SwiGLU branch weight 2 (init) |
| `w3` | `[head_dim, 1, 3, 3]` | 3x3 depthwise conv kernel (init) |
| `proj.weight` | `[dim, dim + head_dim]` | Output projection |
