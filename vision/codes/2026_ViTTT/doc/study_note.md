# ViT³ Study Note — A Concept Mind-Map

> Paper: *ViT³: Unlocking Test-Time Training in Vision* (arXiv:2512.01643)
> Authors: Dongchen Han, Yining Li, Tianyu Li, Zixuan Cao, Ziming Wang, Jun Song, Yu Cheng, Bo Zheng, Gao Huang
> Tsinghua University & Alibaba Group

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
                         Vision Transformers
                               |
                    "O(N²) is too expensive"
                               |
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
        Linear Attention    Mamba (SSM)    Test-Time Training
          (O(N), weak)     (O(N), scan)     (O(N), flexible)
               │                               │
               │         "How to design        │
               │          TTT for vision?"     │
               │               │               │
               └───────────────┼───────────────┘
                               ▼
                    6 Practical Insights
                               │
                               ▼
                          ViT³ Model
                    ┌──────────┴──────────┐
                    ▼                     ▼
               ViT³ (flat)         H-ViT³ (hierarchical)
                                          │
                                   ┌──────┼──────┐
                                   ▼      ▼      ▼
                              Classify  Detect  Segment
```

---

## 1. Softmax Attention

### Definition
The standard attention mechanism in Transformers.
Given queries Q, keys K, values V (all projected from input x):

$$O = \text{Softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V \tag{Eq.1}$$

Each output token is a weighted average of all values,
where the weights come from query-key similarity passed through Softmax.

### Properties
- **Complexity**: O(N²d) in time and O(N²) in memory, because the N×N attention matrix is explicitly computed.
- **Expressive power**: Very strong — the full N×N interaction lets every token attend to every other token with fine-grained, input-dependent weights.
- **MLP interpretation** (Eq.2): Softmax attention is structurally equivalent to a 2-layer MLP of width N:

  $$O = \text{Softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V = \sigma(Q \cdot W_1) \cdot W_2 \tag{Eq.2}$$

  where **W₁ = K^T/√d**, **W₂ = V**, **σ = Softmax**. The $1/√d$ scaling from Eq.1 is absorbed into W₁.

  Why this is a 2-layer MLP: the computation follows Q → Linear(×W₁) → Activation(Softmax) → Linear(×W₂) → O,
  which is identical to a hidden-width-N MLP. The key difference from a standard MLP is that
  W₁ and W₂ are not learned parameters — they are *constructed directly from the input* (K, V).

  Lineage of this interpretation: Fast Weight Programmers (Schmidhuber, 1992 [49]) → "Linear Transformers
  Are Secretly Fast Weight Programmers" (Schlag et al., 2021 [48]) → formalized for TTT by Sun et al. [53].

  This is the key insight that connects attention to the TTT framework.

### Application
- Backbone of ViT [13], DeiT [56], Swin Transformer [35], CSwin [12], etc.
- State-of-the-art across classification, detection, segmentation, generation.
- The baseline that all O(N) methods aim to match.

### Links
- → **Linear Attention**: removes the Softmax non-linearity to achieve O(N)
- → **TTT**: generalizes the MLP interpretation — instead of "constructing" the MLP, it "trains" an arbitrary inner model
- → **DeiT** [56]: the specific ViT variant used as the experimental backbone in §4

---

## 2. Linear Attention

### Definition
A family of attention variants that replace Softmax with a linear kernel,
enabling a change in computation order from $\left(QK^\top \right)V$ to $Q\left(K^\top V\right)$:

$$O_i = \frac{Q_i \left(\sum_j K_j^\top V_j\right)}{Q_i \left(\sum_j K_j^\top\right)} \tag{Eq.3}$$

Ignoring the scalar denominator, this simplifies to:

$$O = Q(K^\top V) \;\triangleq\; Q \cdot W = \text{FC}(Q) \tag{Eq.4}$$

### Properties
- **Complexity**: O(Nd²) — linear in sequence length N.
- **State**: a single d×d matrix W = K^T V. This is a "one-shot" compression of (K, V) into a linear layer.
- **Limitation**: The d×d linear state is too small to capture rich non-linear relationships, leading to weaker expressive power than Softmax [6, 20, 45].

### Application
- Used in Performer [6], CosFormer [45], FLatten [20], MILA [22], SOFT++ [38], etc.
- Achieves O(N) but consistently underperforms Softmax attention in practice.
- Paper's Tab.5 shows linear attention methods (SOFT++, VVT) falling behind both Transformers and TTT.

### Links
- → **Softmax Attention**: Linear Attention is obtained by dropping the Softmax
- → **TTT**: Linear Attention is a *special case* of TTT where the inner model is a single linear layer and the "training" is just direct matrix construction (K^T V)
- → **ViT³**: replaces both Softmax and Linear Attention with a TTT block

---

## 3. Test-Time Training (TTT) — The Core Paradigm

### Definition
TTT reformulates the attention operation as an online learning problem.
Instead of computing attention weights, TTT:

1. Treats (K, V) pairs as a mini-dataset D = {(Kᵢ, Vᵢ)}
2. Trains (adapts) a compact inner model F_W on this dataset via gradient descent
3. Applies the trained model to queries: O = F_{W*}(Q)

The training step (Eq.5):

$$\hat{V}_B = \mathcal{F}_W(K_B), \quad W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}(\hat{V}_B, V_B)}{\partial W} \tag{Eq.5}$$

### Properties
- **Complexity**: O(N) when the inner model itself is O(N) (e.g., an MLP or a depthwise convolution).
- **Flexibility**: The inner model F_W can be *any* differentiable module — MLP, GLU, convolution, etc. This is the key advantage over fixed-form Linear Attention.
- **Non-linear state**: Unlike Linear Attention's d×d linear matrix, TTT stores information in the full parameter set of a non-linear network.
- **Differentiable inner loop**: The entire inner training process is differentiable, so the outer model can learn W₀ (the initialization) end-to-end via backpropagation through the inner loop.
- **Per-sample adaptation**: Each input gets its own adapted W*. The adaptation is not carried over to subsequent inputs — W always resets to W₀.
- **Always-on**: Inner training runs during *both* training and inference. That is literally why it is called "Test-Time" Training.

### Application
- Origin: Yu Sun et al. [55] proposed TTT for language modeling (ICML 2025).
- Extensions: LaCT [77] for language, one-minute video generation [9], 3D reconstruction [5].
- This paper: first *systematic* study of TTT for vision, resulting in ViT³.

### Links
- → **Softmax Attention**: can be viewed as TTT with a specific inner model (width-N MLP)
- → **Linear Attention**: special case where F_W is a linear layer and training = direct construction
- → **MAML / Meta-learning** [17]: shares the bi-level optimization structure (inner loop learns task-specific params, outer loop learns the initialization)
- → **Fast Weight Programmers** [49]: early concept of dynamically generating weights from input
- → **Inner Loop / Outer Loop**: the two levels of learning in TTT (see §4 below)

---

## 4. Inner Loop vs. Outer Loop

### Definition

| | Inner Loop | Outer Loop |
|---|---|---|
| **What** | Mini-training inside TTT block's forward() | Standard model training (e.g., 300-epoch ImageNet training) |
| **Scope** | Updates temporary copies of w1, w2, w3 | Updates ALL model parameters (qkv, proj, w1, w2, w3, norms, stem, head, ...) |
| **Loss** | Dot product loss: L = -(1/N√d) Σ F_W(Kᵢ)·Vᵢ^T | Cross-entropy on ImageNet labels |
| **Optimizer** | Hand-derived 1-step SGD, lr = 1.0 | AdamW, lr = 4e-3, weight decay 0.05 |
| **When** | Every forward pass (train AND inference) | Only during training |
| **Gradient** | ∂L_inner/∂W — used to update W* | ∂L_CE/∂θ — passes *through* the inner loop |

### Properties
- The outer loop gradient passes through the inner loop via second-order differentiation (Eq.6).
  This is what makes W₀ a *learned initialization* rather than a fixed one.
- The inner loop does NOT use torch.autograd. Instead, the paper derives closed-form gradient expressions by hand, because the inner loss is a per-head, per-sample vector (not a scalar).
- Inner loop creates new tensors (w1 - lr*g1); it never modifies self.w1 in-place.

### Application
- Inner loop: `ttt_block.py:54-88` (SwiGLU) and `ttt_block.py:90-134` (DWConv)
- Outer loop: `main_ema.py` training loop with AdamW

### Links
- → **MAML** [17]: identical bi-level structure — inner loop = task adaptation, outer loop = meta-learning
- → **Gradient flow (Eq.6)**: the critical path through which W_V receives outer-loop gradients

---

## 5. Inner Training Configuration — The Three Hyperparameters

### 5.1 Inner Loss Function

#### Definition
The self-supervised objective used in the inner loop.
The paper evaluates five candidates (all measure how well F_W(K) predicts V):

| Loss | Formula | ∂²L/∂V∂V̂ |
|---|---|---|
| Dot Product | L = -(1/B√d) Σ V̂ᵢ Vᵢ^T | -1/(B√d) ≠ 0 |
| MSE (L2) | L = (1/2B√d) Σ ‖V̂ᵢ - Vᵢ‖² | -1/(B√d) ≠ 0 |
| RMSE | L = √(MSE) | non-zero (complicated) |
| MAE (L1) | L = (1/B√d) Σ ‖V̂ᵢ - Vᵢ‖₁ | **= 0 a.e.** |
| Smooth L1 | L = (1/B√d) Σ smooth_l1(V̂ᵢ - Vᵢ) | **= 0 in linear region** |

#### Properties — Insight 1
> Loss functions for which the mixed second derivative ∂²L/(∂V ∂V̂) vanishes are not suitable for TTT.

Why? The outer-loop gradient to W_V flows through this mixed derivative (Eq.6):

$$\frac{\partial G}{\partial W_V} = \frac{\partial \hat{V}_B}{\partial W} \cdot \frac{\partial^2 \mathcal{L}}{\partial \hat{V}_B \partial V_B} \cdot \frac{\partial V_B}{\partial W_V} \tag{Eq.6}$$

If the middle term is zero, W_V gets no gradient signal, and learning collapses.

#### Application
- **ViT³ uses Dot Product Loss** — simplest, non-zero mixed derivative, and fastest.
- Tab.1 results: Dot Product 78.9%, MSE 79.2%, RMSE 78.8%, MAE **76.5%**, Smooth L1 78.1%.
- In code: `ttt_block.py:72` — the loss is implicit (never computed as a scalar) because the hand-derived backward directly computes ∂L/∂V̂.

#### Links
- → **Eq.6**: the mathematical reason behind Insight 1
- → **W_V**: the value projection matrix whose learning depends on this derivative
- → **Remark 1**: Softmax Attention implicitly satisfies F(Kᵢ) ≈ Vᵢ, which is exactly the TTT training target

#### Detailed Derivations (Appendix §8, Eq.8–17)

For a mini-batch of target values and predictions V_B, V̂_B ∈ ℝ^{B×d},
denote the i-th token (row) as V̂ᵢ, Vᵢ ∈ ℝ^{1×d}.
For each loss, the paper gives the explicit formula and computes the mixed second derivative
∂²L / (∂V_{ij} ∂V̂_{ij}).

**(1) Dot Product Loss (Eq.8–9)**

$$\mathcal{L} = -\frac{1}{B\sqrt{d}} \sum_{i=1}^{B} \hat{V}_i V_i^\top \tag{8}$$

$$\frac{\partial^2 \mathcal{L}}{\partial V_{ij}\,\partial \hat{V}_{ij}}
= \frac{\partial}{\partial V_{ij}}\!\left(-\frac{V_{ij}}{B\sqrt{d}}\right)
= -\frac{1}{B\sqrt{d}} \tag{9}$$

Constant, non-zero → outer gradient always flows. Simplest and fastest; chosen for ViT³.

**(2) MSE / L2 Loss (Eq.10–11)**

$$\mathcal{L} = \frac{1}{2B\sqrt{d}} \sum_{i=1}^{B} (\hat{V}_i - V_i)(\hat{V}_i - V_i)^\top \tag{10}$$

$$\frac{\partial^2 \mathcal{L}}{\partial V_{ij}\,\partial \hat{V}_{ij}}
= \frac{\partial}{\partial V_{ij}}\!\left(\frac{\hat{V}_{ij} - V_{ij}}{B\sqrt{d}}\right)
= -\frac{1}{B\sqrt{d}} \tag{11}$$

Same constant as Dot Product → gradient flows equally well.
Slightly higher FLOPs (subtraction + square), hence marginally slower (1296 FPS vs 1315).

**(3) RMSE Loss (Eq.12–13)**

$$\mathcal{L} = \sqrt{\frac{1}{B\sqrt{d}} \sum_{i=1}^{B} (\hat{V}_i - V_i)(\hat{V}_i - V_i)^\top} \tag{12}$$

Let S = (1/B√d) Σ ‖V̂ᵢ - Vᵢ‖².

$$\frac{\partial^2 \mathcal{L}}{\partial V_{ij}\,\partial \hat{V}_{ij}}
= -\frac{1}{B\sqrt{d}\,\sqrt{S}} + \frac{(\hat{V}_{ij} - V_{ij})^2}{B^2 d\, S^{3/2}} \tag{13}$$

Non-zero but **data-dependent**: the derivative shrinks as S grows,
making the gradient signal inconsistent. This leads to slightly lower accuracy (78.8%).

**(4) MAE / L1 Loss (Eq.14–15)**

$$\mathcal{L} = \frac{1}{B\sqrt{d}} \sum_{i=1}^{B} \|\hat{V}_i - V_i\|_1 \tag{14}$$

$$\frac{\partial^2 \mathcal{L}}{\partial V_{ij}\,\partial \hat{V}_{ij}}
= \frac{\partial}{\partial V_{ij}}\!\left(\frac{\text{sign}(\hat{V}_{ij} - V_{ij})}{B\sqrt{d}}\right)
= 0 \quad \text{(a.e.)} \tag{15}$$

The sign function is piecewise constant → its derivative vanishes almost everywhere.
W_V receives **no gradient** → catastrophic accuracy drop (76.5%), the worst among all five.

**(5) Smooth L1 Loss (Eq.16–17)**

$$\mathcal{L} = \frac{1}{B\sqrt{d}} \sum_{i=1}^{B} \sum_{j=1}^{d} \ell(\hat{V}_{ij} - V_{ij}), \quad
\ell(x) = \begin{cases} \tfrac{1}{2}x^2 & |x| < 1 \\ |x| - \tfrac{1}{2} & \text{otherwise} \end{cases} \tag{16}$$

$$\frac{\partial^2 \mathcal{L}}{\partial V_{ij}\,\partial \hat{V}_{ij}}
= -\frac{1}{B\sqrt{d}} \times \begin{cases} 1 & |\hat{V}_{ij} - V_{ij}| < 1 \\ 0 & |\hat{V}_{ij} - V_{ij}| > 1 \end{cases} \tag{17}$$

Hybrid behavior: quadratic region → non-zero (like MSE), linear region → zero (like MAE).
As training progresses and residuals shrink below 1, the loss enters its quadratic regime
and gradient flows. But in the linear region, the same problem as MAE occurs.
Accuracy: 78.1% — between MAE and MSE.

**Summary pattern**:

| Loss | ∂²L/∂V∂V̂ | Behavior | Top-1 |
|---|---|---|---|
| Dot Product | -1/(B√d) | constant ✓ | 78.9 |
| MSE | -1/(B√d) | constant ✓ | 79.2 |
| RMSE | data-dependent | non-zero but noisy | 78.8 |
| MAE | 0 (a.e.) | vanishes ✗ | 76.5 |
| Smooth L1 | piecewise | partial vanishing | 78.1 |

> The 1/√d scaling is consistent with scaled dot-product attention convention [58].
> Core takeaway: losses whose mixed 2nd derivative vanishes block outer-loop gradient to W_V.

---

### 5.2 Inner Batch Size and Epochs

#### Definition
- **Batch size B**: how many (K, V) pairs are used per inner gradient step.
  B = N means full-batch (use all tokens at once); B < N means mini-batch (sequential updates).
- **Epochs**: how many passes over the full dataset D.

#### Properties — Insight 2
> A single epoch of full-batch gradient descent works well for vision.

Key reasoning:
- Mini-batch gradient descent imposes a **causal bias** (earlier batches affect later ones).
- This is appropriate for *language* (inherently sequential), but suboptimal for *vision* (non-causal, spatial).
- Multiple epochs improve accuracy slightly but hurt throughput significantly and risk training instability.

| Epochs | Batch Size | FPS | Top-1 |
|---|---|---|---|
| 1 | N | 1315 | 78.9 |
| 1 | N/2 | 1201 | 78.6 |
| 1 | N/3 | 1131 | 78.3 |
| 2 | N | 971 | 79.1 |
| 3 | N | 787 | 79.2 |
| 4 | N | 659 | 57.0* (diverged) |

#### Application
- **ViT³ uses 1 epoch, full-batch (B = N).**
- Code: the inner training functions process *all* tokens at once — no mini-batch loop.

#### Links
- → **Causal vs. non-causal**: explains why language TTT benefits from mini-batch but vision does not
- → **Remark 2**: parallel vs. sequential linear models in vision — mini-batch training is analogous to sequential scanning (like Mamba), a promising but unexplored direction for TTT
- → **Throughput**: more epochs = more inner forward/backward passes = slower

---

### 5.3 Inner Learning Rate

#### Definition
The step size η for the inner-loop weight update:
W ← W - η · ∂L/∂W

#### Properties — Insight 3
> A relatively large inner learning rate of 1.0 is effective.

- Too small (η < 0.5): insufficient weight updates, inner model barely adapts.
- Too large (η > 5.0): training instability, outer optimization diverges.
- Dynamic per-token rate η_i = η · Sigmoid(xᵢ W_η), as used in prior work [55, 77], is less effective in vision.

| η | 0.1 | 0.2 | 0.5 | **1.0** | 2.0 | 5.0 | 10.0 | Dynamic |
|---|---|---|---|---|---|---|---|---|
| Top-1 | 77.5 | 78.1 | 78.7 | **78.9** | 78.9 | 76.7* | 76.9* | 78.7 |

#### Application
- **ViT³ uses η = 1.0, fixed.**
- In code: `lr=1.0` is the default argument in `inner_train_simplified_swiglu()` and `inner_train_3x3dwc()`.

#### Links
- → **Remark 3**: for linear inner models with MSE loss, η can be absorbed into the scaling of K and V (η·K^T(KW-V) = K̄^T(K̄W-Ṽ) where K̄=√η·K). But in practice η is still critical because rescaling interacts with normalization layers and initialization.
- → **1/√d scaling in Softmax Attention**: analogous — mathematically absorbable but practically essential.

---

## 6. Inner Model Design — Architecture Choices

### 6.1 Scaling Width

#### Properties — Insight 4
> Increasing inner model capacity consistently improves performance.

Using a two-layer MLP as the inner model, varying the hidden dimension ratio r:

| Inner Model | FLOPs | Top-1 |
|---|---|---|
| MLP, r=1 (hidden dim = d) | 4.58G | 78.9 |
| MLP, r=2 (hidden dim = 2d) | 4.92G | 79.2 |
| MLP, r=3 | 5.27G | 79.5 |
| MLP, r=4 | 5.62G | 79.6 |

This is a key advantage over Linear Attention, which is stuck with a fixed d×d linear state.

#### Links
- → **Remark 4**: an inner module costs ~4× the FLOPs of an equivalent outer module (1 forward on K + 2 backward + 1 forward on Q). So scaling inner models is expensive, making *lightweight but expressive* designs critical.

---

### 6.2 Scaling Depth

#### Properties — Insight 5
> In current TTT settings, deep inner models suffer from optimization difficulties.

| Inner Model | Top-1 |
|---|---|
| FC (1 layer, d×d) | 79.1 |
| MLP (2 layers, d→d→d) | 78.9 |
| MLP (3 layers, d→d→d→d) | 77.5 |

Deeper = more capacity in theory, but *worse performance* in practice.

Why? Two complementary problems:
- **Outer-loop problem**: W₀ for deep inner modules is harder to learn during end-to-end training.
- **Inner-loop problem**: deeper networks cause exploding/vanishing gradients in the inner loop, hindering compression of (K, V).

Evidence: the *constrained* design SiLU(FC(x)) — a two-layer MLP with identity output layer — gets **79.4%**, beating the full two-layer MLP at 78.9%. Similarly, removing the output layer from SwiGLU raises accuracy from 79.0% to **79.7%**. These constrained models are shallower in effective depth and easier to optimize.

Standard remedies (residual connections, identity initialization) provide only limited help:

**Table 6. Residual connections & initialization strategies for MLP inner model**

| Inner Model | #Params | FLOPs | FPS | Top-1 |
|---|---|---|---|---|
| SiLU(xW₁)W₂ + x (residual) | 23.5M | 4.58G | 1294 | 78.8 |
| SiLU(xW₁)(W₂ + I) (implicit residual) | 23.5M | 4.58G | 1294 | 79.1 |
| SiLU(xW₁)W₂, W₂ init as I | 23.5M | 4.58G | 1315 | 79.0 |

All three underperform the constrained design SiLU(xW₁) at **79.4%** and simplified SwiGLU at **79.7%**,
confirming that the fundamental issue is depth-related optimization difficulty, not just initialization.

#### Links
- → **Remark 5**: addressing this optimization bottleneck is a fundamental open research direction. Deep networks have exponentially more capacity [10, 15, 40, 44, 46], so enabling deep inner models could unlock major TTT gains.
- → **Simplified SwiGLU**: the constrained design that sidesteps depth optimization issues.

---

### 6.3 Convolution as Inner Model

#### Properties — Insight 6
> Convolutional architectures are particularly appropriate as inner models for visual tasks.

| Inner Model | Params | FLOPs | Top-1 |
|---|---|---|---|
| FC(x) | 23.2M | 4.34G | 79.1 |
| FC(x) ⊙ SiLU(FC(x)) | 23.5M | 4.58G | 79.7 |
| Conv 3×3 | 25.5M | 5.27G | 79.9 |
| **DWConv 3×3** | **22.9M** | **4.25G** | **80.1** |

DWConv achieves the best accuracy with the *fewest* parameters and FLOPs.

Why it works — a natural integration of global and local information:
- The inner training compresses **global** context (K, V from all spatial positions) into the convolution kernel weights.
- The convolution operation itself applies **local** spatial filtering (3×3 receptive field).
- The output thus captures both global relationships (encoded in the adapted weights) and local spatial structure (from the convolution pattern).

#### Application
- **ViT³ uses 3×3 DWConv for one head per TTT block.**
- Generalized dataset: D = {(Kᵢ^{3×3}, Vᵢ)} — each training sample uses the 3×3 local neighborhood of Kᵢ (Remark 6).

#### Links
- → **Depthwise Convolution** [28]: each channel has its own independent 3×3 kernel
- → **Simplified SwiGLU**: used for the remaining heads in the same block
- → **Per-sample adaptation**: the inner-trained kernel is different for every image in the batch

---

## 7. Simplified SwiGLU — The First Inner Module

### Definition
A gated linear unit variant where the output projection is removed (set to identity):

$$\mathcal{F}_1(x) = (x \cdot W_1) \odot \text{SiLU}(x \cdot W_2)$$

where W₁, W₂ ∈ R^{d×d}, and ⊙ is element-wise multiplication.

Compare with full SwiGLU: SwiGLU(x) = (xW₁ ⊙ SiLU(xW₂)) · W₃.
Removing W₃ (the output layer) is what makes it "simplified."

### Properties
- **Doubles the state capacity** vs. a single d×d linear layer (two weight matrices instead of one).
- **Easy to optimize**: by removing the output layer, it avoids the depth-related optimization issues from Insight 5.
- **SiLU gating**: the sigmoid-weighted gate provides non-linearity without making the backward pass unstable.
- Achieves **79.7%** — better than full SwiGLU (79.0%) and full two-layer MLP (78.9%).

### Application
- In code: `ttt_block.py:40-41` — self.w1 and self.w2 are the two weight matrices.
- Inner training: `ttt_block.py:54-88` — hand-derived forward/backward that updates w1, w2.
- Inference: `ttt_block.py:165` — `x1 = (q1 @ w1) * F.silu(q1 @ w2)`
- Used for `num_heads - 1` heads in each TTT block.

### Links
- → **SwiGLU** [50]: the full version with output layer W₃
- → **SiLU (Swish)**: activation function σ(x) · x where σ is sigmoid
- → **Insight 5**: why removing the output layer actually helps

---

## 8. 3×3 Depthwise Convolution — The Second Inner Module

### Definition
A depthwise convolution with a 3×3 kernel:

$$\mathcal{F}_2(x) = \text{DWConv}_{3\times3}(x)$$

where each of the d channels has its own independent 3×3 kernel (weight shape: [d, 1, 3, 3]).

### Properties
- **Very lightweight**: d × 9 = 9d parameters per inner model instance, vs. d² for a linear layer.
- **Spatial inductive bias**: the 3×3 kernel captures local spatial relationships, which is ideal for vision.
- **Global + local fusion**: inner training encodes global (K,V) context into the kernel; convolution applies local filtering. The result combines both.
- **Equivalent head_dim = 9**: the paper uses scale = 9^{-0.5} because the effective "attention dimension" of the 3×3 DWConv is 1×(3×3) = 9.
- **Per-sample kernels**: after inner training, each image in the batch has its own adapted kernel (w3 shape: [B×d, 1, 3, 3]).
- **Two implementations** of the backward:
  - `'conv'`: uses grouped convolution to compute the gradient
  - `'prod'`: manually iterates over the 9 kernel positions and computes dot products (slightly faster)
  - Both produce identical results (`ttt_block.py:227-237` test verifies this).

### Application
- In code: `ttt_block.py:42` — self.w3 with shape [head_dim, 1, 3, 3]
- Inner training: `ttt_block.py:90-134` — hand-derived backward for depthwise conv
- Inference: `ttt_block.py:167` — `F.conv2d(q2, w3, padding=1, groups=b*d)`
- Used for exactly **1 head** per TTT block.

### Links
- → **MobileNet** [28]: introduced depthwise separable convolutions
- → **Conditional Positional Encoding** [7]: another way to inject spatial structure into Transformers
- → **Insight 6**: why convolution works so well as an inner model for vision

---

## 9. The TTT Block — Putting It All Together

### Definition
A drop-in replacement for a standard attention block.
Within one TTT block, the input x ∈ R^{B×N×C} is processed as:

```
Input x
  │
  ▼
QKV projection (single linear layer → 6 tensors)
  │
  ├── (q1, k1, v1) for Simplified SwiGLU branch   [B, num_heads, N, d]
  │         │
  │    Inner train SwiGLU: (k1, v1) → update w1, w2
  │    Inner infer: x1 = (q1 @ w1*) ⊙ SiLU(q1 @ w2*)
  │         │
  │         ▼
  │      x1: [B, N, C]
  │
  ├── (q2, k2, v2) for DWConv branch               [B, d, H, W]
  │         │
  │    Inner train DWConv: (k2, v2) → update w3
  │    Inner infer: x2 = DWConv(q2, w3*)
  │         │
  │         ▼
  │      x2: [B, N, d]
  │
  └──→ Concatenate [x1, x2] → Linear projection → Output [B, N, C]
```

### Properties
- **Output shape = input shape**: [B, N, C] → [B, N, C], so it can replace any attention block.
- **Linear complexity**: O(N) in both time and memory.
- **Parallelizable**: unlike recurrent alternatives, the full-batch inner training processes all tokens simultaneously.
- **Multi-head**: num_heads - 1 heads use SwiGLU, 1 head uses DWConv.
- **Batch-independent**: each sample in a batch gets independently adapted weights. The unit test at `ttt_block.py:264-276` verifies this.
- **RoPE-compatible**: the forward pass optionally applies Rotary Position Embedding to q1 and k1.

### Application
- Code: `ttt_block.py:12-176` — the complete TTT class.
- Used as the core building block in both ViT³ and H-ViT³ architectures.

### Links
- → **Softmax Attention block**: what TTT block replaces
- → **Simplified SwiGLU** (§7): the first inner module
- → **3×3 DWConv** (§8): the second inner module
- → **Output projection**: `self.proj` maps [C + d] back to C

---

## 10. Hand-Derived Backward — Why and How

### Definition
Instead of using `torch.autograd.backward()` for the inner loop,
the paper manually derives closed-form gradient expressions.

### Why?
The inner loss is a **per-head, per-sample vector** with shape [B, num_heads],
not a scalar. `torch.autograd.backward` only supports scalar losses.
Computing B × num_heads separate backward passes would be extremely wasteful.

### How — SwiGLU Branch
For the simplified SwiGLU inner model with dot product loss:

```
Forward:
  z1 = K @ W1,  z2 = K @ W2
  sig = σ(z2)           (sigmoid)
  a = z1 * sig          (element-wise)
  V̂ = a                 (identity output layer)

Loss gradient (never compute loss itself, go straight to ∂L/∂V̂):
  e = -V / N * scale

Parameter gradients:
  g1 = K^T @ (e * a)
  g2 = K^T @ (e * z1 * (sig * (1 + z2 * (1 - sig))))

Gradient clipping (stability):
  g = g / (‖g‖ + 1)

Update:
  W1* = W1 - g1,  W2* = W2 - g2
```

### How — DWConv Branch
For the 3×3 depthwise convolution inner model:

```
The 'prod' implementation:
  For each of the 9 kernel positions (dy, dx):
    dot = (K_shifted * e).sum(spatial_dims)
  Stack all 9 dots → gradient tensor [B*C, 1, 3, 3]

Gradient clipping:
  g = g / (‖g‖ + 1)

Update:
  W3* = W3.repeat(B) - g
```

### Properties
- **Gradient clipping with +1**: `g / (‖g‖ + 1)` bounds the gradient norm to less than 1, providing stability without a hard threshold.
- **No loss computation**: the loss value itself is never calculated. Only the gradient ∂L/∂V̂ (called `e` in code) is needed.
- **Fully differentiable**: even though the backward is manual, PyTorch's autograd can still differentiate through these operations for the outer loop.

### Links
- → **Meta-learning**: differentiating through gradient steps is standard in MAML [17]
- → **Outer loop gradient (Eq.6)**: the manual backward must be differentiable for the outer loop to work
- → **Gradient clipping**: prevents inner-loop instability, especially important for deep inner models

---

## 11. ViT³ Architecture Family

### 11.1 ViT³ (Non-Hierarchical)

#### Definition
A DeiT-style [56] architecture where all attention blocks are replaced with TTT blocks.
Single-resolution processing with patch size 16.

| Variant | Embedding | Heads | Blocks | Params |
|---|---|---|---|---|
| ViT³-T | 192 | 6 | 12 | 6M |
| ViT³-S | 384 | 6 | 12 | 24M |
| ViT³-B | 768 | 12 | 12 | 90M |

#### Properties
- **Global average pooling** → linear classifier.
- Processes 14×14 = 196 tokens at 224² resolution.

#### Results (ImageNet-1K, Tab.7)

| Model | Type | Top-1 |
|---|---|---|
| DeiT-S | Transformer | 79.8 |
| Vim-S | Mamba | 80.3 |
| Agent-DeiT-S | Linear | 80.5 |
| **ViT³-S** | **TTT** | **81.6** |

ViT³ beats all linear-complexity alternatives and narrows the gap to DeiT.

---

### 11.2 H-ViT³ (Hierarchical)

#### Definition
A Swin-style [35] 4-stage hierarchical architecture with TTT blocks.
Feature map resolution decreases by 2× at each stage while channel dimension increases.

| Stage | Resolution | H-ViT³-T dims | Heads | Blocks |
|---|---|---|---|---|
| Stage 1 | H/4 × W/4 | 64 | 2 | 1 |
| Stage 2 | H/8 × W/8 | 128 | 4 | 3 |
| Stage 3 | H/16 × W/16 | 320 | 10 | 9 |
| Stage 4 | H/32 × W/32 | 512 | 16 | 4 |

#### Properties
- **Multi-scale features**: natural for dense prediction (detection, segmentation).
- **Conditional positional encoding** [7]: adds spatial information without explicit position embeddings.
- **Linear complexity at every stage**: processes high-resolution feature maps with global receptive field.

#### Results (ImageNet-1K, Tab.5)

| Model | Type | Params | Top-1 |
|---|---|---|---|
| VMamba-T | Mamba | 31M | 82.5 |
| MILA-T (MESA) | Linear | 25M | 83.5 |
| **H-ViT³-T** | **TTT** | **29M** | **83.5** |
| **H-ViT³-T (MESA)** | **TTT** | **29M** | **84.0** |

H-ViT³-S surpasses the *larger* SOFT-L++ and VMamba-B with roughly half the parameters.

---

### 11.3 DiT³ (Diffusion)

#### Definition
DiT [42] with Softmax attention replaced by TTT blocks,
for class-conditional image generation.

#### Architecture (Tab.13)

| | DiT³-S/8 | DiT³-S/4 | DiT³-S/2 |
|---|---|---|---|
| Backbone | Patch ↓8, B(384, 6) × 12 | Patch ↓4, B(384, 6) × 12 | Patch ↓2, B(384, 6) × 12 |

| | DiT³-B/8 | DiT³-B/4 | DiT³-B/2 |
|---|---|---|---|
| Backbone | Patch ↓8, B(768, 12) × 12 | Patch ↓4, B(768, 12) × 12 | Patch ↓2, B(768, 12) × 12 |

"Patch ↓n" = patch size n, "B(C, H)" = one building block with embedding dim C and H attention heads.
DiT³ reuses the same TTT block, just changes the host architecture from ViT/Swin to DiT.

#### Results (Tab.10)
DiT³ consistently improves FID over DiT across all patch sizes (S and B variants):

| Model | FID↓ |
|---|---|
| DiT-S/2 | 68.40 |
| **DiT³-S/2** | **62.65** |
| DiT-B/2 | 43.47 |
| **DiT³-B/2** | **39.31** |

---

## 12. Downstream Tasks

### 12.1 Object Detection (COCO, Tab.8)

#### Setup
Mask R-CNN [27] with H-ViT³ backbone, 1× and 3× training schedules.

#### Key result
Token sequences in detection are much longer than classification (high-resolution inputs, N >> d).
This is where limited linear states (Linear Attention) struggle most.
H-ViT³ with non-linear inner modules excels:

| Backbone | AP^b (1×) |
|---|---|
| VMamba-T | 47.3 |
| MILA-T | 46.9 |
| **H-ViT³-T** | **47.3** |

At base scale, H-ViT³-B reaches **50.0 AP^b**, competitive with TransNeXt-B (51.1).

---

### 12.2 Semantic Segmentation (ADE20K, Tab.9)

#### Setup
UPerNet [65] framework with H-ViT³ backbone.

#### Key result

| Backbone | mIoU |
|---|---|
| VMamba-T | 47.9 |
| SOFT-T++ | 46.5 |
| **H-ViT³-T** | **48.0** |

H-ViT³ sets a strong linear-complexity baseline, outperforming Mamba and Linear Attention,
though still behind highly optimized Transformers like TransNeXt.

---

## 13. MESA — Training Strategy

### Definition
**MESA** (Model EMA as Soft Augmentation) [14] is a training strategy that uses
an Exponential Moving Average (EMA) teacher model to generate soft labels,
acting as an additional regularizer that alleviates overfitting at little cost.

### How it works
```
Training with MESA:

  Student (model)            Teacher (EMA model)
  ┌──────────────┐          ┌──────────────────┐
  │  θ_model     │──update──▶ θ_ema = α·θ_ema  │
  │              │          │         + (1-α)·θ │
  └──────┬───────┘          └────────┬──────────┘
         │                           │
     logits_s                    logits_t
         │                           │
         ▼                           ▼
  L = (1-r)·CE(logits_s, label) + r·KL(logits_s, logits_t)
```

- **EMA decay α = 0.9996**: teacher weights are a slow-moving average of student weights
- **MESA ratio r**: interpolation between hard labels (CE) and soft labels (KL divergence)
  - r = 0: standard training (no MESA)
  - r = 1.0: fully soft targets (ViT³ default)
- **Activation epoch**: MESA turns on after 25% of training (epoch 75/300)
  - Early epochs: EMA hasn't converged → soft labels unreliable
  - After warmup: EMA is a strong teacher → soft augmentation stabilizes training

### Properties
- **Low overhead**: only one extra forward pass through the EMA model per iteration (~20% time increase)
- **Consistent improvement**: +0.5%p across all ViT³ variants
- **No extra data or parameters at inference**: EMA model is only used during training

### Application
- Results with MESA (Tab.5, ‡ marks MESA-trained models):

| Model | Without MESA | With MESA (‡) |
|---|---|---|
| H-ViT³-T | 83.5% | **84.0%** |
| H-ViT³-B | 84.9% | **85.5%** |

- Code locations:
  - EMA model creation: `main_ema.py:63` — `ModelEma(model, decay=0.9996)`
  - EMA update: `main_ema.py` — `model_ema.update(model)` after each step
  - MESA activation check: `main_ema.py:198` — enabled after epoch ≥ 25%
  - MESA ratio: `h_vittt_t_mesa.yaml:6` — `MESA_RATIO: 1.0`

### Links
- → **Knowledge Distillation**: MESA is a self-distillation variant where teacher = EMA of student
- → **H-ViT³ training config**: all training hyperparameters are in the MESA yaml
- → **Overfitting**: MESA primarily combats overfitting, especially important for smaller models

---

## 14. Efficiency Analysis

### Definition
Comparison of computational cost between ViT³ and standard ViT (DeiT) as resolution scales.

### Properties
- **Linear time**: ViT³'s throughput degrades gracefully with resolution, while DeiT's quadratic attention becomes a bottleneck.
- **Linear memory**: ViT³ avoids materializing the N×N attention matrix.
- At 1248² resolution (6,084 tokens):
  - **4.6× speedup** over DeiT-T
  - **90.3% memory reduction**

### Links
- → **O(N²) vs. O(N)**: the fundamental motivation for TTT and all linear-complexity methods
- → **Long-sequence tasks**: detection and segmentation benefit most from linear complexity

---

## 15. The Six Insights — Quick Reference Card

| # | Insight | Design Choice | Evidence |
|---|---|---|---|
| 1 | ∂²L/∂V∂V̂ must not vanish | Dot product loss | Tab.1: MAE = 76.5% vs. Dot = 78.9% |
| 2 | Full-batch, 1 epoch for vision | B = N, 1 epoch | Tab.2: B=N beats B=N/k; 4 epochs diverges |
| 3 | Large inner lr works | η = 1.0 | Tab.3: η=1.0 best, dynamic lr not helpful |
| 4 | More inner capacity = better | Use wide inner models | Tab.4: r=1 → r=4 monotonically improves |
| 5 | Depth hurts (for now) | Remove output layers | Tab.4: 3-layer MLP = 77.5% < FC = 79.1% |
| 6 | Conv is great for vision | 3×3 DWConv inner model | Tab.4: DWConv = 80.1%, best overall |

---

## 16. Open Questions and Future Directions

The paper explicitly identifies several promising research directions:

1. **Deep inner models**: How to overcome the optimization difficulties of Insight 5?
   Deeper inner models have exponentially more capacity [10, 15, 40, 44, 46].
   Solving this could be transformative for TTT.

2. **Vision-specific mini-batch strategies**: Insight 2 shows naive sequential mini-batch is suboptimal for vision.
   But carefully designed scan paths (like Mamba's spatial scanning) could unlock the benefits of mini-batch inner training.

3. **Unexplored design axes**: inner optimizer (beyond SGD), inner data augmentation, Transformer as inner model, etc. The paper acknowledges it is not exhaustive.

4. **Closing the gap to Transformers**: H-ViT³ narrows but does not fully close the gap to highly optimized models like TransNeXt and DAT++. The path forward likely combines deeper inner models with better training strategies.

---

## 17. Concept Dependency Graph

```
Softmax Attention (Eq.1-2)
  │
  ├──→ MLP interpretation ──→ "attention = inner model"
  │                                    │
  ▼                                    ▼
Linear Attention (Eq.3-4)        TTT Paradigm (Eq.5)
  │                                    │
  │  "special case:                    ├── Inner Loss (Insight 1)
  │   linear inner model"             │     └── Eq.6: mixed 2nd derivative
  │                                    │
  └────────────► ◄─────────────────────┤
                                       ├── Inner Batch/Epoch (Insight 2)
                                       │     └── causal vs. non-causal
                                       │
                                       ├── Inner LR (Insight 3)
                                       │     └── Remark 3: absorbable but critical
                                       │
                                       ├── Inner Width (Insight 4)
                                       │     └── Remark 4: 4× cost overhead
                                       │
                                       ├── Inner Depth (Insight 5)
                                       │     └── Remark 5: outer + inner loop issues
                                       │
                                       └── Conv Inner Model (Insight 6)
                                             └── Remark 6: local+global fusion
                                                        │
                                                        ▼
                                              ViT³ TTT Block
                                            ┌───────┴───────┐
                                            ▼               ▼
                                     Simplified SwiGLU   3×3 DWConv
                                     (num_heads-1)       (1 head)
                                            │               │
                                            └───────┬───────┘
                                                    ▼
                                              Concat + Proj
                                                    │
                                        ┌───────────┼───────────┐
                                        ▼           ▼           ▼
                                      ViT³       H-ViT³       DiT³
                                   (DeiT-style) (Swin-style) (DiT-style)
                                        │           │           │
                                        ▼           ▼           ▼
                                    Classify   Detect/Seg    Generate
```

---

## 18. Key Equations at a Glance

| Eq. | Name | Formula | Meaning |
|---|---|---|---|
| (1) | Softmax Attention | O = Softmax(QK^T/√d) V | Standard attention |
| (2) | MLP view | O = σ(QW₁)W₂, W₁=K^T/√d, W₂=V | Attention as width-N MLP (√d absorbed into W₁) |
| (3) | Linear Attention | Oᵢ = Qᵢ(ΣKⱼ^TV_j) / Qᵢ(ΣKⱼ^T) | Kernel trick for O(N) |
| (4) | FC view | O = QW | Linear attn as d×d linear layer |
| (5) | TTT update | W ← W - η ∂L/∂W | Inner-loop gradient step |
| (6) | Outer gradient | ∂G/∂W_V = ... ∂²L/∂V̂∂V ... | Why mixed derivative matters |
| (7) | LR absorption | η K^T(KW-V) = K̄^T(K̄W-Ṽ) | LR ↔ K,V scaling equivalence |

---

## 19. Reference Map

Grouped by topic for quick lookup:

**Vision Transformers**: ViT [13], DeiT [56], CaiT [57], Swin [35], CSwin [12], PVT [60], BiFormer [80], DAT [62,63], TransNeXt [51]

**Linear Attention**: Performer [6], CosFormer [45], FLatten [20], MILA [22], SOFT++ [38], VVT [54], GLA [70], Delta Rule [71]

**Mamba / SSM**: Mamba [18], VMamba [34], LocalVMamba [30], Vim [81], EfficientVMamba [43], MambaTree [66]

**TTT Origins**: Sun et al. [55] (ICML 2025), LaCT [77], one-minute video [9], TTT3R [5], Titans [1], Atlas [2]

**Meta-Learning**: MAML [17], meta-learning update rules [39]

**ConvNets**: ResNet [26], DenseNet [29], MobileNet [28,47], ConvNeXt [36], InternImage [61], InceptionNeXt [73]

**Training Strategy**: MESA [14], Mixup [75], CutMix [74], RandAugment [8], Random Erasing [78]

**Dense Prediction**: Mask R-CNN [27], DETR [4], UPerNet [65], SegFormer [67]

**Generation**: DiT [42]

**Theory (depth & expressivity)**: [10, 15, 40, 44, 46] — supporting Insight 5's claim that deep networks have exponential capacity
