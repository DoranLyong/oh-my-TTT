# tttLRM Study Note — A Concept Mind-Map

> Paper: tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction (arXiv:2602.20160v2)
> Authors: Chen Wang, Hao Tan, Wang Yifan, Zhiqin Chen, Yuheng Liu, Kalyan Sunkavalli, Sai Bi, Lingjie Liu, Yiwei Hu
> Affiliations: University of Pennsylvania, Adobe Research, UCI

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** — what it is, stated plainly
- **Properties** — its mathematical or behavioral characteristics
- **Application** — how the paper (or the field) uses it
- **Links** — connections to other concepts in this map

---

## 0. The Big Picture

```
               3D Reconstruction from Images
                          │
              "Need long-context, streaming,
               and explicit 3D outputs"
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  Optimization-based   Feedforward LRMs   Implicit Latent
  (3DGS, NeRF)         (GS-LRM, LRM)     (TTT-LVSM)
  - slow (minutes)     - O(N²) attention  - fast rendering ✗
  - per-scene optim    - ≤32 views        - no explicit 3D
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                ┌─────────────────────┐
                │       tttLRM        │
                │                     │
                │  TTT fast weights   │
                │  as implicit 3D     │
                │  memory (O(Nd²))    │
                │         │           │
                │    ┌────┴────┐      │
                │    ▼         ▼      │
                │  Query     Query    │
                │  → 3DGS   → NeRF   │
                └─────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      Feedforward    Autoregressive  Distributed
      (all views     (streaming      (sequence
       at once)       4 views/step)   parallelism)
```

---

## 1. Test-Time Training (TTT) and Fast Weights

### Definition

Test-Time Training (TTT) [47] replaces the fixed hidden state of sequence models with a set of **fast weights** $W$ that are updated at inference time according to the input. The fast weights encode the key-value pairs $(\mathbf{k}_i, \mathbf{v}_i)$ of input tokens as training data, using mean-square error:

$$W \leftarrow W - \eta \, \nabla_W \mathcal{L}_{\text{MSE}}\!\bigl(f_W(\mathbf{k}),\; \mathbf{v}\bigr) \tag{TTT update}$$

The updated fast weights can then be applied to queries to obtain the final output:

$$\mathbf{o} = f_W(\mathbf{q})$$

In this way, the fast weights effectively encode the key-value (KV) cache of the input sequence into a fixed-size neural memory.

### Properties

- **Fixed-size memory**: The fast weights $W$ compress the entire context into a fixed set of parameters (three matrices $W_0, W_1, W_2 \in \mathbb{R}^{d_h \times d_{\text{out}}}$), independent of sequence length.
- **Linear complexity**: Update and apply operations are $\mathcal{O}(N d^2)$ where $N$ is the number of tokens and $d$ is the hidden dimension — compared to $\mathcal{O}(N^2 d)$ for standard attention.
- **Online learning**: The fast weights can be updated incrementally as new tokens arrive, making it naturally suited for streaming/autoregressive settings.
- **Capacity limitation**: The fixed-size memory may struggle with highly complex scenes or extremely long sequences. Empirically, higher scene complexity degrades performance (indoor PSNR 24.45 vs. outdoor 24.96; high-frequency PSNR 24.20 vs. low-frequency 25.97).

### Application

In tttLRM, the fast weights serve as an **implicit 3D representation**. Multi-view image tokens are compressed into the fast weights during the update phase, forming a latent 3D memory that can be queried to produce explicit 3D outputs (Gaussian splats, triplanes).

- In code: `lact_ttt.py:246-357` — `FastWeightGluMLPMultihead` class, which defines the fast weight matrices `w0`, `w1`, `w2`.
- Fast weight update: `lact_ttt.py:156-228` — gradient computation and weight update within `fast_weight_swish_glu_weight_norm_mini_batch_apply()`.

### Links

- → **LaCT (Large Chunk Test-Time Training)**: LaCT is the specific TTT variant used in tttLRM, extending TTT to large chunk sizes (up to 1M tokens).
- → **Virtual Tokens**: Fast weights are queried via virtual tokens to extract 3D representations.
- → **Autoregressive Reconstruction**: The online-update nature of fast weights enables streaming 3D reconstruction.
- → **Selective Update**: Fisher information-based regularization prevents fast weight drift during long autoregressive sequences.

---

## 2. Large Chunk Test-Time Training (LaCT)

### Definition

Large Chunk Test-Time Training (LaCT) [70] extends the original TTT model to process large chunks of tokens (up to 1M) by computing the gradient of the summed loss over all keys and values within the chunk. Each LaCT layer includes a **window attention** module that captures local relationships within each view, followed by the fast weight update/apply mechanism.

The per-layer operations are:

$$\mathbf{T}_i = \mathbf{T}_i + \text{WinAttn}(\mathbf{T}_i) \tag{Eq.1}$$

$$W = \text{Update}(\{\mathbf{T}_i\}_{i=1}^{N}) \tag{Eq.2}$$

$$\mathbf{T}_i = \text{Apply}(W, \mathbf{T}_i) \tag{Eq.3}$$

where $\mathbf{T}_i$ are the visual tokens from view $i$, and each LaCT layer also contains feedforward (MLP) layers omitted in the equations for simplicity.

### Properties

- **Window attention**: Each head has 64-dimensional features with QK-normalization for stability. This captures local spatial relationships within a single view before cross-view information flows through the fast weights.
- **GLU-based fast weight function**: The fast weight function $f_W$ uses a SwiGLU architecture:

$$\text{gate} = \text{SiLU}(\mathbf{k} \, W_0), \quad \text{hidden} = \mathbf{k} \, W_2, \quad \text{output} = (\text{gate} \odot \text{hidden}) \, W_1 \tag{SwiGLU}$$
- **MUON optimizer** [19]: Updates use Newton-Schulz iteration to compute orthogonal gradient directions, improving stability and robustness. Five iteration steps with coefficients $a=3.4445$, $b=-4.7750$, $c=2.0315$.
- **Weight normalization**: Maintains norm invariance across updates for stable learning.
- **Linear FLOPS**: $\mathcal{O}(N d^2)$ per layer. Even a 3-layer attention model is slower than 24-layer LaCT at 2M tokens (Fig. 8).

### Application

tttLRM stacks 24 LaCT layers with hidden dimension 768 (12 heads × 64 dim/head).

- In code: `lact_ttt.py:70-104` — `zeropower_via_newtonschulz5()` implements the MUON orthogonal update.
- In code: `lact_ttt.py:107-244` — `fast_weight_swish_glu_weight_norm_mini_batch_apply()` is the core update/apply function.
- In code: `lact_ttt.py:56-67` — `silu_backprop()` for custom SiLU gradient in the GLU structure.
- In code: `block.py:86-131` — `Block` class combines SelfAttention (window attention), FastWeightGluMLP (TTT), and MLP layers.

### Links

- → **TTT and Fast Weights**: LaCT is the specific instantiation of TTT used in this work.
- → **tttLRM Architecture**: 24 LaCT layers form the backbone of tttLRM.
- → **MUON Optimizer**: Used for the fast weight gradient update, replacing standard SGD for better stability.
- → **Distributed Feedforward Reconstruction**: The linearity of LaCT fast-weight updates enables easy gradient synchronization across GPUs via DDP.

---

## 3. tttLRM Architecture

### Definition

tttLRM is a large reconstruction model that takes a set of posed images $\{\mathbf{I}_i \in \mathbb{R}^{H \times W \times 3} | i = 1, 2, \ldots, N\}$ as input and produces explicit 3D representations (3DGS or NeRF triplanes). The architecture consists of four stages:

1. **Tokenization**: Images are concatenated channel-wise with ray embeddings $\{\mathbf{R}_i \in \mathbb{R}^{H \times W \times 9} | i = 1, 2, \ldots, N\}$, divided into non-overlapping $p \times p$ patches, and projected into tokens via a lightweight linear layer:

$$\{\mathbf{T}_{i,j}\}_{i=1}^{N} {}_{j=1}^{HW/p^2} = \text{Tokenize}\big(\text{Patchify}([\{\mathbf{I}_i\}_{i=1}^{N}, \{\mathbf{R}_i\}_{i=1}^{N}])\big) \tag{tokenization}$$

2. **LaCT Backbone**: 24 LaCT layers iteratively update the fast weights $W$ and apply them to tokens (Eq.1–3).

3. **Virtual Token Querying**: A set of virtual view tokens $\mathbf{T}_i^v$ query the updated fast weights:

$$\mathbf{T}_i^v = \text{Apply}(W, \mathbf{T}_i^v) \tag{Eq.4}$$

4. **Decoding**: A linear token decoder transforms the updated query tokens into per-patch 3D parameters.

```
Input Images + Ray Embeddings
         │
    ┌────┴────┐
    │Patchify │  (p=8, channels=12: 3 RGB + 9 ray)
    │+ Linear │
    └────┬────┘
         │
    Visual Tokens T [B, N×(HW/p²), 768]
         │
    ┌────┴────┐
    │ 24×LaCT │  Window Attn → TTT Update/Apply → MLP
    │  Layers │  (fast weights updated across views)
    └────┬────┘
         │
    ┌────┴─────────────┐
    │                  │
  Input Tokens    Virtual Tokens (query)
  (discard)       Apply(W, T_v)
                       │
                  ┌────┴────┐
                  │ Decoder │  Linear → per-patch Gaussians
                  └────┬────┘
                       │
              3D Gaussian Splats / Triplanes
```

### Properties

- **Model size**: 24 layers, hidden dim 768, 12 heads, ~same parameterization as TTT-LVSM [70] except the decoding module.
- **Input channels**: 12 per pixel — 3 (RGB) + 3 (ray origin) + 3 (ray direction) + 3 (cross product of ray origin and direction).
- **Patch size**: $8 \times 8$ for the image tokenizer.
- **Output per patch**: Each patch predicts $p^2$ Gaussians. Per-Gaussian parameter count:

$$\underbrace{3}_{\text{xyz}} + \underbrace{(l+1)^2 \times 3}_{48\;\text{SH}} + \underbrace{3}_{\text{scale}} + \underbrace{4}_{\text{quat}} + \underbrace{1}_{\text{opacity}} = 59 \;\text{params} \quad (l = 3)$$
- **Resolution scalability**: Seamlessly scales to $1024\!\times\!1024$ resolution, whereas attention-based models (GS-LRM) run out of memory.

### Application

- In code: `model.py:116-414` — `tttLRM` main class.
- Tokenizer: `model.py:149-162` — patch embedding and linear projection layers.
- Decoder: `model.py:167-170` — linear decoder mapping tokens to Gaussian parameters.
- Forward pass: `model.py:248-413` — full forward including tokenization, LaCT layers, querying, decoding, and rendering.
- Config: `configs/dl3dv_full.yaml` — d: 768, n_layer: 24, patch_size: 8, sh_degree: 3.

### Links

- → **LaCT**: The 24-layer backbone that processes tokens and maintains fast weights.
- → **Virtual Tokens**: The query mechanism that extracts 3D from fast weights.
- → **3D Gaussian Splatting Decoder**: Converts decoded tokens into renderable Gaussians.
- → **Distributed Feedforward Reconstruction**: Sequence parallelism for scaling to many views.
- → **Pretraining from TTT-LVSM**: Architecture compatibility enables weight transfer.

---

## 4. Virtual Tokens and Fast Weight Querying

### Definition

Virtual tokens are a set of learnable/constructed tokens $\{\mathbf{T}_i^v \in \mathbb{R}^{H \times W \times 3} | i = 1, 2, \ldots, M\}$ that represent novel camera viewpoints. They are tokenized identically to input images (patchified and projected), but participate **only in the apply phase** — they do not update the fast weights:

$$\mathbf{T}_i^v = \text{Apply}(W, \mathbf{T}_i^v) \tag{Eq.4}$$

For 3DGS reconstruction, these virtual views are pixel-aligned viewpoints from which Gaussians are predicted. For triplane NeRFs, the virtual tokens are learnable triplane features instead of camera views.

### Properties

- **No weight update**: Virtual tokens query the fast weights without modifying them, cleanly separating the "encoding" (update) and "decoding" (apply) phases.
- **Flexible output format**: By changing what the virtual tokens represent, the same architecture produces different 3D representations:
  - Virtual views → 3DGS (pixel-aligned Gaussian prediction)
  - Triplane features → NeRF (triplane-based reconstruction)
- **Scalability**: The number of virtual views $M$ can differ from the number of input views $N$. During evaluation, the model trained with 8 input views generalizes to 16 or 24 views (Table 1, "V." column shows 10 virtual views used for 16 inputs).

### Application

- In code: `model.py:276-296` — virtual token preparation, ray computation, and concatenation with input tokens.
- In code: `model.py:307-312` — TTT config separates input tokens (update + apply) from virtual tokens (apply only).
- In code: `model.py:332-338` — after LaCT layers, virtual token outputs are decoded into Gaussian parameters (xyz, SH features, scaling, rotation, opacity).
- Gaussian position decoding: depth prediction per pixel → convert to 3D position using camera rays and known pose.
  - Object data: range function (object-centric depth)
  - Scene data: linear depth

### Links

- → **TTT and Fast Weights**: Virtual tokens read from the fast weight memory without writing to it.
- → **tttLRM Architecture**: Virtual tokens are the bridge between the LaCT backbone and the 3D output.
- → **3D Gaussian Splatting Decoder**: Virtual token outputs are decoded into per-patch Gaussians.
- → **Decoding into Other 3D Formats**: Changing virtual tokens from camera views to triplane features switches the output format.

---

## 5. 3D Gaussian Splatting Decoder

### Definition

The decoder is a lightweight linear projection that maps each updated virtual token to a set of per-patch Gaussian splat parameters. For each $p \times p$ patch, it predicts $p^2$ Gaussians, each described by:

| Parameter | Dimension | Activation |
|-----------|-----------|------------|
| Position $(\mathbf{x})$ | $3$ | Depth decode $\rightarrow$ 3D back-projection |
| SH coefficients | $(l+1)^2 \!\times\! 3 = 48$ | None (direct) |
| Scale $(\mathbf{s})$ | $3$ | $e^{\mathbf{s}}$ at render time |
| Rotation $(\mathbf{q})$ | $4$ | $\mathbf{q} / \|\mathbf{q}\|$ at render time |
| Opacity $(\alpha)$ | $1$ | $\sigma(\alpha)$ at render time |

Total: 59 parameters per Gaussian.

### Properties

- **Pixel-aligned prediction**: Each Gaussian is predicted at a pixel location in the virtual view, with its 3D position determined by decoding depth and back-projecting along the camera ray.
- **Pruning**: Gaussians with low opacity are pruned. The paper uses 70% pruning for 64-view scenes and 60% otherwise (`prune_ratio: 0.4` means keep top 40%).
- **Rendering**: Uses the `gsplat` library for differentiable rasterization with `torch.compile` for ~30% speedup. Mixed-precision training [33] with BFloat16 format. Deferred backpropagation [67] reduces GPU memory.
- **Gradient clipping**: Iterations with gradient norm > 5.0 are skipped for training stability.

### Application

- Decoder definition: `model.py:167-170` — `nn.Sequential(LayerNorm, Linear)`.
- Gaussian unpacking: `model.py:332-338` — splits decoder output into xyz, features, scaling, rotation, opacity.
- Renderer: `gaussian_renderer.py:5-74` — `GaussianRenderer` with custom autograd for forward/backward rendering via gsplat.
- Rendering call: `model.py:383-392` — `gs_renderer()` renders novel views and computes losses.

### Links

- → **Virtual Tokens**: Provides the input to the decoder after fast weight querying.
- → **Training Objective**: Rendered images from the decoder are compared against ground truth for loss computation.
- → **Decoding into Other 3D Formats**: The decoder can be swapped for triplane decoding to produce NeRFs instead.

---

## 6. Autoregressive Reconstruction

### Definition

tttLRM supports autoregressive (streaming) 3D reconstruction by processing input views in mini-batches sequentially. As described in Algorithm 1:

> **Algorithm 1: Autoregressive 3DGS Reconstruction**
>
> **Input:** Reconstructor $\mathcal{F}$ with initial fast weights $W_0$; input/query view batches $\{(\mathcal{I}_{(b)},\; \mathcal{I}^v_{(b)})\}_{b=1}^{B}$
>
> **Output:** Reconstructed GS $G$
>
> 1: $W \leftarrow W_0$
>
> 2: **for** $b = 1$ **to** $B$ **do**
>
> 3: $\quad W \leftarrow \mathcal{F}(W,\; \mathcal{I}_{(b)})$ $\qquad$ *(update fast weights)*
>
> 4: $\quad G_{(b)} \leftarrow \mathcal{F}(W,\; \mathcal{I}^v_{(b)})$ $\qquad$ *(predict Gaussians)*
>
> 5: **end for**
>
> 6: **return** $G_{(B)}$

The streaming variant incorporates **causal dependencies** among tokens: each mini-batch of input views updates the fast weights, and the model immediately predicts Gaussians for the corresponding query views.

### Properties

- **Causal structure**: Unlike the feedforward mode where all views are processed jointly, the autoregressive mode enforces temporal ordering — each batch only sees previous + current inputs.
- **Progressive improvement**: Starting from only 4 views, the reconstruction progressively improves as more views (8, 32) arrive, with both rendering quality and scene coverage increasing.
- **Predict & Merge alternative**: Instead of full reconstruction per batch, one could generate only new Gaussians and merge:

$$G_{(b')} \leftarrow \mathcal{F}\!\left(W,\; \mathcal{I}^v_{(b')}\right), \quad G_{(b)} = G_{(b-1)} \cup G_{(b')}$$

  However, this accumulates errors in $G_{(b-1)}$ and produces worse results (Table 4: PSNR 21.50 vs. 23.63).
- **Training**: AR models are finetuned from full-model checkpoints for ~3K iterations with peak LR $10^{-4}$, batch size 64, on input views 8 to 64.

### Application

- AR scanning config: `lact_ttt.py:33-48` — `ar_ttt_op()` defines sequential mini-batch updates followed by a single apply phase.
- Full scanning config: `lact_ttt.py:14-30` — `full_ttt_op()` defines batch-wise update + apply for feedforward mode.
- AR config: `configs/dl3dv_ar.yaml` — `ttt_scan: "ar"`, `sample_ar: true`, `lr: 1.0e-05`.
- TTT config selection: `model.py:307-325` — selects between `full_ttt_op` and `ar_ttt_op` based on `ttt_scan` setting.

**Table 4. Predict & Merge vs. Full Reconstruction (32 views, 1K iter finetuning)**

|                  | PSNR↑ | SSIM↑ | LPIPS↓ |
|------------------|-------|-------|--------|
| Predict & Merge  | 21.50 | 0.891 | 0.318  |
| **Ours (full)**  | **23.63** | **0.904** | **0.259** |

### Links

- → **TTT and Fast Weights**: The online-update property of fast weights naturally enables autoregressive processing.
- → **Selective Update of Fast Weights**: Fisher-based regularization prevents drift during long autoregressive sequences.
- → **tttLRM Architecture**: The same architecture supports both feedforward and autoregressive modes by switching the scan pattern.

---

## 7. Distributed Feedforward Reconstruction

### Definition

To handle large numbers of high-resolution input views efficiently, tttLRM introduces **sequence parallelism** across multiple GPUs. The tokenized input views are partitioned along the sequence dimension and assigned to separate devices:

```
GPU 1                         GPU 2
┌─────────────────────┐      ┌─────────────────────┐
│ Input Views Shard 1 │      │ Input Views Shard 2 │
│         │           │      │         │           │
│    Fast Weight      │      │    Fast Weight      │
│    (local update)   │      │    (local update)   │
│         │           │      │         │           │
│  Predict Gaussians  │      │  Predict Gaussians  │
│  for assigned views │      │  for assigned views │
└────────┬────────────┘      └────────┬────────────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
            GS all_gather
         (merge all Gaussians)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   GPU 1: Render            GPU 2: Render
   novel views 1            novel views 2
         │                     │
         └──────────┬──────────┘
                    ▼
              all_reduce
           (gradient sync)
```

### Properties

- **Linear acceleration**: Thanks to the linearity of LaCT fast-weight updates, gradients across devices can be synchronized via PyTorch DDP `all_reduce`, ensuring consistent global optimization.
- **Training pipeline**: (1) Scatter tokens → (2) Local fast weight update → (3) Synchronize fast weights → (4) Predict Gaussians per GPU → (5) Gather all Gaussians → (6) Render on each GPU → (7) AllReduce gradients.
- **Inference acceleration**: During inference, distributed reconstruction also accelerates results by partitioning virtual views across GPUs.

### Application

- Scatter-projection support: `sp_support.py:140-166` — `sp_input_broadcast_scatter()` partitions tokens across GPUs.
- Gradient-aware AllReduce: `sp_support.py:8-19` — custom autograd Function for AllReduce in backward pass.
- Gather Gaussians: `model.py:351-355` — `sp_all_gather()` merges predicted Gaussians from all ranks before rendering.
- Gather outputs: `model.py:389-392` — gather rendered images and targets after per-GPU rendering.

### Links

- → **LaCT**: The linearity of fast weight updates is what makes gradient synchronization via AllReduce possible.
- → **tttLRM Architecture**: Distributed training is an integral part of scaling tttLRM to 64+ views.
- → **Training Objective**: Each GPU computes its own rendering losses; gradients are synchronized globally.

---

## 8. Training Objective

### Definition

tttLRM is trained without explicit 3D supervision. The model renders the reconstructed Gaussians onto target views and minimizes a combination of photometric and regularization losses:

$$\mathcal{L}_{\text{RGB}} = \text{MSE}(\mathbf{I}_{\text{pred}}, \mathbf{I}_{\text{gt}}) + \lambda \, \text{Perceptual}(\mathbf{I}_{\text{pred}}, \mathbf{I}_{\text{gt}}) \tag{Eq.5}$$

$$\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{opacity}} \mathcal{L}_{\text{opacity}} \tag{Eq.6}$$

### 8.1 Loss Components

| Loss | Formula | $\lambda$ | Purpose |
|------|---------|---------|---------|
| MSE (L2) | $\lVert \mathbf{I}_{\text{pred}} - \mathbf{I}_{\text{gt}} \rVert_2^2$ | $1.0$ | Pixel-level reconstruction |
| Perceptual | $\sum_l w_l \lVert \phi_l(\mathbf{I}_{\text{pred}}) - \phi_l(\mathbf{I}_{\text{gt}}) \rVert_1$ | $0.5$ | Structural/textural similarity (VGG-19 [69]) |
| Depth | $\lVert \hat{d}_{\text{pred}} - \hat{d}_{\text{gt}} \rVert_{\text{smooth-L1}}$ | $0.01$ | Geometric regularization [71] |
| Opacity | $\frac{1}{n}\sum_{i} \sigma(\alpha_i)$ | $0.01$ | Sparse Gaussian encouragement |

### 8.2 Depth Supervision

Depth regularization aligns the predicted Gaussian depth (z-axis in camera frame) with pseudo ground truth from the monocular depth estimator DepthAnythingV2 [53]. Feedforward MVS methods like VGGT [52] provide less detailed depth prediction, though they are multi-view consistent. The comparison is done on normalized disparity:

$$\hat{d} = \frac{d - \text{median}(d)}{\text{Var}(d) + \epsilon} \tag{depth norm}$$

### 8.3 Training Sampling

- **Feedforward**: Randomly sample unordered input-target image pairs.
- **Autoregressive**: Sample ordered input sequences to simulate streaming.
- **Mixed-length training**: `sample_mixed_length: true` in config allows variable numbers of input views during training.

### Application

- Loss computation: `model.py:195-246` — `compute_loss()` method.
- L2 loss: `model.py:200` — `F.mse_loss(rendering, target)`.
- Perceptual loss: `loss.py:107-193` — `PerceptualLoss` class using VGG-19 with layer weights [1, 0.38, 0.21, 0.18, 0.07].
- Depth loss: `model.py:214-228` — smooth L1 on normalized disparities.
- Opacity loss: `model.py:210-211` — `opacity.sigmoid().mean()`.
- Loss weights: `configs/dl3dv_full.yaml` — l2: 1.0, perceptual: 0.5, opacity: 0.01, depth: 0.01.

### Links

- → **3D Gaussian Splatting Decoder**: The decoder produces the Gaussians that are rendered for loss computation.
- → **Pretraining from TTT-LVSM**: The rendering loss is the same used in TTT-LVSM pretraining, enabling smooth transfer.
- → **Distributed Feedforward Reconstruction**: Each GPU computes its own portion of the rendering loss.
- → **MUON Optimizer**: Depth and opacity losses complement MUON's orthogonal gradient updates (Table 5).

---

## 9. Pretraining from TTT-LVSM

### Definition

tttLRM shares the same parameterization as TTT-LVSM [70] except for the decoding module. This allows the model to be initialized from TTT-LVSM pretrained weights — a model trained on novel view synthesis (NVS) tasks using implicit latent-space representations.

### Properties

- **Faster convergence**: Pretrained initialization accelerates convergence substantially, especially in early training stages where models quickly reach high PSNR compared to training from scratch (Fig. 7).
- **Higher final quality**: The gains persist after full training. Pretrained models achieve higher final PSNR for both GS and triplane representations (Table 3).
- **Cross-format transfer**: Pretrained knowledge transfers even when adapting to different 3D formats (GS → triplane), suggesting that the NVS pretraining provides an effective inductive bias for 3D reconstruction generally.

**Table 3. Pretraining Effect on Final 3D Reconstruction Quality (8 views, 256×256)**

| 3D Rep. | Type | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|------|-------|-------|--------|
| GS | w/o Pretrain | 32.77 | 0.026 | 0.969 |
| GS | **w Pretrain** | **33.14** | **0.024** | **0.972** |
| Triplane | w/o Pretrain | 26.40 | 0.903 | 0.093 |
| Triplane | **w Pretrain** | **27.87** | **0.925** | **0.075** |

### Application

- Architecture compatibility: `model.py:116-170` — the tttLRM class matches TTT-LVSM parameterization (24 LaCT layers, dim 768) except for the final decoder.
- The pretrained checkpoint enables curriculum training: fast low-res pretraining → high-res finetuning (Appendix B.1).

### Links

- → **tttLRM Architecture**: Architectural compatibility with TTT-LVSM is by design.
- → **LaCT**: The shared LaCT backbone is what enables weight transfer.
- → **Training Objective**: The rendering loss bridges implicit NVS pretraining and explicit 3D reconstruction.
- → **Decoding into Other 3D Formats**: Pretrained weights improve both GS and triplane outputs (Table 3).

---

## 10. Selective Update of Fast Weights

### Definition

During autoregressive reconstruction, updating fast weights only according to current inputs can cause **weight drift** — earlier information is gradually forgotten as more tokens are processed. Inspired by [32], tttLRM explores a training-free selective update mechanism based on **Fisher information** to prevent this drift.

The approach has three components:

1. **Fisher information approximation**: The diagonal of the Fisher information matrix is estimated using an exponential moving average (EMA) of squared gradients:

$$\mathbf{F} \leftarrow \alpha \, \mathbf{F} + (1 - \alpha) \, \lvert \nabla_\theta \mathcal{L} \rvert^2 \tag{Fisher EMA}$$

2. **EMA anchor**: A sliding anchor $\theta^*$ tracks the historical trajectory of fast weights:

$$\theta^* \leftarrow \beta \, \theta^* + (1 - \beta) \, \theta \tag{anchor EMA}$$

3. **Elastic regularization**: After each gradient update, parameters with low Fisher values (unimportant for the current input) are pulled back toward the anchor:

$$\theta \leftarrow \theta - \lambda \, (1 - \hat{\mathbf{F}}) \, (\theta - \theta^*) \tag{elastic reg.}$$

  where $\hat{\mathbf{F}} = \mathbf{F} / \max(\mathbf{F})$ is the normalized Fisher importance.

   Parameters with high Fisher values (important for current input) are left unconstrained.

### Properties

- **Training-free**: This mechanism requires no additional training — it is applied purely at inference time.
- **Selective adaptation**: Encourages adaptation to the current input on important parameters while suppressing drift on unimportant ones.
- **Measurable improvement**: Table 6 shows consistent gains across all metrics.

**Table 6. Selective Update Results (Autoregressive Mode)**

|                | PSNR↑ | SSIM↑ | LPIPS↓ |
|----------------|-------|-------|--------|
| w/o selective  | 24.81 | 0.814 | 0.225  |
| **w selective**| **24.95** | **0.818** | **0.223** |

### Application

- In code: `lact_ttt.py:189-220` — Anti-Fisher regularization within `fast_weight_swish_glu_weight_norm_mini_batch_apply()`.
- Fisher EMA computation: `lact_ttt.py:189-200` — exponential moving average of squared gradients.
- Anchor EMA: `lact_ttt.py:205-210` — sliding anchor weight update.
- Elastic regularization: `lact_ttt.py:212-220` — pull-back toward anchor weighted by inverse Fisher importance.

### Links

- → **TTT and Fast Weights**: Addresses the capacity limitation of fixed-size fast weight memory during long sequences.
- → **Autoregressive Reconstruction**: Specifically designed to mitigate drift in the streaming setting.
- → **LaCT**: Applied within the LaCT fast weight update loop.

---

## 11. Decoding into Other 3D Formats

### Definition

tttLRM's architecture is format-agnostic: by changing what the virtual tokens represent, the same model can produce different 3D output formats:

- **3DGS (default)**: Virtual tokens are pixel-aligned camera viewpoints. Each virtual token is decoded into per-patch Gaussians with position, SH coefficients, scale, rotation, and opacity.
- **Triplane NeRF**: Virtual tokens are replaced with learnable triplane features. The fast weights are queried as a triplane representation for NeRF-based reconstruction. The model is finetuned with a rendering loss to enable this.

### Properties

- **Shared backbone**: Both formats use the same 24-layer LaCT backbone and fast weight memory. Only the virtual tokens and decoder head differ.
- **Triplane results**: At $512 \times 512$ resolution with 4 input views, the triplane variant achieves PSNR 27.87 with pretraining (Table 3).
- **Flexibility**: This demonstrates that the fast weights encode a general implicit 3D representation, not one specific to any output format.

### Application

- In code: `model.py:332-338` — Gaussian parameter unpacking from decoder output.
- Fig. 6 shows triplane decoding results: depth maps and novel view renderings from the queried triplanes.

### Links

- → **Virtual Tokens**: The format of virtual tokens determines the output 3D representation.
- → **3D Gaussian Splatting Decoder**: The default decoder for GS output.
- → **Pretraining from TTT-LVSM**: Pretrained weights improve both GS and triplane outputs.

---

## 12. MUON Optimizer for Fast Weight Updates

### Definition

The MUON (Momentum with Unitary ONormalization) optimizer [19] is used for fast weight gradient updates within LaCT layers. Instead of standard SGD, MUON converts the gradient to an orthogonal direction using Newton-Schulz iteration, improving the conditioning of the update.

### Properties

- **Newton-Schulz iteration**: Computes the matrix square root inverse via 5 iterative steps:

$$X_{k+1} = a \, X_k + b \, X_k^{\,3} + c \, X_k^{\,5}, \quad (a,\, b,\, c) = (3.4445,\; -4.7750,\; 2.0315) \tag{NS-5}$$
- **Orthogonal updates**: The resulting gradient direction lies on the Stiefel manifold, providing better optimization landscape traversal.
- **Adaptive learning rates**: Each head has its own learnable learning rate, bounded by softplus activation.

**Table 5. MUON Optimizer and Loss Ablation (32 views, 256×144)**

| Muon | Opacity+Depth | PSNR↑ | SSIM↑ | LPIPS↓ | Opacity>0.001 |
|------|---------------|-------|-------|--------|---------------|
| ✗    | ✗             | 20.44 | 0.649 | 0.295  | 96%           |
| ✓    | ✗             | 20.68 | 0.661 | 0.290  | 97%           |
| **✓**| **✓**         | **20.76** | **0.666** | **0.285** | **47%** |

- MUON alone provides +0.24 dB PSNR improvement.
- Adding depth and opacity regularization further improves quality and dramatically reduces opaque Gaussians (97% → 47%).

### Application

- In code: `lact_ttt.py:70-104` — `zeropower_via_newtonschulz5()` function.
- Called during fast weight update: `lact_ttt.py:179-181`.
- Learning rate bounding: `lact_ttt.py:315-320` — softplus activation on per-head learning rates.

### Links

- → **LaCT**: MUON is the optimizer used within LaCT's fast weight update mechanism.
- → **Selective Update**: Fisher information-based regularization is applied after the MUON update step.
- → **Training Objective**: The depth and opacity losses complement MUON's optimization (Table 5).

---

## 13. Datasets and Evaluation Setup

### 13.1 Object-Level Dataset

**Training**: Objaverse [10] — 730K 3D objects, each centered and normalized to fit within $[-1, 1]$. 32 views per object rendered at $512 \times 512$ with cameras randomly distributed at distances uniformly sampled from $[1.5, 2.8]$ under uniform lighting.

**Evaluation**: Google Scanned Objects (GSO) — 100 objects. A few views are selected as input, 8 random views for testing. Following prior works [16, 68].

### 13.2 Scene-Level Dataset

**Training**: DL3DV-10K [27] — 10,510 high-resolution videos, each containing up to 500 keyframes with camera pose annotation from COLMAP [37].

**Evaluation**:
- **DL3DV-140**: 140 test scenes from DL3DV. Testing views are evenly selected from every 8 frames (~40 images per scene). Input views are selected based on K-means clustering of camera positions and view directions. Same split as Long-LRM [71].
- **Tanks&Temples** [23]: Additional scene-level benchmark for generalization evaluation.

### 13.3 Metrics

Three metrics for novel view synthesis quality:
- **PSNR** (Peak Signal-to-Noise Ratio): pixel-level reconstruction accuracy
- **SSIM** (Structural Similarity Index): structural similarity
- **LPIPS** [69] (Learned Perceptual Image Patch Similarity): perceptual quality using deep features

### Links

- → **Training Objective**: Datasets provide the training supervision and evaluation benchmarks.
- → **Experimental Results**: All tables report results on these datasets.

---

## 14. Training Strategy and Curriculum

### 14.1 Scene-Level Curriculum Training

A curriculum strategy that progresses from low to high resolution, motivated by: (1) fast low-res pretraining enables training with large batch sizes, and (2) pretrained TTT-LVSM checkpoints cannot predict reasonable Gaussians at high resolution initially, causing excessive GPU memory from rendering many Gaussians.

**Stage 1** — $144 \times 256$: LR $3 \times 10^{-4}$ with 2K warmup, cosine decay. AdamW optimizer with betas $(0.9, 0.95)$ and weight decay $0.05$. Batch size 128. 80K steps (~0.3T tokens).

**Stage 2** — $288 \times 512$: LR $5 \times 10^{-5}$. Batch size 64. 6K steps. Enable depth and opacity loss.

**Stage 3** — $540 \times 960$: 32 input views, LR $1 \times 10^{-5}$. Batch size 64. 5K steps. Prune 70% of Gaussians for 64 views, 60% otherwise.

**Final stage**: Train with 16 to 64 input views for another 1K steps.

**AR finetuning**: Finetune on final stage checkpoints for ~3K iterations, peak LR $10^{-4}$, batch size 64, input views 8 to 64.

Training data sampling: for each stage, a continuous range of 128–512 frames is selected from each video. 124 frames are randomly sampled, from which input and target views are further sampled with overlap for stable training.

### 14.2 Object-Level Training

**GS model** (8 input views, 8 supervision views, patch $8 \times 8$):
- $256 \times 256$: batch 512, LR $4 \times 10^{-4}$, 80K iterations
- $512 \times 512$: batch 128, LR $5 \times 10^{-5}$, 10K iterations
- $1024 \times 1024$: batch 64, LR $5 \times 10^{-5}$, 4K iterations

**Triplane model** (4 input views, 4 supervision views, patch $16 \times 16$):
- $256 \times 256$: batch 256, 60K iterations
- $512 \times 512$: batch 64, 20K iterations

Data sampling: 15 images (from 32 renderings) form a data point, from which 8 input and 8 supervision views are randomly selected independently. This encourages more overlap between input and rendering views.

### 14.3 Implementation Details

- **Hardware**: 64 Nvidia A100 80GB GPUs
- **Gaussian rendering**: `gsplat` Python library
- **Speedup**: `torch.compile` for ~30% per-iteration speedup
- **Memory optimization**: Gradient checkpointing [6] and deferred backpropagation [67]
- **Precision**: Mixed-precision training [33] with BFloat16 format
- **Stability**: Iterations with gradient norm > 5.0 are skipped

### Links

- → **Pretraining from TTT-LVSM**: Pretrained checkpoints serve as the starting point for curriculum training.
- → **Training Objective**: Loss components are progressively enabled across stages.
- → **3D Gaussian Splatting Decoder**: Pruning ratio varies by stage and view count.

---

## 15. Image-to-3D Generation

### Definition

tttLRM can be combined with a **multi-view diffusion model** to achieve single-image-to-3D generation. Given one input image, the multi-view generator produces multiple posed views, which tttLRM then reconstructs into high-quality 3DGS (Fig. 5).

### Properties

- **High resolution**: Produces $1024 \times 1024$ 3DGS reconstructions — a resolution where GS-LRM encounters out-of-memory issues.
- **Fine-grained details**: Enables reconstruction of hair, fur, text, and other photorealistic details from a single image.
- **Generality**: Works across diverse subjects — humans, animals, clothing, objects (Fig. 5 shows Mona Lisa, a skirt, paintbrushes, and other examples).

### Links

- → **tttLRM Architecture**: The linear complexity enables processing high-resolution multi-view outputs.
- → **3D Gaussian Splatting Decoder**: The generated Gaussians can be rendered in real time.
- → **Virtual Tokens**: Virtual views are constructed from the generated multi-view images.

---

## 16. Scaling to 128+ Views

### Definition

With distributed training, tttLRM can scale to hundreds of input views. By finetuning the full model with more iterations on 128 input views (more than 1M tokens), the model achieves **26.80 PSNR** on DL3DV-140.

### Properties

- **Million-token regime**: 128 views at scene-level resolution produce over 1M tokens — well beyond what attention-based models can handle.
- **Linear scaling**: The $\mathcal{O}(N d^2)$ complexity of LaCT means computation grows linearly with view count, not quadratically.
- **Attention comparison** (Fig. 8): At 2M tokens, even a 3-layer attention model is slower than 24 LaCT layers. At 8M tokens, the gap widens dramatically (attention: ~550s vs. LaCT: ~100s).

### Links

- → **LaCT**: Linear complexity is what makes million-token processing feasible.
- → **Distributed Feedforward Reconstruction**: Sequence parallelism enables distributing 128+ views across GPUs.

---

## 17. Figure Descriptions

### Fig. 1 — Teaser
Three capabilities of tttLRM: (1) high-resolution $1024\text{px}$ single-image-to-3D via multi-view generator, producing Gaussians and novel views; (2) feedforward 3DGS reconstruction from multiple scene views; (3) autoregressive reconstruction showing progressive improvement from 4 → 8 → 32 views.

### Fig. 4 — Qualitative Comparison
Visual comparison on DL3DV scenes (5 examples). tttLRM reconstructs sharper geometry and more detailed textures with fewer artifacts than optimization-based methods (3DGS, Mip-Splatting, Scaffold-GS) and the feedforward baseline (Long-LRM). Red boxes highlight regions where tttLRM produces sharper details — cleaner edges on vehicles, finer texture on surfaces. PSNR values annotated on each rendering confirm the quantitative advantage.

### Fig. 5 — Image-to-3D Generation
Four examples of single-image-to-3D using tttLRM with a multi-view generator at $1024 \times 1024$. Subjects include the Mona Lisa, a skirt, paintbrushes, and another object. Each example shows: input image → 3DGS novel view rendering, demonstrating fine-grained reconstruction of hair, fur, text, and fabric details.

### Fig. 7 — Pretraining Convergence
PSNR vs. training step plot comparing "w/ Pretrain" (blue) vs. "w/o Pretrain" (orange). The pretrained model reaches high PSNR quickly in early training (within ~2000 steps) and maintains a gap throughout training (~10K steps), converging to a higher final value. Demonstrates that NVS pretraining accelerates learning and improves final quality.

### Fig. 8 — Time Comparison: Attention vs. LaCT
Wall-clock time (seconds) vs. number of tokens (0.5M to 8M). Red line (3-layer attention with token merging) grows super-linearly; blue line (24-layer LaCT) grows linearly. Crossover occurs around 2M tokens. At 8M tokens: attention ~550s, LaCT ~100s — a 5.5× speedup that grows with sequence length.

---

## 18. What Existed Before and What This Paper Changes

### 18.1 Prior Approaches and Their Limitations

**Optimization-based methods** (3DGS [21], Mip-Splatting [65], Scaffold-GS [31]) optimize 3D representations per scene from scratch. They achieve high visual quality but require minutes of per-scene optimization — 13–16 minutes per scene for 3DGS variants. They cannot operate in a feedforward manner and do not generalize across scenes.

**Feedforward Large Reconstruction Models** take multi-view images and directly predict 3D representations. LRM [16] uses a transformer-based architecture with triplane output. GS-LRM [68] extends this to predict pixel-aligned 3DGS but is limited to very few input views (e.g., 4–8) due to the quadratic complexity of attention layers. At $512 \times 512$ with 16+ views, GS-LRM encounters out-of-memory issues.

**Long-LRM** [71] extends the input capacity to 32 views using bidirectional attention layers, but its $\mathcal{O}(N^2)$ attention complexity hinders scalability and prevents efficient processing of streamed context. It trains a separate model for each input view count and requires multiple GPUs for inference.

**Implicit latent-space models** such as TTT-LVSM [70] use TTT fast weights for novel view synthesis with high quality. However, they produce implicit representations that require repetitive network inference to render each novel view — they lack the speed and controllability of explicit representations like 3DGS.

**Mamba-based models** (Gamba [41], MVGamba [62]) apply state space models to reduce attention complexity but remain limited to very few input views.

### 18.2 What This Paper Contributes

**Contribution 1**: tttLRM is the first large reconstruction model that uses TTT for both feedforward long-context and autoregressive 3D modeling with linear complexity. It fills the gap between implicit latent models (high quality, slow rendering) and explicit feedforward models (fast rendering, limited context).

**Contribution 2**: A unified 3D modeling framework that interprets TTT fast weights as implicit 3D memory that can be queried to produce controllable explicit representations (3DGS, NeRF triplanes). This decouples the encoding (fast weight update) from the output format (virtual token design).

**Contribution 3**: State-of-the-art results on both object-level (GSO) and scene-level (DL3DV-140, Tanks&Temples) datasets, with superior quality and efficiency. At $512 \times 512$ with 16 views, tttLRM achieves 34.67 PSNR vs. GS-LRM's 33.55, while being twice as fast.

### 18.3 Side-by-Side: Prior Art vs. This Paper

| Dimension | 3DGS (optim.) | GS-LRM [68] | Long-LRM [71] | tttLRM |
|-----------|---------------|-------------|---------------|--------|
| Complexity | per-scene optim | $\mathcal{O}(N^2\!d)$ | $\mathcal{O}(N^2\!d)$ | $\mathcal{O}(Nd^2)$ |
| Max views | unlimited | ~8 | 32 | 64+ (scalable) |
| Speed (32 views) | ~13 min | N/A | 1s–12s | 7.2s |
| Autoregressive | No | No | No | **Yes** |
| Output format | 3DGS | 3DGS | 3DGS | 3DGS / NeRF |
| Single model all views | N/A | No | No | **Yes** |
| $1024^2$ resolution | Yes | OOM | OOM | **Yes** |

| Dimension | Long-LRM [71] (closest) | tttLRM |
|-----------|------------------------|--------|
| Attention complexity | $\mathcal{O}(N^2\!d)$ quadratic | $\mathcal{O}(Nd^2)$ linear |
| Max input views (single model) | 32 (separate model per count) | 64+ (one model) |
| Streaming/AR support | No | Yes |
| GPU scaling | Multi-GPU required | Linear multi-GPU acceleration |
| DL3DV-140 PSNR (32 views) | 24.10–24.99 | **25.07** |
| Tanks&Temples PSNR (32 views) | 18.38–18.69 | **19.22** |

### 18.4 The Core Shift in Thinking

Previous feedforward 3D reconstruction methods treat the problem as a direct mapping: images in, 3D out, mediated by attention layers. This forces a trade-off between context length (more views = better reconstruction) and computational cost (attention is quadratic). Long-LRM pushed this boundary by using more attention layers and view count-specific models, but the fundamental quadratic bottleneck remained.

tttLRM reframes the problem. Instead of using attention to relate all input views, it uses TTT fast weights as a **compressed implicit 3D memory** that grows in information without growing in size. The memory is a fixed set of weight matrices — no matter how many views are processed, the memory footprint stays the same. This is the same insight behind the original TTT work for language modeling, now applied to 3D: the KV cache (which grows linearly with sequence length in attention) is replaced by fast weights (which stay fixed in size).

The second shift is the **separation of encoding and decoding**. By introducing virtual tokens that query the fast weights without modifying them, tttLRM decouples what the model memorizes (input images → fast weights) from what it outputs (fast weights → 3DGS or NeRF). This means one trained model can produce different 3D formats, and the autoregressive mode falls out naturally from the online-update property of TTT.

---

## 19. Experimental Results

### 19.1 Object-Level Results (GSO Dataset)

**Table 1. tttLRM vs. GS-LRM on GSO (various resolutions and view counts)**

| Method | Resolution | Views | Time (s)↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-----------|-------|-----------|-------|-------|--------|
| GS-LRM [68] | 256×256 | 8 | 0.1 | 31.55 | 0.964 | 0.028 |
| **Ours** | 256×256 | 8 | 0.1 | **33.14** | **0.972** | **0.024** |
| GS-LRM [68] | 512×512 | 8 | 0.7 | 32.83 | 0.969 | 0.029 |
| **Ours** | 512×512 | 8 | 0.3 | **34.02** | **0.974** | **0.025** |
| GS-LRM [68] | 512×512 | 16 | 2.5 | 33.55 | 0.976 | 0.023 |
| **Ours** | 512×512 | 16 (10 V.) | 0.8 | **34.67** | **0.978** | **0.022** |
| GS-LRM [68] | 512×512 | 24 | 5.5 | 33.26 | 0.976 | 0.022 |
| **Ours** | 512×512 | 24 (10 V.) | 1.1 | **34.80** | **0.979** | **0.022** |

tttLRM consistently outperforms GS-LRM in both inference speed and reconstruction quality. It generalizes to 16 and 24 views when trained on 8, using only 10 virtual views.

### 19.2 Scene-Level Results (DL3DV-140 and Tanks&Temples)

**Table 2. Scene-Level Quantitative Comparison**

| Views | Method | Time↓ | DL3DV-140 ||| Tanks&Temples |||
|-------|--------|-------|-------|-------|--------|-------|-------|--------|
| | | | PSNR↑ | SSIM↑ | LPIPS↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
| 16 | 3DGS₃₀ₖ | 13m | 21.20 | 0.708 | 0.264 | 16.76 | 0.598 | 0.334 |
| 16 | Mip-Splatting₃₀ₖ | 13m | 20.65 | 0.712 | 0.274 | 16.82 | 0.616 | 0.372 |
| 16 | Scaffold-GS₃₀ₖ | 16m | 22.13 | 0.738 | 0.250 | 17.02 | 0.634 | 0.321 |
| 16 | Long-LRM (16v) | 0.4s | 22.66 | 0.740 | 0.292 | 17.51 | 0.555 | 0.408 |
| 16 | **Ours (single model)** | 3.6s | **23.60** | **0.784** | **0.255** | **18.15** | **0.613** | **0.360** |
| 32 | 3DGS₃₀ₖ | 13m | 23.60 | 0.779 | 0.213 | 18.10 | 0.688 | 0.269 |
| 32 | Long-LRM (32v) | 1s | 24.10 | 0.783 | 0.254 | 18.38 | 0.601 | 0.363 |
| 32 | Long-LRM (32v w/ optim) | 12s | 24.99 | 0.809 | 0.243 | 18.69 | 0.623 | 0.360 |
| 32 | Ours (single model, AR) | 7.5s | 24.31 | 0.803 | 0.237 | 18.96 | 0.653 | 0.322 |
| 32 | **Ours (single model)** | 7.2s | **25.07** | **0.822** | **0.215** | **19.22** | **0.662** | **0.305** |
| 64 | 3DGS₃₀ₖ | 13m | 26.55 | 0.852 | 0.164 | 20.78 | 0.778 | 0.205 |
| 64 | Scaffold-GS₃₀ₖ | 16m | 27.07 | 0.857 | 0.175 | 20.96 | 0.768 | 0.240 |
| 64 | Long-LRM (64v) | 3.7s | 24.63 | 0.799 | 0.243 | 19.11 | 0.627 | 0.346 |
| 64 | Ours (single model, AR) | 15.2s | 24.81 | 0.814 | 0.225 | 19.80 | 0.675 | 0.308 |
| 64 | **Ours (single model)** | 14.8s | **25.95** | **0.844** | **0.195** | **20.31** | **0.700** | **0.274** |

Key observations:
- tttLRM outperforms Long-LRM by ~1 dB PSNR across all view counts.
- One single tttLRM model works across 16, 32, and 64 views.
- The AR variant remains competitive with feedforward, confirming the streaming capability.
- With 64 views, tttLRM approaches optimization-based methods while being hundreds of times faster.

### 19.3 Post-Optimization Results (Supplementary Table 7)

**Table 7. With Post-Optimization (32 and 64 views)**

| Views | Method | Time↓ | DL3DV PSNR↑ | DL3DV SSIM↑ | DL3DV LPIPS↓ | T&T PSNR↑ | T&T SSIM↑ | T&T LPIPS↓ |
|-------|--------|-------|-------------|-------------|--------------|-----------|-----------|------------|
| 32 | 3DGS₃₀ₖ | 13m | 23.60 | 0.779 | 0.213 | 18.10 | 0.688 | 0.269 |
| 32 | Mip-Splatting₃₀ₖ | 13m | 23.32 | 0.784 | 0.217 | 18.39 | 0.700 | 0.262 |
| 32 | Scaffold-GS₃₀ₖ | 16m | 24.77 | 0.805 | 0.205 | 18.41 | 0.691 | 0.290 |
| 32 | Long-LRM (w/ 3-step) | 12s | 24.99 | 0.809 | 0.243 | 18.69 | 0.623 | 0.360 |
| 32 | Long-LRM (w/ 10-step) | 37s | 25.60 | 0.826 | 0.215 | 18.90 | 0.642 | 0.350 |
| 32 | **Ours** | **7.2s** | **25.07** | **0.822** | **0.215** | **19.22** | **0.662** | **0.305** |
| 32 | Ours (w/ 3-step) | 18s | 25.86 | 0.842 | 0.208 | 19.57 | 0.687 | 0.300 |
| 32 | **Ours (w/ 10-step)** | **42s** | **26.37** | **0.854** | **0.201** | **19.78** | **0.704** | **0.291** |
| 64 | 3DGS₃₀ₖ | 13m | 26.55 | 0.852 | 0.164 | 20.78 | 0.778 | 0.205 |
| 64 | Mip-Splatting₃₀ₖ | 13m | 26.29 | 0.850 | 0.166 | 20.08 | 0.759 | 0.220 |
| 64 | Scaffold-GS₃₀ₖ | 16m | 27.07 | 0.857 | 0.175 | 20.96 | 0.768 | 0.240 |
| 64 | Long-LRM (w/ 3-step) | 38.9s | 25.74 | 0.833 | 0.225 | 19.69 | 0.659 | 0.333 |
| 64 | Long-LRM (w/ 10-step) | 114s | 26.72 | 0.852 | 0.212 | 20.03 | 0.681 | 0.320 |
| 64 | **Ours** | **14.8s** | **25.95** | **0.844** | **0.195** | **20.31** | **0.700** | **0.274** |
| 64 | Ours (w/ 3-step) | 47s | 26.97 | 0.866 | 0.185 | 20.76 | 0.724 | 0.269 |
| 64 | **Ours (w/ 10-step)** | **124s** | **27.65** | **0.880** | **0.177** | **21.07** | **0.743** | **0.260** |

Long-LRM's quality with 3-step post-optimization is still lower than tttLRM *without* post-optimization. With the same post-optimization setup, tttLRM surpasses both purely optimization-based methods and Long-LRM.

---

## 20. Quick Reference Card

| # | Finding | Design Choice | Evidence |
|---|---------|---------------|----------|
| 1 | TTT fast weights as implicit 3D memory | LaCT layers update/query fast weights | Eq.1–4; outperforms attention-based GS-LRM |
| 2 | Virtual tokens decouple encoding/decoding | Separate token sets for update vs. query | Eq.4; same model produces 3DGS and NeRF |
| 3 | One model handles all view counts | Single model trained with mixed-length | Table 1–2; 8-view model generalizes to 24 |
| 4 | Pretraining from NVS transfers to explicit 3D | Initialize from TTT-LVSM | Table 3; +0.37 dB PSNR for GS, +1.47 dB for triplane |
| 5 | Full reconstruction > Predict & Merge | Regenerate all Gaussians each step | Table 4; 23.63 vs. 21.50 PSNR |
| 6 | MUON + depth/opacity regularization | Orthogonal gradient updates + regularization | Table 5; PSNR 20.76 vs. 20.44; 47% vs. 96% opaque |
| 7 | Fisher-based selective update | Elastic regularization on unimportant params | Table 6; +0.14 dB PSNR, training-free |
| 8 | Linear complexity scales to 1M+ tokens | LaCT vs. attention | Fig. 8; 24 LaCT layers faster than 3 attention at 2M tokens |
| 9 | Distributed sequence parallelism | Scatter tokens → local update → AllReduce | Fig. 3; linear acceleration with more GPUs |
| 10 | Post-optimization further improves quality | Few-step optimization on tttLRM output | Table 7; surpasses optim-based methods |

---

## 21. Open Questions

1. **Better memory mechanisms**: The fixed-size fast weight memory limits capacity for highly complex scenes. Future work might design adaptive or hierarchical memory structures.
2. **Real-time streaming**: Current autoregressive reconstruction is not real-time. Speeding up inference to enable real-time high-quality reconstruction from streaming inputs remains an open goal.
3. **Implicit vs. explicit trade-off**: Compared to the pretrained LVSM model, tttLRM's quality is slightly degraded but has much faster rendering speed and explicit 3D representations. Closing this quality gap while maintaining explicit outputs is desirable.
4. **Incorporating selective update into training**: The Fisher-based selective update is currently training-free (inference only). Incorporating it into training could yield further improvements.
5. **Scaling to hundreds of views**: With distributed training, tttLRM achieves 26.80 PSNR with 128 input views (1M+ tokens) by finetuning with more iterations. The upper limit of scaling is unexplored.

---

## 22. Concept Dependency Graph

```
  TTT & Fast Weights [§1]
        │
        ▼
  LaCT (Large Chunk TTT) [§2] ◄──── MUON Optimizer [§12]
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  tttLRM Architecture [§3]         Selective Update [§10]
        │                          (Fisher regularization)
        │                                  │
        ├──────────┬───────────┐            │
        ▼          ▼           ▼            │
  Virtual       Distributed  Training      │
  Tokens [§4]  Feedforward   Objective     │
        │      Recon. [§7]   [§8]          │
        │          │           │            │
        ├──────────┤           │            │
        ▼          ▼           ▼            │
  3DGS         Other 3D    Pretraining     │
  Decoder [§5] Formats     from            │
        │      [§11]       TTT-LVSM [§9]   │
        │          │           │            │
        │          │      Curriculum        │
        │          │      Training [§14]    │
        │          │                        │
        ├──────────┤                        │
        ▼          ▼                        │
  Image-to-3D  Autoregressive ◄────────────┘
  Gen. [§15]   Recon. [§6]
                   │
              Scaling to
              128+ Views [§16]
                   │
            Datasets [§13]
            & Eval Setup
```

---

## 23. Key Equations

| Eq. | Name | Formula | Section |
|:---:|------|---------|:-------:|
| Eq.1 | Window Attention Residual | $\mathbf{T}_i = \mathbf{T}_i + \text{WinAttn}(\mathbf{T}_i)$ | §2 |
| Eq.2 | Fast Weight Update | $W = \text{Update}\!\bigl(\{\mathbf{T}_i\}_{i=1}^{N}\bigr)$ | §2 |
| Eq.3 | Fast Weight Apply | $\mathbf{T}_i = \text{Apply}(W,\, \mathbf{T}_i)$ | §2 |
| Eq.4 | Virtual Token Querying | $\mathbf{T}_i^v = \text{Apply}(W,\, \mathbf{T}_i^v)$ | §4 |
| Eq.5 | RGB Loss | $\mathcal{L}_{\text{RGB}} = \text{MSE}(\mathbf{I}_{\text{pred}},\, \mathbf{I}_{\text{gt}}) + \lambda\,\text{Perceptual}(\mathbf{I}_{\text{pred}},\, \mathbf{I}_{\text{gt}})$ | §8 |
| Eq.6 | Total Loss | $\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_{\text{depth}}\,\mathcal{L}_{\text{depth}} + \lambda_{\text{opacity}}\,\mathcal{L}_{\text{opacity}}$ | §8 |
| — | TTT Update Rule | $W \leftarrow W - \eta\,\nabla_W \mathcal{L}_{\text{MSE}}\!\bigl(f_W(\mathbf{k}),\,\mathbf{v}\bigr)$ | §1 |
| — | SwiGLU Fast Weight | $\text{out} = \bigl[\text{SiLU}(\mathbf{k}\,W_0) \odot \mathbf{k}\,W_2\bigr]\,W_1$ | §2 |
| — | Tokenization | $\{\mathbf{T}_{i,j}\} = \text{Tokenize}\!\bigl(\text{Patchify}([\{\mathbf{I}_i\},\,\{\mathbf{R}_i\}])\bigr)$ | §3 |
| — | Depth Normalization | $\hat{d} = (d - \text{median}(d)) \,/\, (\text{Var}(d) + \epsilon)$ | §8 |
| — | Fisher EMA | $\mathbf{F} \leftarrow \alpha\,\mathbf{F} + (1-\alpha)\,\lvert\nabla_\theta\mathcal{L}\rvert^2$ | §10 |
| — | Anchor EMA | $\theta^* \leftarrow \beta\,\theta^* + (1-\beta)\,\theta$ | §10 |
| — | Elastic Regularization | $\theta \leftarrow \theta - \lambda\,(1-\hat{\mathbf{F}})\,(\theta - \theta^*)$ | §10 |
| — | Newton-Schulz Step | $X_{k+1} = a\,X_k + b\,X_k^{\,3} + c\,X_k^{\,5}$ | §12 |

---

## 24. Reference Map

### Test-Time Training (TTT)
- [47] Sun et al. (2024) — Original TTT: Learning to learn at test time with RNNs
- [70] Zhang et al. (2025) — TTT done right: large chunk TTT (LaCT), TTT-LVSM
- [39] Schlag et al. (2021) — Linear transformers as fast weight programmers
- [2] Behrouz et al. (2024) — Titans: learning to memorize at test time
- [50] von Oswald et al. (2022) — MesaNet: sequence modeling by locally optimal TTT
- [59] Yang et al. (2024) — DeltaNet: parallelizing linear transformers with delta rule

### 3D Gaussian Splatting
- [21] Kerbl et al. (2023) — Original 3DGS
- [22] Kerbl et al. (2024) — Hierarchical 3DGS
- [65] Yu et al. (2024) — Mip-Splatting: alias-free 3D Gaussian splatting
- [31] Lu et al. (2024) — Scaffold-GS: structured 3D Gaussians

### Large Reconstruction Models
- [16] Hong et al. (2023) — LRM: large reconstruction model for single image to 3D
- [68] Zhang et al. (2024) — GS-LRM: large reconstruction model for 3D Gaussian splatting
- [71] Chen et al. (2025) — Long-LRM: long-sequence large reconstruction model
- [55] Wei et al. (2024) — MeshLRM: large reconstruction model for high-quality meshes

### Neural Radiance Fields
- [35] Mildenhall et al. (2021) — NeRF: neural radiance fields
- [36] Müller et al. (2022) — Instant neural graphics primitives
- [64] Yu et al. (2021) — pixelNeRF

### State Space Models
- [9] Dao and Gu (2024) — Transformers are SSMs
- [15] Gu et al. (2021) — Efficiently modeling long sequences with structured state spaces
- [30] Liu et al. (2024) — VMamba: visual state space model
- [25] Lenz et al. (2025) — Jamba: hybrid transformer-mamba

### Multi-View Stereo and Reconstruction
- [40] Schonberger and Frahm (2016) — Structure-from-motion revisited (COLMAP)
- [52] Wang et al. (2025) — VGGT: visual geometry grounded transformer
- [54] Wang et al. (2024) — Dust3r: geometric 3D vision made easy
- [58] Yang et al. (2025) — Fast3r: towards 3D reconstruction of 1000+ images

### Efficient Attention
- [20] Katharopoulos et al. (2020) — Linear attention transformers
- [42] Shen et al. (2021) — Efficient attention with linear complexities
- [49] Vaswani et al. (2017) — Attention is all you need

### Datasets
- [10] Deitke et al. (2023) — Objaverse: a universe of annotated 3D objects
- [27] Ling et al. (2024) — DL3DV-10K: large-scale scene dataset
- [23] Knapitsch et al. (2017) — Tanks and Temples

### Training Techniques
- [19] Jordan et al. (2024) — Muon optimizer for hidden layers in neural networks
- [33] Micikevicius et al. (2017) — Mixed precision training
- [67] Yuan et al. (2025) — Test3r: deferred backpropagation
- [32] Ma et al. (2025) — Elastic TTT: fast spatial memory with scalable TTT
- [69] Zhang et al. (2018) — LPIPS perceptual metric
