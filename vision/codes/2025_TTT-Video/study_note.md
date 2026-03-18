# One-Minute Video Generation with TTT — Study Note: A Concept Mind-Map

> Paper: *One-Minute Video Generation with Test-Time Training* (arXiv:2504.05298)
> Authors: Dalal\*, Koceja\*, Hussein\*, Xu\*, Zhao, Song, Han, Cheung, Kautz, Guestrin, Hashimoto, Koyejo, Choi, Sun, Wang
> Affiliations: NVIDIA, Stanford, UCSD, UC Berkeley, UT Austin
> Published: CVPR 2025

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** -- what it is, stated plainly
- **Properties** -- its mathematical or behavioral characteristics
- **Application** -- how the paper (or the field) uses it
- **Links** -- connections to other concepts in this map

---

## 0. The Big Picture

```
          Video Transformers generate short clips (3-20 sec)
          with single scenes, no complex stories
                            │
               "Self-attention is O(N²) -- too slow
                for 300k+ tokens in one-minute video"
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      Self-Attention    Linear RNNs     TTT Layers
      (quadratic)       (Mamba,         (neural network
                         DeltaNet)       hidden states)
            │               │               │
      too expensive    matrix hidden    MLP hidden state:
      for long ctx     state too small  2x more cells,
                       to remember      richer nonlinearity
                       deep relations
            │               │               │
            └───────────────┼───────────────┘
                            ▼
                Add TTT-MLP layers to
                CogVideo-X 5B (DiT)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         Gating +      Local Attn    Multi-stage
         Bi-direction  (3-sec seg)   Fine-tuning
         (smooth       + Global TTT  (3→9→18→30→63 sec)
          integration) (full video)
              │             │             │
              └─────────────┼─────────────┘
                            ▼
              One-minute videos from storyboards
              +34 Elo over 2nd best (human eval)
              Complex multi-scene stories
```

---

## 1. The Long-Context Problem in Video Generation

### Definition

The long-context problem is the fundamental bottleneck preventing video Transformers from generating long, complex videos. Self-attention layers have $O(N^2)$ cost in context length $N$. A one-minute video at standard tokenization requires over **300k tokens** in context.

### Properties

- With self-attention, generating one one-minute video takes $11\times$ longer than generating twenty 3-second videos
- Training takes $12\times$ longer
- Video tokens cannot be easily compressed by a tokenizer because videos contain dynamic motion
- As of March 2025, public APIs max out at: Sora 20s, MovieGen 16s, Ray 2 10s, Veo 2 8s
- None of these APIs autonomously generate complex multi-scene stories

### Application

This problem motivates the entire paper: replace global self-attention with a linear-complexity alternative that can handle 300k+ tokens while still remembering long-range dependencies between distant parts of the story.

The *Tom and Jerry* cartoon domain is intentionally limited in scope for fast research iteration. It emphasizes complex, multi-scene, long-range stories with dynamic motion (where progress is still needed), with less emphasis on visual and physical realism (where remarkable progress has already been made). The authors believe improvements in long-context capabilities for this domain will transfer to general-purpose video generation.

### Links

- -> **TTT Layers**: the proposed solution -- linear complexity with expressive hidden states
- -> **Linear RNN Layers**: existing linear-complexity alternatives (Mamba, DeltaNet) that are less expressive
- -> **Local Attention + Global TTT**: the architecture design that addresses the problem

---

## 2. TTT Layers -- Neural Network Hidden States

### Definition

A TTT (Test-Time Training) layer is an RNN layer whose hidden state is itself a **neural network** with weights $W$. The update rule is a gradient step on a self-supervised loss $\ell$:

$$W_t = W_{t-1} - \eta \,\nabla \ell(W_{t-1}; x_t) \tag{Eq.1}$$

The output token is the prediction made by the inner model $f$ on $x_t$:

$$z_t = f(x_t; W_t) \tag{Eq.2}$$

The self-supervised task is reconstruction from corrupted input $\tilde{x}_t$:

$$\ell(W; x_t) = \|f(\tilde{x}_t; W) - x_t\|^2 \tag{Eq.3}$$

Similar to denoising autoencoders, $f$ must discover correlations between dimensions of $x_t$ to reconstruct it from partial information.

```
Input tokens:   x_1      x_2      ...     x_t
                 │        │                 │
Hidden state: W_0 ──→ W_1 ──→ W_2 ──→ ... ──→ W_t   (gradient descent)
                 │        │                 │
Output tokens: z_1      z_2      ...      z_t        (f(x_t; W_t))
```

### Properties

- **Linear complexity**: $O(1)$ per token for both update and output rules (same as RNNs)
- **Expressive hidden states**: $W$ is a full neural network, not just a matrix -- can store much more information than the $d \times d$ matrices used in Mamba/DeltaNet
- Calling backward on $\nabla \ell$ means taking **gradients of gradients** -- a meta-learning technique
- Even at test time, the layer trains a different weight sequence $W_1, \ldots, W_T$ for every input sequence -- hence "Test-Time Training"
- TTT layers have the **same interface** as RNN layers and self-attention, so they can replace either in any architecture

**Two-loop structure:**
- **Outer loop**: training the larger network (the Diffusion Transformer)
- **Inner loop**: training $W$ within each TTT layer (happens during both training and inference)

### Application

The paper adds TTT layers to CogVideo-X 5B to process the entire one-minute video globally, while self-attention layers handle local 3-second segments.

### Links

- -> **Learned Self-Supervised Task**: the corruption $\tilde{x}_t$ and reconstruction label are learned, not handcrafted
- -> **TTT-MLP**: the specific instantiation where $f$ is a two-layer MLP
- -> **TTT-Linear**: the baseline where $f$ is a linear model
- -> **Linear RNN Layers**: TTT layers are more expressive because their hidden states are neural networks, not matrices
- -> **Parallelization**: the sequential dependency $W_{t-1} \to W_t$ is handled with inner-loop mini-batches

---

## 3. Learned Self-Supervised Task

### Definition

Instead of handcrafting the self-supervised task from human priors, Sun et al. (2024) learn it end-to-end as part of the outer loop. Starting from Eq. 3, they replace $\tilde{x}_t$ with a learnable low-rank projection $\theta_K x_t$ (perhaps not all information in $x_t$ is worth remembering), and the reconstruction label with $\theta_V x_t$:

$$\ell(W; x_t) = \|f(\theta_K x_t; W) - \theta_V x_t\|^2 \tag{Eq.4}$$

Since $\theta_K x_t$ has fewer dimensions than $x_t$, the output rule is also changed:

$$z_t = f(\theta_Q x_t; W_t) \tag{Eq.5}$$

### Properties

- In the inner loop, only $W$ is optimized; the $\theta$s are "hyperparameters" of the inner-loop loss
- $\theta_K, \theta_V, \theta_Q$ are optimized in the **outer loop**, analogous to the Key, Value, and Query parameters of self-attention
- This end-to-end approach removes the need for human-designed corruption schemes

### Links

- -> **TTT Layers**: this is the mechanism that defines how the hidden state is updated
- -> **Self-Attention Analogy**: $\theta_Q \leftrightarrow$ Query, $\theta_K \leftrightarrow$ Key, $\theta_V \leftrightarrow$ Value

---

## 4. TTT-MLP vs. TTT-Linear

### Definition

**TTT-MLP**: The inner model $f$ is a two-layer MLP with hidden dimension $4\times$ the input dimension, GELU activation, Layer Norm, and a residual connection:

$$f(x) = x + \text{LN}(f_{\text{MLP}}(x))$$

**TTT-Linear**: The inner model $f$ wraps a linear model instead of an MLP:

$$f(x) = x + \text{LN}(f_{\text{Linear}}(x))$$

### Properties

| Dimension | TTT-Linear | TTT-MLP | Mamba 2 |
|---|---|---|---|
| Inner model | Linear | 2-layer MLP | N/A (matrix state) |
| Hidden state size | Smaller | $2\times$ TTT-Linear | $\approx 4\times$ TTT-Linear, $\approx 2\times$ smaller than TTT-MLP |
| Nonlinearity | None | GELU | None |
| Inner-loop LR $\eta$ | 1.0 | 0.1 | N/A |
| 18-sec Elo (avg) | 1001 | 1004 | 1005 |
| **63-sec Elo (avg)** | eliminated | **1033** | 978 |

- At short context (~100k tokens, 18 seconds), TTT-Linear and TTT-MLP perform similarly, and both are beaten by Gated DeltaNet
- At long context (~341k tokens, 63 seconds), TTT-MLP pulls far ahead -- the MLP's richer hidden state becomes essential for remembering long-range dependencies

### Application

TTT-MLP is the default throughout the paper. TTT-Linear is used as a baseline and is eliminated in the 18-second round (it performs worse than TTT-MLP at that length).

### Links

- -> **TTT Layers**: both are instantiations of the same TTT framework with different $f$
- -> **On-Chip Tensor Parallel**: needed specifically for TTT-MLP because the MLP weights $W^{(1)}, W^{(2)}$ are too large for one SM's SMEM
- -> **Long-Context Problem**: the MLP's advantage over linear models is precisely in long-range memory

---

## 5. Architecture: Gating, Bi-direction, and Modified Blocks

### Definition

Three design choices for integrating TTT layers into a pre-trained Diffusion Transformer:

**Gating.** Inserting randomly-initialized TTT layers would initially degrade the pre-trained model. A learned gate smoothly blends TTT output with the original:

$$\text{gate}(\text{TTT}, X; \alpha) = \tanh(\alpha) \otimes \text{TTT}(X) + X \tag{Eq.6}$$

$\alpha \in \mathbb{R}^d$ is initialized to 0.1, so $\tanh(\alpha) \approx 0$ at the start of fine-tuning.

**Bi-direction.** Diffusion models are non-causal ($z_t$ conditions on all tokens, not just past ones). TTT layers are made non-causal by applying a reverse scan:

$$\text{TTT}'(X) = \text{rev}(\text{TTT}(\text{rev}(X))) \tag{Eq.7}$$

$\text{TTT}'(X)$ is still in chronological order, but the inner TTT scans in reverse.

**Modified block.** The standard Transformer block (Eq. 8-9) is extended by inserting TTT after each attention layer:

$$X' = \text{self\_attn}(\text{LN}(X)) \tag{Eq.8}$$
$$Y = X' + X \tag{Eq.9}$$

The paper modifies this by inserting TTT after Eq. 8, replacing Eq. 9:

$$Z = \text{gate}(\text{TTT}, X'; \alpha) \tag{Eq.10}$$
$$Z' = \text{gate}(\text{TTT}', Z; \beta) \tag{Eq.11}$$
$$Y = Z' + X \tag{Eq.12}$$

$\text{TTT}$ and $\text{TTT}'$ share parameters $\theta_K, \theta_V, \theta_Q$ but use different gate parameters $\alpha, \beta$.

```
     X (input)
      │
  ┌───┴───┐
  │ LN → Self-Attn (local, 3-sec segments)
  └───┬───┘
      │ = X'
  ┌───┴───┐
  │ gate(TTT, X'; α)        ← forward scan
  └───┬───┘
      │ = Z
  ┌───┴───┐
  │ gate(TTT', Z; β)        ← reverse scan
  └───┬───┘
      │ = Z'
      │
  Z' + X = Y (output)
```

### Properties

- Only sequence modelling blocks are modified; everything else (MLP blocks, etc.) is unchanged
- The approach is architecture-agnostic in principle -- CogVideo-X 5B is chosen as the most popular video DiT (generates 3 sec at 16 fps, or 6 sec at 8 fps)
- Gating ensures the pre-trained model is not disrupted at the start of fine-tuning
- Self-attention attends **locally** (within 3-sec segments); TTT attends **globally** (entire video)

### Links

- -> **TTT Layers**: the gating and bi-direction are wrappers around the core TTT mechanism
- -> **Local Attention + Global TTT**: this is the resulting dual-attention architecture
- -> **Pre-trained Model**: CogVideo-X 5B provides the initial weights

---

## 6. Local Attention + Global TTT Pipeline

### Definition

The overall pipeline splits the one-minute video into 3-second **segments** grouped into **scenes**. Self-attention operates locally within each segment. TTT layers operate globally across the entire sequence.

### Properties

- **3-second segment** is the atomic unit because: (a) CogVideo-X's max generation is 3 sec, (b) most *Tom and Jerry* scenes are at least 3 sec, (c) multi-stage dataset construction is most convenient
- Sequence segments overlap by 1 latent frame (1350 tokens) as a pre-processing artifact
- Text prompt is always in **Format 3** (storyboard): each 3-sec segment described by a paragraph of 3-5 sentences (avg. 98 words = 132 tokens)
- Three prompt formats of increasing detail (Format 1: plot summary, Format 2: sentence-per-segment, Format 3: storyboard with `<scene start>` / `<scene end>`)
- Conversion: Claude 3.7 Sonnet converts $1 \to 2 \to 3$ (direct $1 \to 3$ produces worse style matching)

**From text to sequences:** Given $n$ paragraphs, produce $n$ sequence segments (text tokens + noisy video tokens each), concatenate all $n$ to form the full input sequence with interleaved text and video tokens.

### Application

At inference: user provides a plot summary (Format 1). Claude 3.7 Sonnet converts to storyboard (Format 3). The model generates all segments in a single forward pass. No editing, stitching, or post-processing.

### Links

- -> **Architecture**: self-attention handles local 3-sec coherence; TTT handles global story coherence
- -> **Multi-Stage Fine-Tuning**: extends context from 3 sec to 63 sec in five stages
- -> **Tom and Jerry Dataset**: provides the storyboard annotations

---

## 7. Multi-Stage Fine-Tuning

### Definition

Context length is extended from 3 seconds to one minute in **five progressive stages**, following standard practice for LLMs.

### Properties

| Stage | Video len. | Ctx. len | Trainable parameters | LR | Schedule | Steps |
|---|---|---|---|---|---|---|
| 1 | 3 sec | 18,048 | TTT / Pre-trained | $10^{-4}$ / $10^{-5}$ | Cosine / Constant | 5,000 |
| 2 | 9 sec | 51,456 | TTT + Attn (QKVO) | $10^{-5}$ | Constant | 5,000 |
| 3 | 18 sec | 99,894 | TTT + Attn (QKVO) | $10^{-5}$ | Constant | 1,000 |
| 4 | 30 sec | 168,320 | TTT + Attn (QKVO) | $10^{-5}$ | Constant | 500 |
| 5 | 63 sec | 341,550 | TTT + Attn (QKVO) | $10^{-5}$ | Constant | 250 |

- Stage 1: fine-tune the **entire** pre-trained model; new TTT layers + gates get higher LR
- Stages 2-5: only fine-tune TTT layers, gates, and self-attention (QKVO) with lower LR, to avoid forgetting world knowledge from pre-training
- Total training: equivalent of 50 hours on 256 H100s

### Application

**Dataset construction:**
1. 81 episodes of *Tom and Jerry* (1940-1948), ~5 min each, ~7 hours total
2. Video super-resolution to $720 \times 480$
3. Human annotators: episode → scenes → 3-sec segments → paragraph per segment
4. Contiguous segments concatenated for longer stages (9, 18, 30, 63 sec)
5. Scene boundaries marked with keywords in Format 3

**Training configuration (Appendix A):**
- Optimizer: AdamW with $(\beta_1, \beta_2) = (0.9, 0.95)$
- LR warmup: linear over 2% of training steps
- Batch size: 64; Gradient clipping: 0.1
- Weight decay: $10^{-4}$ (all params except biases and normalization layers)
- VAE scale factor: 1.0; Dropout: zero-out text prompt with probability 0.1
- Precision: Mixed Precision with PyTorch FSDP2
- Diffusion: v-prediction, 1000 noise steps, ZeroSNR at final step
- TTT inner-loop LR: $\eta = 1.0$ (TTT-Linear), $\eta = 0.1$ (TTT-MLP)
- Sampling: DDIM with 50 steps, dynamic CFG from 1 to 4, negative prompts

### Links

- -> **Local Attention + Global TTT**: the architecture being fine-tuned
- -> **Tom and Jerry Dataset**: provides the training data at each stage

---

## 8. Parallelization for Non-Causal Sequences

### Definition

The sequential dependency $W_{t-1} \to W_t$ in Eq. 1 prevents naive parallelization. The solution: update $W$ on **mini-batches** of $b$ tokens at a time (inner-loop mini-batch, $b = 64$ throughout).

$$W_{ib} = W_{(i-1)b} - \frac{\eta}{b} \sum_{t=(i-1)b+1}^{ib} \nabla \ell\left(W_{(i-1)b}; x_t\right) \tag{Eq.13}$$

Because the sequence is non-causal, the same $W_{ib}$ produces outputs for all tokens in mini-batch $i$:

$$z_t = f(W_{ib}; x_t), \quad t = (i-1)b+1, \ldots, ib \tag{Eq.14}$$

### Properties

- Intermediate weights $W_{(i-1)b+1}, \ldots, W_{ib-1}$ are no longer needed
- $f$ can now process a mini-batch of tokens in parallel (like a regular MLP processing a training batch)
- Averaging gradients across tokens **reduces variance** and stabilizes updates
- The number of sequential steps is reduced from $T$ to $T/b$ (from 341k to ~5.3k for 63-sec video)

### Links

- -> **TTT Layers**: this is the parallelization scheme for the inner-loop update
- -> **On-Chip Tensor Parallel**: addresses the *memory* bottleneck (too large for one SM); this section addresses the *compute* bottleneck (sequential dependency)

---

## 9. On-Chip Tensor Parallel

### Definition

The TTT-MLP hidden state ($W^{(1)}$ and $W^{(2)}$ of the two-layer MLP) is too large to fit in a single Streaming Multiprocessor's (SM) on-chip SMEM. The solution: shard $W^{(1)}$ and $W^{(2)}$ across multiple SMs using **Tensor Parallelism**, keeping all computation on-chip and using **DSMEM** on NVIDIA Hopper GPUs for AllReduce among SMs.

```
  HBM (slow, large)
    │
    ├── W₀⁽¹⁾ ──→ SM 1: W₀⁽¹⁾_shard  ──→ compute on SMEM ──→ W_{t+1}⁽¹⁾_shard
    │                    AllReduce ↕
    └── W₀⁽²⁾ ──→ SM 2: W₀⁽²⁾_shard  ──→ compute on SMEM ──→ W_{t+1}⁽²⁾_shard
    │
    └── Only initial load and final output touch HBM
```

### Properties

- Sharding strategy: first MLP layer column-wise, second row-wise (standard Tensor Parallel)
- GeLU is element-wise, so only one reduction is needed for computing the inner loss
- Implementation uses **ThunderKittens** (Spector et al., 2025) for the fused kernel
- Additional optimizations: **producer-consumer asynchrony** (dedicated warpgroups for data loading vs. computation), gradient checkpointing along the sequence dimension with TMA
- **General principle**: if $f$ can be sharded across GPUs with Tensor Parallelism, the same strategy works across SMs when $f$ is the hidden state

### Application

This implementation keeps the hidden state entirely on-chip during updates. HBM is only accessed during initial loading and final output. Significantly improves efficiency, though TTT-MLP is still $1.4\times$ (inference) / $2.1\times$ (training) slower than Gated DeltaNet.

### Links

- -> **TTT-MLP**: this is the specific instantiation that requires on-chip TP (TTT-Linear would fit in one SM)
- -> **Parallelization**: on-chip TP addresses memory; mini-batching addresses sequential computation

---

## 10. Human Evaluation Results

### Definition

The paper uses **pairwise preference** human evaluation on four axes adapted from MovieGen (omitting "realness" and "motion completeness" which don't apply to cartoons), aggregated into Elo scores via the LMSys Chatbot Arena system. **Protocol**: 100 plots sampled via Claude 3.7 Sonnet (Format 1->2->3), one video per method per plot, blind pairwise comparisons where each evaluator sees a random axis and a random pair of videos from the same plot.

### Properties

**One-minute videos (63 sec, 341k tokens):**

| | Text follow. | Motion nat. | Aesthetics | Temporal cons. | **Average** |
|---|---|---|---|---|---|
| Mamba 2 | 985 | 976 | 963 | 988 | 978 |
| Gated DeltaNet | 983 | 984 | 993 | 1004 | 991 |
| Sliding window | **1016** | 1000 | 1006 | 975 | 999 |
| **TTT-MLP** | 1014 | **1039** | **1037** | **1042** | **1033** |

**18-second videos (100k tokens) -- elimination round:**

| | Text follow. | Motion nat. | Aesthetics | Temporal cons. | **Average** |
|---|---|---|---|---|---|
| Local Attention | 965 | 972 | 969 | 944 | 962 |
| TTT-Linear | 1003 | 995 | 1007 | 1001 | 1001 |
| Mamba 2 | **1023** | 987 | 1008 | 1004 | 1005 |
| **Gated DeltaNet** | 1020 | **1039** | **1044** | 1026 | **1032** |
| SWA | 995 | 1004 | 993 | 980 | 993 |
| TTT-MLP | 994 | 1002 | 1002 | 1019 | 1004 |

For context on the magnitude of the 34-Elo gap: GPT-4 scores 46 Elo points over GPT-3.5 Turbo (1163 vs. 1117), and GPT-4o scores 29 over GPT-4 Turbo (1285 vs. 1256) in LMSys Chatbot Arena.

**Key insight**: TTT-MLP's advantage is **context-length dependent**:
- At 18 sec (100k tokens): Gated DeltaNet leads by 28 Elo over TTT-MLP
- At 63 sec (341k tokens): TTT-MLP leads by 34 Elo over all others
- The crossover happens somewhere between 100k and 341k tokens

**Efficiency (63-second videos):**

| | vs. Local Attention (inference) | vs. Local Attention (training) |
|---|---|---|
| Full Attention | $11\times$ slower | $12\times$ slower |
| **TTT-MLP** | $2.5\times$ slower | $3.8\times$ slower |
| Gated DeltaNet | $1.8\times$ slower | $1.8\times$ slower |

Training efficiency is less important than inference efficiency because RNN layers are integrated after pre-training (which constitutes most of the training budget). Fine-tuning is a small fraction.

### Links

- -> **Long-Context Problem**: the results directly validate that TTT-MLP solves the long-context challenge better than alternatives
- -> **TTT-MLP vs. TTT-Linear**: TTT-Linear is eliminated at 18 sec; TTT-MLP thrives at 63 sec

---

## 11. Limitations and Artifacts

### Definition

Three categories of limitations, with artifacts common across **all methods** (not specific to TTT-MLP):

**Short context.** At 18-second scale (~100k tokens), Gated DeltaNet with its matrix hidden state outperforms TTT-MLP. The MLP hidden state's advantage only materializes at longer contexts.

**Wall-clock time.** TTT-MLP is $1.4\times$ / $2.1\times$ slower than Gated DeltaNet for inference/training. The kernel is bottlenecked by register spills and suboptimal asynchronous instruction ordering.

**Video artifacts** (common to all methods, likely from the pre-trained CogVideo-X 5B):
- **Temporal consistency**: objects morph at 3-sec segment boundaries because the diffusion model samples from different modes (e.g., boxes morph between segments of the same scene)
- **Motion naturalness**: objects float unnaturally because gravitational effects are not properly modelled (e.g., cheese hovers in mid-air rather than falling)
- **Aesthetics**: lighting changes don't consistently align with actions unless explicitly prompted (e.g., kitchen lighting becomes dramatically brighter as Tom turns around); complex camera movements like parallax sometimes inaccurate

### Links

- -> **On-Chip Tensor Parallel**: partially addresses the wall-clock time issue but doesn't fully close the gap
- -> **Future Work**: faster kernels and larger hidden states are the proposed remedies

---

## 12. What Existed Before and What This Paper Changes

### 12.1 Prior Approaches and Their Limitations

**Self-attention in video Transformers** generates high-quality short clips (3-20 seconds) but its $O(N^2)$ cost makes it impractical for one-minute videos with 300k+ tokens. Full attention on a one-minute video would take $11\times$ longer for inference and $12\times$ longer for training.

**Mamba and Mamba 2** (Gu and Dao, 2024; Dao, 2024) are linear-time RNN layers with matrix hidden states. They compress context into a fixed-size matrix, which works well for natural language but struggles with complex visual stories. The matrix has limited rank, making it hard to encode deep relationships between distant video tokens. At 63 seconds, Mamba 2 scores only 978 Elo (vs. TTT-MLP's 1033).

**Gated DeltaNet** (Yang et al., 2025) extends DeltaNet and Mamba 2 with an improved update rule. It is the strongest baseline, scoring 1032 Elo at 18 seconds. But at 63 seconds, it drops to 991 Elo -- a 42-point gap below TTT-MLP.

**Sliding-window attention** uses a fixed window of 8192 tokens (~1.5 sec of video). It can capture local context well (1016 Elo for text following at 63 sec) but has no mechanism for long-range consistency (975 Elo for temporal consistency).

**Story synthesis methods** (StoryGAN, Make-a-Story, StoryDiffusion) generate sequences of images/videos from text stories, but require additional components for cross-scene coherence and are not end-to-end.

### 12.2 What This Paper Contributes

**Contribution 1**: TTT-MLP layers for video generation. Neural network hidden states with $2\times$ more cells and richer nonlinearities than linear RNN variants. Added to a pre-trained DiT with gating and bi-direction.

**Contribution 2**: End-to-end one-minute video generation. First demonstration of autonomous, multi-scene, story-driven video generation from text storyboards, in a single shot.

**Contribution 3**: On-chip Tensor Parallel for TTT-MLP. Shards the hidden state across SMs, keeping computation in fast SMEM. General principle applicable to any TTT instantiation.

**Contribution 4**: Tom and Jerry dataset with human-annotated storyboards for long-context video research.

### 12.3 Side-by-Side Comparison

| Dimension | Self-Attention | Mamba 2 / DeltaNet | **TTT-MLP** |
|---|---|---|---|
| Complexity | $O(N^2)$ | $O(N)$ | $O(N)$ |
| Hidden state | KV cache (grows) | $d \times d$ matrix (fixed) | **Neural network** (fixed, larger) |
| Long-range memory | Unlimited (if fits) | Limited by matrix rank | Rich (MLP expressiveness) |
| 63-sec Elo | N/A (too slow) | 978-991 | **1033** |
| Max practical length | ~20 sec | Unknown | **63 sec (tested)** |

### 12.4 The Core Shift in Thinking

Prior video generation work focused on improving visual quality within short clips. The assumption was that long videos could be created by stitching short clips or using autoregressive generation frame-by-frame. This paper asks a different question: can a single model generate a long, complex story in one shot?

The answer requires solving long-context memory. Linear RNN layers (Mamba, DeltaNet) were the state of the art for linear-complexity sequence modelling. But their matrix hidden states act as a bottleneck when the context grows to hundreds of thousands of tokens: compressing that much information into a matrix with limited rank loses deep relationships between distant tokens.

TTT layers break this bottleneck by making the hidden state a neural network -- specifically a two-layer MLP. The weights of the MLP form a much richer state space than a single matrix. Gradient descent on a self-supervised task serves as the update rule, similar to how a human might "learn the story" by reading through it. The result: at one-minute scale, the MLP hidden state's advantage over matrix states becomes decisive (+34 Elo), precisely on the axes that require long-range memory (temporal consistency +38, motion naturalness +39).

---

## 13. Quick Reference Card

| # | Finding | Design Choice | Evidence |
|---|---|---|---|
| 1 | TTT-MLP beats all baselines at 63 sec | Neural network hidden states | +34 Elo over second-best (Table 1) |
| 2 | Advantage is context-length dependent | Use TTT-MLP only when >100k tokens | Gated DeltaNet leads at 18 sec (Table 3) |
| 3 | Biggest gains on long-range axes | TTT's memory enables story coherence | Temporal consistency +38, Motion +39 |
| 4 | Gating enables smooth integration | $\tanh(\alpha)$ initialized near 0 | Pre-trained model not disrupted (Eq. 6) |
| 5 | 1-min video = 341k tokens | 3-sec segments with global TTT | Context extends 5 stages (Table 2) |
| 6 | All 7.2B parameter models | Fair comparison, same backbone | Same fine-tuning recipe for all |
| 7 | TTT-MLP still slower than DeltaNet | On-chip TP helps but not enough | $1.4\times$ inference, $2.1\times$ training |
| 8 | Artifacts from pre-trained model | Not TTT-specific, common to all | CogVideo-X 5B limited (Figure 7) |

---

## 14. Open Questions

1. **Faster TTT-MLP kernels.** Register spills and suboptimal async instruction ordering are the current bottlenecks. Compiler-aware implementations could close the gap with Gated DeltaNet.
2. **Better integration strategies.** Gating + bi-direction is one approach. Autoregressive video backbones may need entirely different strategies. Better strategies could accelerate fine-tuning.
3. **Longer videos with larger hidden states.** The approach extends with linear complexity. The key: instantiate $f$ as a much larger neural network -- even a Transformer itself.
4. **Where is the crossover point?** TTT-MLP loses at 18 sec but wins at 63 sec. What context length is the break-even? Does it depend on story complexity?
5. **Beyond cartoons.** The Tom and Jerry domain was chosen for fast iteration. Will long-context gains transfer to photorealistic video with physical realism?
6. **Handling segment boundaries.** Objects morph at 3-sec boundaries because the diffusion model samples from different modes. Can segment overlaps or consistency losses fix this?

---

## 15. Concept Dependency Graph

```
              Long-Context Problem
              (300k+ tokens, O(N²) too slow)
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Self-Attn     Linear RNNs    TTT Layers
     (too slow)    (Mamba, DN)    (MLP hidden)
                   (matrix too      │
                    small)     ┌────┴────┐
                               ▼         ▼
                          TTT-MLP    TTT-Linear
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              Learned SS   Gating +   Paralleliz.
              Task (Eq.4)  Bi-dir     (mini-batch)
                    │      (Eq.6-7)       │
                    └──────────┼──────────┘
                               ▼
                    Architecture: Modified Block
                    (Local Attn + Global TTT)
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              Multi-Stage   On-Chip    T&J Dataset
              Fine-Tuning   Tensor     (storyboards)
              (3→63 sec)    Parallel
                    │          │
                    └──────────┼──────────┘
                               ▼
                    One-Minute Video Generation
                    (+34 Elo, human eval)
```

---

## 16. Key Equations

| # | Equation | Description | Section |
|---|---|---|---|
| Eq.1 | $W_t = W_{t-1} - \eta\nabla\ell(W_{t-1}; x_t)$ | Hidden state update rule | 2.1 |
| Eq.2 | $z_t = f(x_t; W_t)$ | Output rule | 2.1 |
| Eq.3 | $\ell(W;x_t) = \|f(\tilde{x}_t;W) - x_t\|^2$ | Naive reconstruction loss | 2.1 |
| Eq.4 | $\ell(W;x_t) = \|f(\theta_K x_t;W) - \theta_V x_t\|^2$ | Learned self-supervised loss | 2.2 |
| Eq.5 | $z_t = f(\theta_Q x_t; W_t)$ | Learned output rule | 2.2 |
| Eq.6 | $\text{gate}(\text{TTT},X;\alpha) = \tanh(\alpha)\otimes\text{TTT}(X)+X$ | Gating mechanism | 3.1 |
| Eq.7 | $\text{TTT}'(X) = \text{rev}(\text{TTT}(\text{rev}(X)))$ | Bi-direction | 3.1 |
| Eq.8-9 | $X'=\text{self\_attn}(\text{LN}(X));\; Y=X'+X$ | Standard Transformer block | 3.1 |
| Eq.10-12 | $Z=\text{gate}(X';\alpha);\; Z'=\text{gate}(Z;\beta);\; Y=Z'+X$ | Modified block with TTT | 3.1 |
| Eq.13 | $W_{ib} = W_{(i-1)b} - \frac{\eta}{b}\sum\nabla\ell(W_{(i-1)b};x_t)$ | Mini-batch parallelization | 3.4 |
| Eq.14 | $z_t = f(W_{ib}; x_t)$ for $t$ in mini-batch $i$ | Non-causal output | 3.4 |
| $f$ | $f(x) = x + \text{LN}(f_{\text{MLP}}(x))$ | TTT-MLP instantiation | 2.3 |

---

## 17. Reference Map

**TTT Foundations**
- Sun et al. (2024): TTT layers with expressive hidden states (*arXiv:2407.04620*)
- Behrouz et al. (2024): Titans -- even larger and more nonlinear hidden states

**Linear RNN Layers**
- Katharopoulos et al. (2020): Linear attention (Transformers are RNNs)
- Dao and Gu (2024): Mamba 2 (Transformers are SSMs)
- Yang et al. (2024): DeltaNet (parallelizing linear transformers)
- Yang et al. (2025): Gated DeltaNet (improved update rule)

**Fast Weight Programmers**
- Schmidhuber (1992): Learning to control fast-weight memories
- Irie et al. (2021): Going beyond linear transformers with recurrent FWP
- Kirsch and Schmidhuber (2021): Meta learning backpropagation

**Video Generation**
- Peebles and Xie (2023): Scalable diffusion models with transformers (DiT)
- Yang et al. (2025) / Hong et al. (2023): CogVideoX / CogVideo
- The Movie Gen team (2024): MovieGen

**Long Video Modelling**
- Ge et al. (2022): TATS -- sliding-window attention for longer-than-training video
- Villegas et al. (2023): Phenaki -- variable length video generation
- Wang et al. (2024): Lingen -- minute-length text-to-video with linear complexity

**Story Synthesis**
- Li et al. (2019): StoryGAN
- Rahman et al. (2023): Make-a-Story -- visual memory conditioned consistent story generation
- Schwenk et al. (2018): Craft -- scripts to videos via retrieval
- Zhou et al. (2024): StoryDiffusion -- consistent self-attention for long-range generation
- Liu et al. (2024): Grimm -- visual storytelling via latent diffusion

**GPU Systems**
- Spector et al. (2025): ThunderKittens (fused GPU kernels)
- Dao et al. (2022): FlashAttention (io-aware exact attention)
- Shoeybi et al. (2019): Megatron-LM (model parallelism)

**Evaluation**
- Chiang et al. (2024): Chatbot Arena (Elo-based LLM evaluation)
- The Movie Gen team (2024): MovieGen evaluation axes
