# Test-Time Training on Video Streams — Study Note: A Concept Mind-Map

> Paper: *Test-Time Training on Video Streams* (arXiv:2307.05014v3)
> Authors: Renhao Wang\*, Yu Sun\*, Arnuv Tandon, Yossi Gandelsman, Xinlei Chen, Alexei A. Efros, Xiaolong Wang
> Affiliations: UC Berkeley, Stanford, Meta AI, UC San Diego
> Published: JMLR 26 (2025), pp. 1-29

This note organizes every key concept in the paper as a mind-map.
Each concept is broken down into four facets:

- **Definition** -- what it is, stated plainly
- **Properties** -- its mathematical or behavioral characteristics
- **Application** -- how the paper (or the field) uses it
- **Links** -- connections to other concepts in this map

---

## 0. The Big Picture

```
              Fixed models trained on still images
                          │
           "Must prepare for ALL possible futures,
            but only ONE future actually happens"
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   TTT on corrupted    TTT-MAE on       Continual
   still images        still images      Learning
   (Sun 2020)          (Gandelsman 2022) (replay buffer)
         │                │                │
         │    locality is │meaningless     │ forgetting
         │    (shared     │                │ is "bad"
         │     corruption)│                │
         └────────────────┼────────────────┘
                          ▼
            This paper: TTT on VIDEO STREAMS
                          │
              ┌───────────┼───────────────┐
              ▼           ▼               ▼
         Implicit     Explicit        Temporal
         Memory       Memory          Smoothness
         (carry-over  (sliding        (x_t ≈ x_{t+1})
          params)      window k)
              │           │               │
              └───────────┼───────────────┘
                          ▼
                    LOCALITY wins:
              online > offline oracle
              forgetting is BENEFICIAL
                          │
              ┌───────────┼───────────────┐
              ▼           ▼               ▼
         Bias-Variance  4 tasks,       Connection to
         Theory for k   3 datasets     RNN / Sequence
                                        Modelling
```

---

## 1. Test-Time Training (TTT) -- The Core Paradigm

### Definition

Test-Time Training is a framework that continues training a model on each test instance before making a prediction. Since no ground truth labels are available at test time, training uses a **self-supervised task** (e.g., image reconstruction). The core idea: a model fixed for all possible futures wastes capacity. Instead, adapt to the specific future that actually arrives.

The general architecture is Y-shaped with three components:
- **Encoder** $f$: shared feature extractor (the stem)
- **Self-supervised head** $g$: decoder for reconstruction
- **Main task head** $h$: for the downstream task (segmentation, colorization)

```
                    x (input)
                       │
                  ┌────┴────┐
                  │ f (encoder) │
                  └────┬────┘
                       │
              features of f
                       │
              ┌────────┼────────┐
              ▼                 ▼
     ┌────────────┐    ┌────────────┐
     │ g (decoder)│    │ h (main    │
     │ self-sup   │    │  task head)│
     └────────────┘    └────────────┘
         ℓ_s               ℓ_m
    (reconstruction)  (segmentation)
```

### Properties

- At test time, only $\ell_s$ is available (no labels for $\ell_m$)
- The self-supervised gradient $\nabla \ell_s^t$ is a noisy proxy for $\nabla \ell_m^t$ (Assumption 3 of the Theorem: $\nabla \ell_m^t = \nabla \ell_s^t + \delta_t$ with $\delta_t$ zero-mean, variance $\sigma^2$)
- Because MAE's decoder $g$ is small and only processes 20% of patches via $f$, the self-supervised task costs ~25% of the main task. Each gradient step (forward + backward) costs half a main-task forward pass
- TTT-MAE with **reset** (Gandelsman et al., 2022) treats each test image independently: always start from $f_0, g_0$, discard $f', g'$ after prediction
- This paper's extension: **do not reset** -- carry parameters forward through time
- Three training-time options: joint training (used here), probing (Gandelsman 2022), fine-tuning (unsuitable -- makes $h$ rely too much on main-task features)
- The idea of training at test time has deep roots in **transductive learning** (Joachims, 2002; Vapnik, 2013). The principle of transduction (Gammerman et al., 1998; Vapnik and Kotz, 2006): "Try to get the answer that you really need but not a more general one"

### Application

At test time, TTT-MAE solves the one-sample problem:

$$f', g' = \arg\min_{f,g}\; \ell_s(g \circ f(\bar{x}'), x') \tag{Eq.1}$$

then predicts $h_0 \circ f'(x')$. The self-supervised loss $\ell_s$ is pixel-wise MSE between reconstructed and original masked patches.

The paper uses **Mask2Former** with **Swin-S** backbone (69M parameters) as the architecture. Encoder $f$ = Swin-S backbone; head $h$ = everything after the backbone; decoder $g$ = same architecture as $h$ except the last layer maps to pixel space. Because Swin uses convolutions (not ViT), masked patches are replaced with black + a 4th binary channel indicating masked pixels (following Pathak et al., 2016).

### Links

- -> **Implicit Memory**: obtained by NOT resetting $f, g$ after each test frame
- -> **Explicit Memory**: obtained by optimizing $\ell_s$ over a window of frames, not just one
- -> **Temporal Smoothness**: the assumption that makes both forms of memory effective
- -> **Locality**: the central thesis -- local TTT outperforms global TTT
- -> **Joint Training**: how $f, g, h$ are trained together at training time to prepare $g$ for TTT
- -> **Sequence Modelling**: TTT can be viewed as an RNN update rule

---

## 2. Locality -- The Central Thesis

### Definition

**Locality** is the principle that for making a prediction on the current frame $x_t$, training on only a small number of recent frames (local data) is better than training on all frames (global data). The question is: should we perform TTT *offline* on all of $x_1, \ldots, x_T$ or *online* on only $x_t$ and a few previous frames?

### Properties

- The principle holds because of the **bias-variance trade-off** formalized in Theorem 1
- Including distant frames increases **bias** (they are less relevant to $x_t$)
- Using too few frames increases **variance** (noisy estimate of useful gradient)
- The theoretical optimal window size balances these: $k^* = (\sigma^2 / \beta^2 \eta^2)^{1/3}$
- Empirically, $k = 16$ (1.6 seconds at 10 fps) is near-optimal
- **Forgetting is beneficial**: limiting the sliding window is a form of forgetting, and it helps

The key empirical result:

| Setting | Method | COCO Instance | COCO Panoptic | KITTI Val. |
|---|---|---|---|---|
| No memory (most local) | TTT-MAE No Memory | 35.4 | 20.1 | 53.6 |
| Online (local window) | **Online TTT-MAE** | **37.6** | **21.7** | **55.4** |
| Offline (all frames) | Offline TTT-MAE | 33.6 | 19.6 | 53.2 |

Even the most local method (No Memory, using only the current frame) beats the most global method (Offline, using all frames) on COCO Videos -- **local is better than global if one has to pick an extreme**.

The offline oracle was given every advantage: all frames shuffled into a training set, and results reported from the **best iteration on each test video as measured by actual test performance** (around iteration 1000 for many videos) -- information that would not be available in the real world. Even so, online TTT wins.

> **Figure 5 (locality illustration)**: The current frame $t$ is shot inside a lecture hall. Frame at $t - 10$ is still inside the same hall -- including it in the sliding window decreases variance. Frame at $t - 200$ was shot before entering the hall -- including it would significantly increase bias because it is no longer relevant.

### Application

Locality manifests concretely through two design choices:
1. **Implicit memory** with short carry-over (no reset, so recent optimization matters most)
2. **Explicit memory** with a small sliding window ($k = 16$)

The paper shows that online TTT produces >2.2x improvement for instance segmentation and >1.5x for panoptic segmentation relative to the fixed model baseline, even outperforming the offline oracle that accesses the entire video.

### Links

- -> **Bias-Variance Trade-off**: the theoretical formalization of why locality works
- -> **Temporal Smoothness**: the underlying property of video that enables locality
- -> **Implicit Memory**: one mechanism for implementing locality
- -> **Explicit Memory**: the other mechanism (sliding window = local data)
- -> **Continual Learning**: challenges the conventional wisdom that forgetting is harmful
- -> **UDA**: challenges the perspective of seeing each video as a single target domain

---

## 3. Temporal Smoothness

### Definition

Temporal smoothness is the property that consecutive video frames $x_t$ and $x_{t+1}$ are visually similar. Formally stated as Assumption 2 of the Theorem:

$$\|x_{t+1} - x_t\| \leq \eta \tag{Assumption 2}$$

where $\eta$ is small for natural video. The norm is $L^2$, though any norm works if the strong convexity assumption is adjusted.

### Properties

- This is a property of **natural video** -- it does not hold for synthetic corrupted image sequences where all frames share the same corruption
- It is the fundamental reason both implicit and explicit memory work: if the world changes slowly, what the model learned on $x_{t-1}$ is still useful for $x_t$
- **Verification**: shuffling all frames within each video destroys temporal smoothness. For Online TTT-MAE, shuffling drops KITTI-STEP performance **below the fixed-model baseline** (Main Task Only). Methods that treat frames independently (*Main Task Only*, *MAE Joint Training*, *TTT-MAE No Memory*) are unaffected; so is *Offline TTT-MAE All Frames* (which already shuffles during training)
- Temporal smoothness enters the bias term of the Theorem as $\eta^2$: larger changes between frames increase bias

### Application

In practice, temporal smoothness means:
- **Implicit memory** works: parameters optimized on $x_{t-1}$ provide a good initialization for $x_t$, so only 1 gradient step per frame is needed (vs. 20 in TTT-MAE on still images)
- **Explicit memory** works: recent frames in the sliding window contain useful self-supervised signal for the current frame
- The method runs 2.3x slower than the fixed baseline (vs. Gandelsman et al.'s 20-iteration method that would be much slower), because implicit memory replaces many iterations with carry-over

### Links

- -> **Locality**: temporal smoothness is what makes local data more relevant than distant data
- -> **Implicit Memory**: works because $x_t \approx x_{t-1}$ means $f_{t-1}$ is already close to what $f_t$ needs to be
- -> **Explicit Memory**: works because nearby frames share visual content
- -> **Bias-Variance Trade-off**: $\eta$ is the smoothness parameter that controls bias

---

## 4. Implicit Memory

### Definition

Implicit memory means **not resetting** model parameters between timesteps. At timestep $t$, initialize test-time training with $f_{t-1}$ and $g_{t-1}$ (the parameters from the previous frame) instead of $f_0$ and $g_0$ (the original training-time parameters). Information from past frames is encoded implicitly in the parameter trajectory.

In prior work (Sun et al., 2020), TTT with implicit memory is called the "online" version, in contrast to the "standard" version with reset.

### Properties

- Biologically plausible: humans do not constantly reset their minds
- Creates a chain of dependencies: $f_0 \to f_1 \to \cdots \to f_t$, where each $f_t$ carries information from all past frames
- Because $x_t \approx x_{t-1}$, parameters $f_{t-1}$ already provide a near-optimal starting point for training on $x_t$
- Only **one gradient step per frame** is sufficient (vs. 20 iterations in TTT-MAE on still images)
- Ablation (Table 4): Implicit Memory Only improves over No Memory on all metrics (+0.7 AP instance, +0.6 PQ panoptic, +0.7/+1.9 mIoU on KITTI-STEP val/test)
- Implicit memory is the larger contributor compared to explicit memory

| Method | Instance | Panoptic | KITTI Val. | KITTI Test |
|---|---|---|---|---|
| TTT-MAE No Memory | 35.4 | 20.1 | 53.6 | 52.5 |
| Implicit Memory Only | 36.1 | 20.7 | 54.3 | 54.4 |
| Explicit Memory Only | 35.7 | 20.2 | 53.6 | 52.5 |
| **Online TTT-MAE (Both)** | **37.6** | **21.7** | **55.4** | **54.3** |

### Application

In the algorithm loop, implicit memory is trivially implemented: just don't call `model.load_state_dict(initial_weights)` between frames. The model parameters persist from one frame to the next.

### Links

- -> **Temporal Smoothness**: the reason $f_{t-1}$ is a good initialization for frame $t$
- -> **Explicit Memory**: complementary mechanism -- both contribute to improvement
- -> **Sequence Modelling**: the parameter trajectory $f_0 \to f_1 \to \cdots$ is exactly an RNN hidden state
- -> **Bias-Variance Trade-off**: implicit memory corresponds to the extreme $k=1$ in the window-size analysis
- -> **TTT on Nearest Neighbours**: the sliding window retrieves temporal "neighbours" from the unlabeled test video instead of a labelled training set

---

## 5. Explicit Memory

### Definition

Explicit memory means keeping recent frames in a **sliding window** of size $k$. At each timestep $t$, instead of optimizing $\ell_s$ on just $x_t$ (Eq. 1), optimize over the window:

$$f_t, g_t = \arg\min_{f,g}\; \frac{1}{k} \sum_{t'=t-k+1}^{t} \ell_s(g \circ f(\bar{x}_{t'}), x_{t'}) \tag{Eq.2}$$

before predicting $h_0 \circ f_t(x_t)$.

### Properties

- Optimization uses SGD: sample a batch **with replacement**, uniformly from the window
- Masking (80%) is applied independently within and across batches
- Only **one iteration** is needed per frame (combined with implicit memory)
- The batch size and computational cost are **fixed** regardless of window size $k$
- Window size $k$ interpolates between two extremes:
  - $k = 1$: same as Implicit Memory Only (no explicit window)
  - $k = \infty$: approaches Offline TTT-MAE All Frames (except future frames are excluded)
- Optimal $k$ is around 8-32 for all three tasks (Figure 4); performance is not very sensitive in linear scale
- In all experiments, $k = 16$ is used (selected on KITTI-STEP validation set; covers 1.6 seconds at 10 fps)

### Application

The sliding window stores the $k$ most recent frames. At each timestep:
1. Add $x_t$ to the window, remove $x_{t-k}$ if window is full
2. Sample a batch from the window (with replacement)
3. Apply random masking (80%) to each sampled frame
4. Take one gradient step on the reconstruction loss
5. Predict $h_0 \circ f_t(x_t)$

### Links

- -> **Locality**: the sliding window is the concrete mechanism for local training
- -> **Implicit Memory**: complementary -- explicit memory provides diverse data, implicit provides good initialization
- -> **Bias-Variance Trade-off**: $k$ directly appears in the bound; controls the bias-variance balance
- -> **Temporal Smoothness**: frames inside the window are similar to $x_t$ because they are temporally close

---

## 6. Joint Training

### Definition

Joint training is the training-time procedure where all three model components ($f$, $g$, $h$) are optimized in a single stage, end-to-end, on both the self-supervised and main task losses simultaneously:

$$g_0, h_0, f_0 = \arg\min_{g,h,f}\; \frac{1}{n} \sum_{i=1}^{n} \left[\ell_m(h \circ f(x_i), y_i) + \ell_s(g \circ f(\bar{x}_i), x_i)\right]$$

This contrasts with **probing** (used by Gandelsman et al., 2022), a two-stage process: first pre-train $f, g$ with $\ell_s$, then train $h$ with frozen $f_0$.

### Properties

- The purpose is to make $g$ **well-initialized for TTT**. Without joint training, $g$ would have to be trained from scratch at test time
- Joint training starts from a Mask2Former checkpoint already trained for the main task; only $g$ is initialized from scratch
- Training uses **labeled still images** (e.g., COCO training set, CityScapes), not unlabeled videos
- In TTT-MAE (Gandelsman et al., 2022), probing performed better. This paper successfully tuned joint training to match probing, and prefers it because it is simpler
- The fixed model $h_0 \circ f_0$ after joint training (MAE Joint Training baseline) performs roughly the same as Main Task Only. Joint training does not hurt or help the fixed model

### Application

- For KITTI-STEP: joint training on CityScapes (same 19 categories, still images instead of videos)
- For COCO Videos: joint training on COCO still images; 3 videos used only for evaluation
- All hyperparameters tuned on KITTI-STEP validation set, then transferred to COCO Videos in a single run

### Links

- -> **TTT**: prepares the model for test-time adaptation by initializing $g$ alongside $f$ and $h$
- -> **Probing**: the alternative two-stage training approach from TTT-MAE (Gandelsman et al., 2022)

---

## 7. Bias-Variance Trade-off Theory

### Definition

The paper provides a formal analysis of how window size $k$ affects the quality of test-time training. The core question: should we use gradients from many past frames (large $k$) or few (small $k$)?

Define the gradient notation:

$$\nabla \ell_m^t(\theta) := \nabla_\theta \ell_m(x_t, y_t; \theta) \tag{Eq.3}$$
$$\nabla \ell_s^t(\theta) := \nabla_\theta \ell_s(x_t; \theta) \tag{Eq.4}$$

TTT uses the averaged self-supervised gradient over a window:

$$\frac{1}{k}\sum_{t'=t-k+1}^{t} \nabla \ell_s^{t'} \tag{Eq.5}$$

**Theorem.** Under three assumptions, the excess risk is bounded:

$$\mathbb{E}\left[\ell_m(x_t, y_t; \bar{\theta}) - \ell_m(x_t, y_t; \theta^*)\right] \leq \frac{1}{2\alpha}\left(k^2 \beta^2 \eta^2 + \frac{1}{k}\sigma^2\right)$$

### Properties

The bound decomposes into two terms:

| Term | Expression | Controlled by | Effect of larger $k$ |
|---|---|---|---|
| **Bias** | $k^2 \beta^2 \eta^2$ | $\eta$ (frame change rate), $\beta$ (loss smoothness) | Increases -- distant frames less relevant |
| **Variance** | $\sigma^2 / k$ | $\sigma^2$ (gap between $\nabla \ell_s$ and $\nabla \ell_m$) | Decreases -- more data reduces noise |

**Three assumptions:**
1. In a local neighbourhood of $\theta^*$, $\ell_m^t$ is $\alpha$-strongly convex in $\theta$ and $\beta$-smooth in $x$. (Widely accepted: Allen-Zhu et al., 2019; Zhong et al., 2017; Wang et al., 2021)
2. $\|x_{t+1} - x_t\| \leq \eta$ -- temporal smoothness in $L^2$ norm
3. $\nabla \ell_m^t = \nabla \ell_s^t + \delta_t$ where $\delta_t$ has mean zero and variance $\sigma^2$ -- self-supervised gradient is a noisy estimate of the main-task gradient (from Sun et al., 2020)

**Optimal window size** (minimizing the bound w.r.t. $k$):

$$k^* = \left(\frac{\sigma^2}{\beta^2 \eta^2}\right)^{1/3}$$

### Derivation: Proof sketch (Appendix C)

**Step 1.** By Assumptions 1 and 2: $\|\nabla \ell_m^t - \nabla \ell_m^{t-1}\| \leq \beta\eta$

**Step 2.** Decompose the averaged gradient using Assumption 3 and telescoping differences:

$$\frac{1}{k}\sum_{t'=t-k+1}^{t} \nabla \ell_s^{t'} = \nabla \ell_m^t + \frac{1}{k}\underbrace{\left[\sum_{t'}\sum_{t''}(\nabla \ell_m^{t''} - \nabla \ell_m^{t''+1})\right]}_{A\text{ (bias)}} + \frac{1}{k}\underbrace{\sum_{t'}\delta_{t'}}_{B\text{ (variance)}}$$

**Step 3.** Bound the squared deviation: $\mathbb{E}\|\frac{1}{k}\sum \nabla \ell_s^{t'} - \nabla \ell_m^t\|^2 \leq k^2\beta^2\eta^2 + \frac{1}{k}\sigma^2$

**Step 4.** Apply the Lemma (perturbation bound for strongly convex functions): if optimizing a perturbed gradient $\nabla f + v$ converges to $\tilde{x}^*$ instead of $x^*$, then $f(\tilde{x}^*) - f(x^*) \leq \frac{1}{2\alpha}\|v\|^2$

### Application

The theory predicts empirical behavior: Figure 4 shows performance peaks at $k \approx 8$-$32$ and declines for both very small and very large windows. The theory also explains **why online beats offline**: offline uses all frames ($k \to \infty$), which drives bias arbitrarily high.

### Links

- -> **Locality**: this theorem is the formal justification for locality
- -> **Temporal Smoothness**: parameter $\eta$ is the smoothness bound; smaller $\eta$ allows larger $k$
- -> **Explicit Memory**: window size $k$ is the variable being optimized
- -> **TTT**: Assumption 3 formalizes the gap between self-supervised and main-task gradients

---

## 8. Online TTT-MAE -- The Full Algorithm

### Definition

Online TTT-MAE is the complete method proposed by this paper. It combines TTT-MAE with both implicit and explicit memory, applied to video streams in temporal order.

```
For each video (independent unit):
  Initialize f = f_0, g = g_0  (from joint training)
  Window W = {}

  For t = 1, 2, ..., T:
    1. Receive frame x_t
    2. Add x_t to sliding window W (keep last k frames)
    3. Sample batch from W with replacement
    4. Apply random 80% masking to each sample
    5. Take 1 gradient step on ℓ_s to update f, g
       (implicit memory: f, g carry from step t-1)
    6. Predict: output = h_0 ∘ f(x_t)
       (h_0 is NEVER updated at test time)
```

### Properties

- **1 gradient step per frame** -- not 20 as in original TTT-MAE
- **2.3x slower** than the fixed baseline (Main Task Only: 1.8s, Online TTT-MAE: 4.1s per frame on A100)
- Works across 4 tasks and 3 datasets with **the same hyperparameters** (tuned only on KITTI-STEP validation)
- Main task head $h_0$ is **never modified** at test time -- only $f$ and $g$ are updated
- Each video is treated as an independent unit -- parameters are reset to $f_0, g_0$ at the start of each new video

### Application

**Main results (Table 1):**

| Setting | Method | Instance AP | Panoptic PQ | KITTI Val. | KITTI Test | Time |
|---|---|---|---|---|---|---|
| Independent | Main Task Only | 16.7 | 13.9 | 53.8 | 52.5 | 1.8s |
| Independent | MAE Joint Training | 16.5 | 13.5 | 53.5 | 52.5 | 1.8s |
| Independent | TTT-MAE No Memory | 35.4 | 20.1 | 53.6 | 52.5 | 3.8s |
| Entire video | Offline TTT-MAE All Frames | 33.6 | 19.6 | 53.2 | 51.2 | 1.8s |
| Stream | LN Adapt | 16.5 | 14.7 | 53.8 | 52.5 | 2.0s |
| Stream | Tent | 16.6 | 14.6 | 53.8 | 52.2 | 2.8s |
| Stream | Tent with Class Balance | 16.7 | 14.8 | 53.8 | 52.5 | 3.7s |
| Stream | Self-Train | - | - | 54.7 | 54.0 | 6.6s |
| Stream | Self-Train with Class Balance | - | - | 54.1 | 53.6 | 6.9s |
| **Stream** | **Online TTT-MAE (Ours)** | **37.6** | **21.7** | **55.4** | **54.3** | 4.1s |

> Time: seconds per frame on a single A100 GPU, averaged over KITTI-STEP test set. COCO Videos times similar, omitted. Self-training not applicable to instance/panoptic (no per-object confidence).

Key observations:
- 2.2x improvement for instance segmentation over Main Task Only on COCO Videos
- 1.5x improvement for panoptic segmentation over Main Task Only on COCO Videos
- Outperforms the offline oracle on all metrics (even though offline was given oracle iteration selection)
- Normalization-layer techniques (LN Adapt, Tent) do not help on real-world video
- Majority vote with augmentation (+1.2% mIoU) and temporal smoothing (+0.4% mIoU) are orthogonal to online TTT, providing roughly the same improvement when combined

### Links

- -> **TTT**: this is the video-stream extension of TTT-MAE
- -> **Implicit Memory**: parameter carry-over between frames
- -> **Explicit Memory**: sliding window of recent frames
- -> **Joint Training**: how $f_0, g_0, h_0$ are obtained before test time
- -> **Locality**: the design principle behind the short window and no-reset policy

---

## 9. Sequence Modelling Perspective

### Definition

Online TTT can be viewed as an RNN where the model parameters $W = \text{params}(f)$ serve as the **hidden state** and gradient descent is the **update rule**. From this perspective, online TTT compresses frames $x_1, \ldots, x_t$ into $W_t$, exactly like an RNN compresses tokens into a hidden state.

```
Input:     x_1        x_2        ...     x_t
            │          │                   │
Hidden:  s_0 ──→ s_1 ──→ s_2 ──→ ... ──→ s_t    Update rule
            │          │                   │
Output:   z_1        z_2        ...      z_t     Output rule
```

### Properties

| Layer | Initial State | Update Rule | Output Rule | Cost |
|---|---|---|---|---|
| Naive RNN | $s_0 = \text{vector}()$ | $s_t = \sigma(\theta_{ss}s_{t-1} + \theta_{sx}x_t)$ | $z_t = \theta_{zs}s_t + \theta_{zx}x_t$ | $O(1)$ |
| Self-attention | $s_0 = \text{list}()$ | $s_t = s_{t-1}.\text{append}(k_t,v_t)$ | $z_t = V_t\text{softmax}(K_t^Tq_t)$ | $O(t)$ |
| **Naive TTT** | $W_0 = f.\text{params}()$ | $W_t = W_{t-1} - \eta\nabla\ell(W_{t-1}; x_t)$ | $z_t = f(x_t; W_t)$ | $O(1)$ |

- RNNs: fixed-size state, constant cost per token, but limited expressiveness
- Self-attention: growing state, growing cost per token, but powerful
- TTT: fixed-size state (the model parameters), constant cost per token, but the state space is the entire parameter space of $f$ -- much larger than a typical RNN hidden state

### Application

Following earlier versions of this paper, Sun et al. (2023) and Sun et al. (2024) programmed TTT into sequence modelling layers as an alternative to self-attention, applying it to language modelling. The TTT layer uses gradient descent as the update rule, achieving $O(1)$ cost per token with an expressive hidden state.

### Links

- -> **Implicit Memory**: the parameter trajectory $W_0 \to W_1 \to \cdots$ is exactly the RNN hidden state
- -> **TTT**: the update rule is one step of gradient descent on $\ell_s$
- -> **In-Context Learning**: an alternative where each video is context to a Transformer/RNN. But this requires the model to be trained on videos; this paper's goal is to study generalization from still images to videos (Brown et al., 2020)
- -> **TTT on Nearest Neighbours**: an alternative heuristic discussed below

### 9.2 TTT on Nearest Neighbours

For each test instance, retrieve its nearest neighbours from a training set and fine-tune on them (Bottou and Vapnik, 1992; Hardt and Sun, 2023). Given temporal smoothness -- proximity in time translates to proximity in the retrieval metric -- the sliding window can be seen as retrieving "neighbours" of the current frame from the **unlabeled test video** instead of a labelled training set. Two consequences: (1) self-supervision must be used (no labels), but (2) the "neighbours" are still relevant even when the test instance is not represented by the training set.

---

## 10. Datasets and Tasks

### 10.1 KITTI-STEP

- 9 validation + 12 test videos of urban driving (Weber et al., 2021)
- 10 fps, up to 106 seconds -- the longest publicly available dataset with dense pixel-wise annotations
- 19 semantic categories (same as CityScapes)
- Joint training on CityScapes (Cordts et al., 2016) still images; KITTI-STEP used only for evaluation
- All hyperparameters selected on KITTI-STEP validation set, then transferred to all other datasets

> **Note**: KITTI-STEP was originally designed for instance-level tracking with a held-out test set. The official website evaluates only tracking-related metrics. The authors perform their own evaluation using segmentation labels. Since they do not train on KITTI-STEP, the training set is used as the test set.

### 10.2 COCO Videos

- 3 egocentric videos (similar to a walking human), each ~5 minutes, professionally annotated in COCO instance/panoptic format (Lin et al., 2014)
- 10,475 frames at 10 fps, 134 classes -- orders of magnitude longer and more diverse than KITTI-STEP
- Each video alone contains more frames than all KITTI-STEP validation videos combined
- Do not follow any tracked object (unlike Oxford Long-Term Tracking or ImageNet-Vid); objects constantly entering/leaving the frame
- Indoor and outdoor (unlike KITTI-STEP and CityScapes which focus on self-driving)
- Mask2Former drops from 44.9 AP / 53.6 PQ (COCO val) to 16.7 AP / 13.9 PQ on COCO Videos: still-image models are fragile on real video
- All COCO Videos results completed in a **single run** with hyperparameters from KITTI-STEP (not tuned on test videos)

### 10.3 Video Colorization

- Goal: add realistic RGB colours to gray-scale images, demonstrating method generality
- Colorization treated as supervised learning (Swin Transformer, two heads, pre-trained on ImageNet to predict colours from gray-scale)
- No domain-specific techniques (perceptual losses, adversarial learning, diffusion)
- Same hyperparameters as segmentation; all colorization experiments in a single run

**Quantitative results (Table 3 -- COCO Videos):**

| Method | FID $\downarrow$ | IS $\uparrow$ | LPIPS $\uparrow$ | PSNR $\uparrow$ | SSIM $\uparrow$ |
|---|---|---|---|---|---|
| Zhang et al. (2016) | 62.39 | $5.00 \pm 0.19$ | 0.180 | 22.27 | **0.924** |
| Main Task Only | 59.96 | $5.23 \pm 0.12$ | 0.216 | 20.42 | 0.881 |
| **Online TTT-MAE** | **56.47** | $\mathbf{5.31 \pm 0.18}$ | **0.237** | **22.97** | 0.901 |

> PSNR and SSIM often misrepresent actual visual quality because colorization is inherently multi-modal (Zhang et al., 2016, 2017).

**Qualitative: Lumiere Brothers Films** -- 10 public-domain black-and-white films from 1895, each ~40 seconds at 10 fps. Online TTT-MAE produces more vibrant and temporally consistent colours than the baseline.

| Dataset | Avg. Length | Frames | FPS | Classes |
|---|---|---|---|---|
| CityScapes-VPS (Kim et al., 2020) | 1.8s | 3,000 | 17 | 19 |
| DAVIS (Pont-Tuset et al., 2017) | 3.5s | 3,455 | 30 | - |
| YouTube-VOS (Xu et al., 2018) | 4.5s | 123,467 | 30 | 94 |
| KITTI-STEP (Weber et al., 2021) | 40s | 8,008 | 10 | 19 |
| **COCO Videos (Ours)** | **350s** | **10,475** | **10** | **134** |

### Links

- -> **Locality**: longer videos (COCO Videos) better showcase locality; short synthetic videos (CityScapes with corruption) do not
- -> **Online TTT-MAE**: the algorithm is evaluated on all three datasets with the same hyperparameters

---

## 11. Baseline Techniques

### 11.0 Additional Non-TTT Baselines

**Alternative architectures**: Mask2Former Swin-S (69M) achieves 53.8% mIoU on KITTI-STEP, outperforming SegFormer B4 (64.1M, 42.0%) and DeepLabV3+/RN101 (62.7M, 53.1%). Confirms the pre-trained model is state-of-the-art.

**Majority vote with augmentation**: 100 augmented predictions per frame, majority vote. Improves Main Task Only by +1.2% mIoU on KITTI-STEP. Orthogonal to online TTT (combining yields same improvement). Not used elsewhere.

**Temporal smoothing**: Averaging predictions across a sliding window. Improves Main Task Only by +0.4% mIoU. Applying to online TTT yields +0.3%. Orthogonal. Not used elsewhere.

### 11.1 Normalization-Layer Techniques

**LN Adapt**: Accumulate layer normalization statistics with a forward pass on each frame. The LN analogue of batch normalization recalculation (Schneider et al., 2020). Result: no improvement on real-world data.

**Tent** (Wang et al., 2020): Optimize only the trainable parameters of normalization layers by minimizing softmax entropy of predictions. Uses implicit and explicit memory in the same loop. Result: no improvement on real-world data. Both techniques help on synthetically corrupted data (Volpi et al., 2022) but not on real videos.

### 11.2 Self-Training

Generate pseudo-labels from model predictions, re-train on high-confidence ones. The paper improves over Volpi et al. (2022) with: (1) proper confidence threshold $\lambda$, (2) 80% pixel masking (vs. 2.5% in Sohn et al., 2020). Still underperforms Online TTT-MAE.

### 11.3 Class Balancing

Heuristic from Volpi et al. (2022): reset model parameters when the predicted class distribution diverges too far from the initial model (likely collapsed). Applied to self-training and Tent; cannot be applied to LN Adapt.

### Links

- -> **TTT**: all baselines are alternative subroutines for updating the model inside the streaming loop
- -> **Online TTT-MAE**: significantly outperforms all baselines

---

## 12. What Existed Before and What This Paper Changes

### 12.1 Prior Approaches and Their Limitations

**TTT on corrupted still images** (Sun et al., 2020): The original TTT framework uses rotation prediction as the self-supervised task and experiments with streaming, but each $x_t$ is drawn independently from the same synthetic corruption. All frames share the same "future", so locality is meaningless. Training on as many $x_t$s as possible is always best.

**TTT-MAE on still images** (Gandelsman et al., 2022): Replaces rotation with masked image reconstruction. Achieves strong results on object recognition. But each test image is treated independently -- parameters are reset after every prediction. The method requires 20 gradient iterations per image. On video, this wastes computation because adjacent frames are almost identical.

**Volpi et al. (2022)**: Experiments in the streaming setting with short clips, but uses synthetic corruptions (CityScapes with Artificial Weather). Each corruption moves all frames into the same "future" (a domain). Their only real-world dataset (CityScapes without corruption) shows only 1.4% improvement. There is no mention of locality.

**Azimi et al. (2022)**: Treats each video as a dataset of unordered frames -- no concept of past vs. future, no temporal order. The same model is used on the entire video. All experiments use artificial corruptions that are i.i.d. across frames.

**Continual learning**: Assumes forgetting is harmful; the oracle has an infinite replay buffer. The standard setting requires performing well on all past tasks, not just the current one.

### 12.2 What This Paper Contributes

**Contribution 1: Online TTT for video streams.** Extends TTT to the streaming setting with implicit + explicit memory. Key finding: only 1 gradient step per frame is needed.

**Contribution 2: Locality as a design principle.** Formalizes why local (online) beats global (offline) with a bias-variance trade-off theorem. Key finding: forgetting is beneficial.

**Contribution 3: COCO Videos dataset.** Orders-of-magnitude longer and more diverse than existing annotated video datasets. Stress-tests locality on challenging real-world video.

### 12.3 Side-by-Side: Prior Art vs. This Paper

| Dimension | TTT-MAE (Gandelsman 2022) | Volpi et al. (2022) | **This Paper** |
|---|---|---|---|
| Test data | Still images | Synthetic video corruptions | **Real-world video streams** |
| Memory | Reset after each image | Implicit only | **Implicit + Explicit** |
| Iterations/instance | 20 | varies | **1** |
| Locality discussed | No | No | **Yes -- the central thesis** |
| Forgetting | Not relevant | Harmful (assumed) | **Beneficial** |
| Tasks | Object recognition | Semantic segmentation | **4 tasks (semantic, instance, panoptic, colorization)** |

### 12.4 The Core Shift in Thinking

Prior TTT work operated in a setting where locality was either irrelevant (synthetic corruptions make all frames equivalent) or unexplored (treating each test image independently). The implicit assumption was that more data is always better for test-time adaptation, consistent with UDA and continual learning paradigms.

This paper inverts that assumption. On real video, recent frames are more relevant than distant ones, and including too much history actively hurts performance. The optimal strategy is a small sliding window -- deliberately forgetting old observations. This connects to neuroscience (Gravitz, 2019): biological forgetting may serve a computational purpose.

The RNN perspective completes the shift: online TTT is not just an adaptation trick but a form of sequence modelling. The model parameters are a hidden state that is updated by gradient descent, compressing the visual stream into a useful representation. This insight later led to TTT layers for language modelling (Sun et al., 2023, 2024).

---

## 13. Quick Reference Card

| # | Finding | Design Choice | Evidence |
|---|---|---|---|
| 1 | Online beats offline | Use streaming, not batch processing | Table 1: Online 37.6 AP vs. Offline 33.6 AP |
| 2 | Forgetting is beneficial | Sliding window, not full history | Figure 4: performance peaks at $k \approx 16$ |
| 3 | 1 iteration suffices | Single gradient step per frame | Implicit memory provides good initialization |
| 4 | Implicit > Explicit memory | Both matter, but carry-over matters more | Table 4: Implicit Only 36.1 vs. Explicit Only 35.7 |
| 5 | Normalization techniques fail on real video | Use MAE reconstruction, not LN/Tent | Table 1: LN Adapt 16.5, Tent 16.6 vs. Online 37.6 |
| 6 | Same hyperparameters transfer across datasets | Tune on KITTI-STEP, apply to COCO Videos | All COCO results from single run |
| 7 | Temporal smoothness is critical | Method requires natural video, not synthetic corruptions | Shuffling drops performance below baseline |
| 8 | Still-image models are fragile on video | TTT is essential for video deployment | Mask2Former: 44.9 AP on COCO val -> 16.7 AP on COCO Videos |
| 9 | Augmentation/smoothing orthogonal to TTT | Can combine for additional gains | Majority vote +1.2%, temporal smoothing +0.4% (both orthogonal) |

---

## 14. Open Questions

1. **What if the video contains scene cuts or fast motion?** Temporal smoothness breaks down at scene boundaries. The paper does not address sudden distribution shifts within a video.
2. **How does window size $k$ interact with frame rate?** At higher fps, the same temporal duration requires larger $k$. Is there an optimal *temporal duration* rather than frame count?
3. **Can the self-supervised task be improved?** MAE reconstruction is one option; the paper notes that any technique that does not use ground truth labels can replace it inside the loop.
4. **Scaling to higher-resolution or longer videos?** COCO Videos are 5 minutes each. What about hours-long surveillance or dashcam footage?
5. **Relationship to TTT layers in LLMs**: Sun et al. (2024) later formalized TTT as a sequence modelling layer. Can the video-stream insights (locality, window size) transfer to language?
6. **Can explicit memory be more than a sliding window?** The paper uses uniform sampling. Relevance-weighted sampling (like attention over past frames) could improve the bias-variance trade-off.

---

## 15. Concept Dependency Graph

```
                    Temporal Smoothness
                    (x_t ≈ x_{t+1})
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     Implicit         Explicit       Bias-Variance
     Memory           Memory          Trade-off
     (carry-over)     (window k)     (Theorem 1)
          │               │               │
          └───────┬───────┘               │
                  ▼                       │
            Online TTT-MAE ◄──────────────┘
                  │
          ┌───────┼───────────────┐
          ▼       ▼               ▼
       TTT    Joint           Locality
       (Y-arch) Training      (central thesis)
          │                       │
          ▼                       ▼
     TTT-MAE                 Continual Learning
     (Gandelsman             (forgetting can help)
      2022)                       │
          │                       ▼
          ▼                    UDA
     Sequence              (domain view
     Modelling              is misleading)
     (RNN view)
```

---

## 16. Key Equations

| # | Equation | Description | Section |
|---|---|---|---|
| Eq.1 | $f', g' = \arg\min_{f,g} \ell_s(g \circ f(\bar{x}'), x')$ | TTT-MAE one-sample problem | 3.2 |
| Eq.2 | $f_t, g_t = \arg\min_{f,g} \frac{1}{k}\sum_{t'=t-k+1}^{t} \ell_s(g \circ f(\bar{x}_{t'}), x_{t'})$ | Online TTT with explicit memory | 4.2 |
| Eq.3 | $\nabla \ell_m^t(\theta) := \nabla_\theta \ell_m(x_t, y_t; \theta)$ | Main task gradient notation | 6.2 |
| Eq.4 | $\nabla \ell_s^t(\theta) := \nabla_\theta \ell_s(x_t; \theta)$ | Self-supervised gradient notation | 6.2 |
| Eq.5 | $\frac{1}{k}\sum_{t'=t-k+1}^{t} \nabla \ell_s^{t'}$ | Windowed gradient for TTT | 6.2 |
| Thm | $\mathbb{E}[\ell_m(\bar{\theta}) - \ell_m(\theta^*)] \leq \frac{1}{2\alpha}(k^2\beta^2\eta^2 + \frac{1}{k}\sigma^2)$ | Bias-variance upper bound | 6.2 |
| $k^*$ | $k^* = (\sigma^2/\beta^2\eta^2)^{1/3}$ | Optimal window size | 6.2 |
| Lemma | $f(\tilde{x}^*) - f(x^*) \leq \frac{1}{2\alpha}\|v\|^2$ | Perturbation bound for strongly convex $f$ | App. C |
| JT | $\arg\min_{g,h,f} \frac{1}{n}\sum[\ell_m(h \circ f(x_i), y_i) + \ell_s(g \circ f(\bar{x}_i), x_i)]$ | Joint training objective | 4.1 |

---

## 17. Reference Map

**TTT Foundations**
- Bottou and Vapnik (1992): Local Learning -- the earliest TTT idea
- Sun et al. (2020): Original TTT framework with rotation prediction (Gidaris et al., 2018)
- Gandelsman et al. (2022): TTT-MAE (masked autoencoder for TTT)
- Hardt and Sun (2023): TTT on nearest neighbours for LLMs

**Transductive Learning**
- Joachims (2002), Collobert et al. (2006), Vapnik (2013): Transductive SVMs
- Gammerman et al. (1998), Vapnik and Kotz (2006): Principle of transduction

**TTT on Video**
- Volpi et al. (2022): Streaming TTT with synthetic corruptions (CityScapes with Artificial Weather)
- Azimi et al. (2022): TTT on videos as unordered frame datasets
- Mullapudi et al. (2018): Student-teacher distillation on video streams (inspiration)

**TTT for Specific Vision Applications**
- Jain and Learned-Miller (2011), Shocher et al. (2018), Nitzan et al. (2022), Xie et al. (2023): Various applications
- Tonioni et al. (2019a,b), Zhang et al. (2020), Zhong et al. (2018), Luo et al. (2020): Depth estimation

**Continual Learning**
- Kirkpatrick et al. (2017): EWC -- overcoming catastrophic forgetting
- Li and Hoiem (2017): Learning without forgetting
- Lopez-Paz and Ranzato (2017): Gradient episodic memory
- Gravitz (2019): The importance of forgetting (neuroscience)
- Hassabis et al. (2017), De Lange et al. (2021): Human memory and generalization

**Self-Supervision / MAE**
- He et al. (2021): Masked Autoencoders (MAE)
- Vincent et al. (2008): Denoising autoencoders
- Pathak et al. (2016): Context encoders (inpainting)
- Bao et al. (2021), Xie et al. (2022): BEIT, SimMIM

**Architecture**
- Cheng et al. (2021): Mask2Former
- Liu et al. (2021c): Swin Transformer
- Dosovitskiy et al. (2020): Vision Transformer (ViT)

**Domain Adaptation**
- Schneider et al. (2020): BN recalculation for corruption robustness
- Wang et al. (2020): Tent -- entropy minimization at test time
- Hendrycks and Dietterich (2019): Corruption benchmarks

**Sequence Modelling / TTT Layers**
- Sun et al. (2023): TTT as sequence modelling (language)
- Sun et al. (2024): TTT layers with expressive hidden states
- Brown et al. (2020): In-context learning (GPT-3)

**Colorization**
- Zhang et al. (2016, 2017): Image colorization baselines
- Lei and Chen (2019): Video colorization

**Datasets**
- Weber et al. (2021): KITTI-STEP
- Cordts et al. (2016): CityScapes
- Lin et al. (2014): COCO
- Kim et al. (2020): CityScapes-VPS
- Pont-Tuset et al. (2017): DAVIS
- Xu et al. (2018): YouTube-VOS

**Theory**
- Allen-Zhu et al. (2019): Convergence theory (strong convexity of neural nets)
- Zhong et al. (2017): Recovery guarantees for neural nets
- Bubeck et al. (2015): Convex optimization (used in the proof)
