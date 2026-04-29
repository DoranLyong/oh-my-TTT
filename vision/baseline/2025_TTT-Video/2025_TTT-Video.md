# One-Minute Video Generation with Test-Time Training

> **Source**: *CVPR 2025* (Dalal\*, Koceja\*, Hussein\*, Xu\*, Zhao, Song, Han, Cheung, Kautz, Guestrin, Hashimoto, Koyejo, Choi, Sun, Wang), pp. 1–14
> **Authors**: Karan Dalal\*, Daniel Koceja\*, Gashon Hussein\*, Jiarui Xu\*, Yue Zhao, Youjin Song, Shihao Han, Ka Chun Cheung, Jan Kautz, Carlos Guestrin, Tatsunori Hashimoto, Sanmi Koyejo, Yejin Choi, Yu Sun, Xiaolong Wang
> **Affiliations**: NVIDIA, Stanford University, UCSD, UC Berkeley, UT Austin
> **Project**: https://test-time-training.github.io/video-dit
> **arXiv**: 2504.05298v1 [cs.CV]

---

## This Paper Covers
- Using **TTT layers** (whose hidden states are neural networks) as an alternative to self-attention and linear RNN layers for long-context video generation
- Adding TTT layers to a pre-trained Diffusion Transformer (CogVideo-X 5B) and fine-tuning it to generate **one-minute videos** from text storyboards
- Demonstrating that TTT-MLP generates much more coherent videos with complex, multi-scene stories than Mamba 2, Gated DeltaNet, and sliding-window attention baselines
- Curating a text-to-video dataset based on ~7 hours of *Tom and Jerry* cartoons with human-annotated storyboards
- Developing **on-chip Tensor Parallel** for efficient TTT-MLP inference on GPUs

> **Note**: Results still contain artifacts, likely due to the limited capability of the pre-trained 5B model. The efficiency of the implementation can also be improved. The authors have only experimented with one-minute videos due to resource constraints, but the approach can be extended to longer videos and more complex stories.

---

## Overview

State-of-the-art video Transformers still generate mostly short clips of single scenes without complex stories. At the time of writing (March 2025), public APIs for video generation max out at 20 seconds (Sora/OpenAI), 16 seconds (MovieGen/Meta), 10 seconds (Ray 2/Luma), and 8 seconds (Veo 2/Google). None autonomously generate complex multi-scene stories.

The fundamental challenge is **long context**: self-attention cost grows quadratically with context length. For one-minute videos, each requires over 300k tokens in context. With self-attention, generating a one-minute video would take $11\times$ longer than twenty 3-second videos, and training would take $12\times$ longer.

Modern RNN layers (Mamba, DeltaNet) offer linear-complexity alternatives, but their hidden states are **less expressive**: they compress hundreds of thousands of vectors into a matrix with only thousands in rank, making it inherently challenging to remember deep relationships between distant tokens.

This paper experiments with **TTT layers** -- RNN layers whose hidden states are themselves neural networks (specifically two-layer MLPs). These hidden states have $2\times$ more hidden cells and richer nonlinearities than linear attention variants. Since the neural network hidden states are updated by training even on test sequences, these layers are called Test-Time Training (TTT) layers.

The *Tom and Jerry* cartoon domain is intentionally limited in scope for fast research iteration. The dataset emphasizes complex, multi-scene, long-range stories with dynamic motion (where progress is still needed), with less emphasis on visual and physical realism (where remarkable progress has already been made). The authors believe that improvements in long-context capabilities for this domain will transfer to general-purpose video generation.

> **Figure 1**: TTT layers enable a pre-trained Diffusion Transformer to generate one-minute videos from text storyboards. The videos tell complex stories with coherent scenes composed of dynamic motion, produced directly in a single shot without editing, stitching, or post-processing. *Tom and Jerry* cartoons are used as proof of concept.

---

## 2. Test-Time Training Layers

### 2.1 TTT as Updating a Hidden State

All RNN layers compress historical context into a hidden state of fixed size. The goal of Sun et al. (2024) is to design RNN layers with **expressive hidden states** that can compress massive context. The key idea: use self-supervised learning to compress the historical context $x_1, \ldots, x_t$ into a hidden state $W_t$, by making the context an unlabeled dataset and the hidden state the weights of a machine learning model $f$.

The update rule is a gradient step on a self-supervised loss $\ell$:

$$W_t = W_{t-1} - \eta \,\nabla \ell(W_{t-1}; x_t), \tag{1}$$

with learning rate $\eta$. The output token is the prediction on $x_t$ made by $f$ with the updated weights:

$$z_t = f(x_t; W_t). \tag{2}$$

One choice of $\ell$ is reconstructing $x_t$ itself. To make the learning problem nontrivial, $x_t$ is first processed into a corrupted input $\tilde{x}_t$, then optimize:

$$\ell(W; x_t) = \|f(\tilde{x}_t; W) - x_t\|^2. \tag{3}$$

Similar to denoising autoencoders, $f$ needs to discover correlations between dimensions of $x_t$ to reconstruct it from partial information $\tilde{x}_t$.

This algorithm maps input sequence $x_1, \ldots, x_T$ to output sequence $z_1, \ldots, z_T$ and can be programmed into the forward pass of a sequence modelling layer. Even at test time, the layer trains a different sequence of weights $W_1, \ldots, W_T$ for every input sequence -- hence the name **Test-Time Training (TTT) layer**.

Calling backward on $\nabla \ell$ means taking gradients of gradients -- a well-explored technique in meta-learning. TTT layers have the same interface as RNN layers and self-attention, so they can be replaced in any larger network architecture. Sun et al. (2024) refers to training the larger network as the **outer loop**, and training $W$ within each TTT layer as the **inner loop**.

> **Figure 2**: All RNN layers can be expressed as a hidden state transitioning according to an update rule. The key idea is to make the hidden state itself a model $f$ with weights $W$, and the update rule a gradient step on the self-supervised loss $\ell$. (Figure from Sun et al., 2024.)

### 2.2 Learning a Self-Supervised Task for TTT

Instead of handcrafting a self-supervised task from human priors, Sun et al. (2024) take an end-to-end approach, learning it as part of the outer loop. Starting from the naive reconstruction task in Eq. 3, they use a low-rank projection $\tilde{x}_t = \theta_K x_t$, where $\theta_K$ is a learnable matrix in the outer loop.

Moreover, the reconstruction label can also be a low-rank projection $\theta_V x_t$ instead of $x_t$. The self-supervised loss becomes:

$$\ell(W; x_t) = \|f(\theta_K x_t; W) - \theta_V x_t\|^2. \tag{4}$$

Since $\theta_K x_t$ has fewer dimensions than $x_t$, the output rule also changes to use another projection $\theta_Q x_t$:

$$z_t = f(\theta_Q x_t; W_t). \tag{5}$$

In the inner loop, only $W$ is optimized; the $\theta$s are "hyperparameters" of the inner-loop loss function. $\theta_K, \theta_V, \theta_Q$ are optimized in the outer loop, analogous to the Query, Key, and Value parameters of self-attention.

### 2.3 TTT-MLP Instantiation

Following Sun et al. (2024), the inner-loop model $f$ is instantiated as $f_{\text{MLP}}$: a two-layer MLP similar to those in Transformers. The hidden dimension is $4\times$ the input dimension, followed by a GELU activation. For better stability during TTT, $f$ always contains a Layer Norm and residual connection:

$$f(x) = x + \text{LN}(f_{\text{MLP}}(x)).$$

A TTT layer with this $f$ is called **TTT-MLP**, the default instantiation throughout this paper. **TTT-Linear** (where $f$ wraps a linear model instead of an MLP) is also tested as a baseline.

---

## 3. Approach

At a high level, the approach simply adds TTT layers to a pre-trained Diffusion Transformer and fine-tunes it on long videos with text annotations.

### 3.1 Architecture

**Pre-trained Diffusion Transformer.** The approach of adding TTT layers then fine-tuning can, in principle, work with **any backbone architecture**. Diffusion Transformers are chosen because they are the most popular architecture for video generation. Since pre-training cost is prohibitive, the paper starts from a pre-trained checkpoint called **CogVideo-X 5B** that can generate 3-second short clips at 16 fps (or 6 seconds at 8 fps). TTT layers are initialized from scratch and fine-tuned to generate one-minute videos from text storyboards. Self-attention layers are limited to 3-second segments to keep cost manageable. The training run takes the equivalent of 50 hours on 256 H100s.

**Gating.** Naively inserting TTT layers into a pre-trained network would worsen predictions at the beginning of fine-tuning (random initialization). To avoid this, TTT is gated with a learned vector $\alpha \in \mathbb{R}^d$ following standard practice:

$$\text{gate}(\text{TTT}, X; \alpha) = \tanh(\alpha) \otimes \text{TTT}(X) + X, \tag{6}$$

where $\tanh(\alpha) \in (-1, 1)^d$ is multiplied element-wise with each $z_t$ in $Z = \text{TTT}(X)$. All values in $\alpha$ are initialized to 0.1, so $\tanh(\alpha) \approx 0$ at the start of fine-tuning. This allows TTT to contribute to $\text{gate}(\text{TTT}, X; \alpha)$ without significantly overwriting $X$.

**Bi-direction.** Diffusion models, including CogVideo-X, are non-causal (an output $z_t$ can condition on all of $x_1, \ldots, x_T$, not only past tokens). To use TTT layers in a non-causal manner, the standard trick called **bi-direction** is applied. Given an operator $\text{rev}(X)$ that reverses the sequence in time:

$$\text{TTT}'(X) = \text{rev}(\text{TTT}(\text{rev}(X))). \tag{7}$$

Since $\text{rev}$ is applied twice, $\text{TTT}'(X)$ is still in chronological order, but the TTT layer inside now scans through $X$ in reverse-chronological order.

**Modified architecture.** Standard Transformers contain interleaving sequence modelling blocks and MLP blocks. Specifically, a standard block takes input $X$ and produces:

$$X' = \text{self\_attn}(\text{LN}(X)) \tag{8}$$
$$Y = X' + X. \tag{9}$$

The paper modifies the sequence modelling blocks only, adding a TTT layer with a learnable gate after each attention layer. Each modified block continues from $X'$ in Eq. 8 and produces:

$$Z = \text{gate}(\text{TTT}, X'; \alpha), \tag{10}$$
$$Z' = \text{gate}(\text{TTT}', Z; \beta), \tag{11}$$
$$Y = Z' + X. \tag{12}$$

Note that $\text{TTT}'$ only makes another call to TTT, so they share the same underlying parameters $\theta_K, \theta_V, \theta_Q$. But for gating, Equations 10 and 11 use different parameters $\alpha$ and $\beta$.

> **Figure 3**: **Left**: Modified architecture adds a TTT layer with a learnable gate after each attention layer. **Right**: The overall pipeline creates input sequences composed of 3-second segments. Self-attention layers attend locally over segments, and TTT layers attend globally over the entire sequence.

### 3.2 Overall Pipeline

**Scenes and segments.** Videos are structured to contain multiple scenes, each containing one or more **3-second segments**. The 3-second segment is the atomic unit of text-to-video pairing for three reasons:
1. The maximum generation length of the original CogVideo-X is 3 seconds
2. The length of most scenes in *Tom and Jerry* episodes is at least 3 seconds
3. Building a multi-stage dataset is most convenient given 3-second segments

**Formats of text prompts.** At inference time, a user can write the text prompt in any of three formats of increasing detail:
- **Format 1**: A short summary of the plot in 5-8 sentences
- **Format 2**: A more detailed plot in ~20 sentences, with each sentence roughly corresponding to a 3-second segment. Sentences can be labelled as belonging to certain scenes
- **Format 3**: A storyboard. Each 3-second segment is described by a paragraph of 3-5 sentences containing details such as background colours and camera movements. Groups of paragraphs are enforced as belonging to certain scenes with `<scene start>` and `<scene end>` keywords

The actual input to the text tokenizer is always Format 3 during both fine-tuning and inference. Conversion between formats is performed by Claude 3.7 Sonnet in the order $1 \to 2 \to 3$ (converting Format 1 directly to Format 3 results in worse ability to follow the style of the human annotations in the fine-tuning dataset).

**From text to sequences.** Given a storyboard in Format 3 with $n$ paragraphs, $n$ sequence segments are produced, each containing text tokens from the corresponding paragraph followed by video tokens. All $n$ segments are concatenated to form the input sequence, which now has interleaved text and video tokens.

**Local attention, global TTT.** CogVideo-X uses self-attention layers to process the entire input sequence globally for each video of maximum length 3 seconds, but global attention becomes inefficient for long videos. Self-attention is made **local** to each 3-second segment, attending independently. The TTT layers process the entire input sequence **globally** because they are efficient in long context. (As an artifact of pre-processing, the sequence segments have an overlap of 1 latent frame, i.e., 1350 tokens.)

### 3.3 Fine-Tuning Recipe and Dataset

**Multi-stage context extension.** Following standard practice for LLMs, context length is extended to one minute in **five stages**:
1. First, fine-tune the entire pre-trained model on 3-second segments of *Tom and Jerry*. New parameters (TTT layers and gates) get a higher learning rate
2-5. Over the next four stages, fine-tune on videos of 9, 18, 30, and eventually 63 seconds. To avoid forgetting world knowledge from pre-training, only the TTT layers, gates, and self-attention layers are fine-tuned with a lower learning rate

**Super-resolution on original videos.** Start with 81 episodes of *Tom and Jerry* released between 1940 and 1948 (~5 minutes each, ~7 hours total). A video super-resolution model is applied to produce visually enhanced videos with shared resolution of $720 \times 480$.

**Multi-stage dataset.** Following the pipeline structure:
1. Human annotators break down each episode into scenes
2. Extract 3-second segments from each scene
3. Human annotators write a detailed paragraph for each 3-second segment (each paragraph: 1-2 sentences background, 1-2 sentences characters, 2 sentences actions/camera movements; average 98 words = 132 tokens)
4. Stage 1 fine-tunes directly on these segments
5. For later stages, concatenate contiguous 3-second segments into videos of 9, 18, 30, and 63 seconds with their text annotations. Scene boundaries are marked by keywords

| Video len. | Ctx. len | Trainable parameters | Learning rate | Schedule | Steps |
|---|---|---|---|---|---|
| 3 sec | 18,048 | TTT / Pre-trained Params | $1 \times 10^{-4}$ / $1 \times 10^{-5}$ | Cosine / Constant | 5,000 |
| 9 sec | 51,456 | TTT + Local Attn (QKVO) | $1 \times 10^{-5}$ | Constant | 5,000 |
| 18 sec | 99,894 | TTT + Local Attn (QKVO) | $1 \times 10^{-5}$ | Constant | 1,000 |
| 30 sec | 168,320 | TTT + Local Attn (QKVO) | $1 \times 10^{-5}$ | Constant | 500 |
| 63 sec | 341,550 | TTT + Local Attn (QKVO) | $1 \times 10^{-5}$ | Constant | 250 |

> **Table 2**: Hyper-parameters for multi-stage fine-tuning. First, the entire pre-trained model is fine-tuned on 3-second segments with higher learning rates for newly introduced TTT layers and gates. Then, only TTT layers, gates, and self-attention parameters are fine-tuned at reduced learning rates.

### 3.4 Parallelization for Non-Causal Sequences

The update rule in Section 2 cannot be naively parallelized across tokens because computing $W_t$ requires $\nabla \ell(W_{t-1}; x_t)$, which requires $W_{t-1}$. To enable parallelization, $W$ is updated on $b$ tokens at a time, called an **inner-loop mini-batch**. Throughout this paper, $b = 64$.

Concretely, for mini-batch $i = 1, \ldots, T/b$ (assuming $T$ is an integer multiple of $b$):

$$W_{ib} = W_{(i-1)b} - \frac{\eta}{b} \sum_{t=(i-1)b+1}^{ib} \nabla \ell\left(W_{(i-1)b}; x_t\right). \tag{13}$$

Because the sequence is non-causal, $W_{ib}$ is used to produce the output tokens for all timesteps in mini-batch $i$:

$$z_t = f(W_{ib}; x_t), \qquad \text{for } t = (i-1)b+1, \ldots, ib. \tag{14}$$

Note that $W_{(i-1)b+1}, \ldots, W_{ib-1}$ are no longer needed. After this modification, $f$ can process an (inner-loop) mini-batch of tokens in parallel, similar to how a regular MLP processes an (outer-loop) mini-batch of training data. Averaging gradients across tokens reduces variance and stabilizes each update to $W$.

### 3.5 On-Chip Tensor Parallel

Implementing TTT-MLP efficiently on GPUs requires exploiting the memory hierarchy. Each Streaming Multiprocessor (SM) has fast but small on-chip SMEM and shares relatively slow but large global HBM. Frequent data transfers between SMEM and HBM hurt efficiency.

Efficient implementations of Mamba and self-attention (Flash Attention) use kernel fusion to minimize this transfer: load inputs into SMEM, compute entirely on-chip, write only final outputs to HBM. However, the TTT-MLP hidden state ($W^{(1)}$ and $W^{(2)}$ of the two-layer MLP $f$) is too large to fit in a single SM's SMEM.

**Solution: Tensor Parallelism across SMs.** Using Tensor Parallelism (Shoeybi et al., 2019), $W^{(1)}$ and $W^{(2)}$ are sharded across SMs. The hidden state is updated entirely on-chip, using the **DSMEM** feature on NVIDIA Hopper GPU architecture to implement AllReduce among SMs.

> **Figure 4**: **Left**: Shard the hidden state $W^{(1)}$ and $W^{(2)}$ across SMs, transferring them between HBM and SMEM only during initial loading and final output. **Right**: Update the hidden state entirely on-chip and use DSMEM to AllReduce intermediate activations among SMs.

The implementation uses ThunderKittens (Spector et al., 2025) for the TTT-MLP kernel. Additional optimizations include multi-stage producer-consumer asynchrony for pipelining, and gradient checkpointing along the sequence dimension integrated into the fused kernel with TMA for asynchronous memory stores.

**General principle**: if a model architecture $f$ can be sharded with standard Tensor Parallelism across GPUs, then the same sharding strategy can be applied across SMs when $f$ is used as the hidden state. This makes the on-chip approach applicable to any TTT instantiation.

---

## 4. Evaluation

Human evaluation on a multi-axis benchmark for TTT-MLP and five baselines, all with linear complexity: local attention, TTT-Linear, Mamba 2, Gated DeltaNet, and sliding-window attention layers.

### 4.1 Baselines

All baselines are added to the same pre-trained CogVideo-X 5B using the approach in Section 3.1; all modified architectures have **7.2B parameters**. All baselines use the same fine-tuning recipe (Section 3.3 and Appendix A):

- **Local attention**: No modification to the original architecture, performing self-attention on each 3-second segment independently
- **TTT-Linear**: A TTT layer where $f(x) = x + \text{LN}(f_{\text{Linear}}(x))$, with $f_{\text{Linear}}$ being a linear model
- **Mamba 2**: A modern RNN layer with a matrix hidden state, $\approx 4\times$ larger than the hidden state in TTT-Linear but $\approx 2\times$ smaller than in TTT-MLP
- **Gated DeltaNet** [53]: An extension of DeltaNet [52] and Mamba 2 with an improved update rule
- **Sliding-window attention**: Self-attention with a fixed window of 8192 tokens (about 1.5 seconds of video)

### 4.2 Evaluation Axes and Protocol

Four evaluation axes adopted from MovieGen (omitting "realness" and "motion completeness" which don't apply to cartoons):

- **Text following**: "alignment with the provided prompt"
- **Motion naturalness**: "natural limb movements, facial expressions, and adherence to physical laws"
- **Aesthetics**: "interesting and compelling content, lighting, color, and camera effects"
- **Temporal consistency**: both inside and across scenes (adapted from MovieGen's "frame consistency" to also include cross-scene consistency)

Evaluation is based on **pairwise preferences in blind comparisons**. An evaluator is given a random axis and a random pair of videos sharing the same plot, then asked to indicate the better video. 100 plots are sampled using Claude 3.7 Sonnet (Format $1 \to 2 \to 3$), then one video is generated per method per plot. Methods are always unknown to evaluators.

Evaluators recruited on prolific.com: living in U.S., English first language, aged 18-35, at least 100 previous submissions, 98%+ approval rate. Demographics: 50.78% male, 47.66% female; 57.03% White, 23.44% Black, 10.94% Mixed.

### 4.3 Results

Elo scores aggregated using the LMSys Chatbot Arena system:

| | Text following | Motion naturalness | Aesthetics | Temporal consistency | **Average** |
|---|---|---|---|---|---|
| Mamba 2 | 985 | 976 | 963 | 988 | 978 |
| Gated DeltaNet | 983 | 984 | 993 | 1004 | 991 |
| Sliding window | **1016** | 1000 | 1006 | 975 | 999 |
| **TTT-MLP** | 1014 | **1039** | **1037** | **1042** | **1033** |

> **Table 1**: Human evaluation results for one-minute videos. TTT-MLP improves over the second best method by **34 Elo points** on average. Axes with the most improvement are scene consistency (+38) and motion smoothness (+39). For context, GPT-4 scores 46 Elo points over GPT-3.5 Turbo, and GPT-4o scores 29 over GPT-4 Turbo in Chatbot Arena.

> **Figure 5**: Video frames comparing TTT-MLP against baselines. **TTT-MLP** preserves temporal consistency over scene changes and across angles, producing smooth, high-quality actions. **Sliding-window attention** alters environments and duplicates events. **Gated DeltaNet** lacks temporal consistency across different angles. **Mamba 2** distorts character appearance.

**18-second elimination round.** Local attention and TTT-Linear were eliminated in an initial round using 18-second videos (context ~100k tokens). Gated DeltaNet performs best on 18-second videos, leading Mamba 2 by 27 Elo points and TTT-MLP by 28. At ~100k token context, RNN layers with linear (matrix) hidden states are still the most effective. But for 63-second videos (341k tokens), TTT-MLP's advantage emerges.

| | Text following | Motion naturalness | Aesthetics | Temporal consistency | **Average** |
|---|---|---|---|---|---|
| Local Attention | 965 | 972 | 969 | 944 | 962 |
| TTT-Linear | 1003 | 995 | 1007 | 1001 | 1001 |
| Mamba 2 | **1023** | 987 | 1008 | 1004 | 1005 |
| Gated DeltaNet | 1020 | **1039** | **1044** | 1026 | **1032** |
| SWA | 995 | 1004 | 993 | 980 | 993 |
| TTT-MLP | 994 | 1002 | 1002 | 1019 | 1004 |

> **Table 3**: Human evaluation results for 18-second videos. Gated DeltaNet leads at this shorter context length.

**Efficiency.** For 63-second videos (Figure 6):
- Inference with full attention (over 300k tokens) would take $11\times$ longer than local attention, and training $12\times$ longer
- TTT-MLP takes $2.5\times$ and $3.8\times$ respectively -- significantly more efficient than full attention
- But TTT-MLP is still $1.4\times$ (inference) and $2.1\times$ (training) slower than Gated DeltaNet
- Gated DeltaNet is $1.8\times$ longer than local attention in both inference and training

### 4.4 Limitations

**Short context.** For 18-second videos (~100k tokens), Gated DeltaNet performs best. RNN layers with linear (matrix) hidden states are still the most effective at this context length. TTT-MLP's advantage emerges at longer contexts.

**Wall-clock time.** Even with on-chip Tensor Parallel (Section 3.5), TTT-MLP is still slower than Gated DeltaNet and Mamba 2. The current TTT-MLP kernel is bottlenecked by register spills and suboptimal ordering of asynchronous instructions. Note that training efficiency is not a significant concern because the RNN layers are integrated after pre-training (which constitutes most of the overall training budget). Training efficiency of the RNN layers is only relevant during fine-tuning, a small part of the budget. **Inference efficiency is much more meaningful.**

**Video artifacts.** The generated 63-second videos contain notable artifacts, but these are **not particular to TTT-MLP** -- they are common among all methods. The artifacts are likely a consequence of the limited capability of the pre-trained CogVideo-X 5B model (videos generated by the original CogVideo-X also show limited motion naturalness and aesthetics). Specific artifacts include:
- **Temporal consistency**: objects morph at 3-second segment boundaries, potentially because the diffusion model samples from different modes across segments
- **Motion naturalness**: objects sometimes float unnaturally (gravitational effects not properly modelled)
- **Aesthetics**: lighting changes do not consistently align with actions unless explicitly prompted; complex camera movements (parallax) sometimes inaccurate

> **Figure 7**: Artifact examples. Temporal consistency: boxes morph between 3-second segments of the same scene. Motion naturalness: cheese hovers in mid-air rather than falling naturally to the ground. Aesthetics: kitchen lighting becomes dramatically brighter as Tom turns around.

---

## 5. Related Work

**Modern RNN layers**: Linear attention variants (Katharopoulos et al., 2020; Schmidhuber, 1992) such as Mamba (Dao, 2024; Gu and Dao, 2024) and DeltaNet (Schlag et al., 2021; Yang et al., 2024). Inspired by Fast Weight Programmers (Clark et al., 2022; Irie et al., 2021; Kirsch and Schmidhuber, 2021; Schmidhuber, 1992), Sun et al. (2024) proposes scalable and practical ways to make hidden states large and nonlinear. Recent work (Behrouz et al., 2024) develops even larger and more nonlinear hidden states.

**Long video modelling**: Early work uses GANs (Goodfellow et al., 2020; Karras et al., 2020) to predict next frames. Recent progress in auto-regression and diffusion-based approaches (Gupta et al., 2023; Hunyuanvideo 2025; Yang et al., 2025; CogVideoX 2025). TATS (Ge et al., 2022) proposes sliding-window attention on the Transformer to generate videos longer than training length. Phenaki (Villegas et al., 2023) works in a similar auto-regressive way. Pre-trained diffusion models can be extended with cascade (He et al., 2022; Lavie 2024; Nuwa-xl 2023), streaming (Streamingt2v 2024), and transitions (Seine 2023).

**Story synthesis**: Methods such as StoryGAN (Li et al., 2019), Make-a-Story (Rahman et al., 2023), Craft (Schwenk et al., 2018), Grimm (Liu et al., 2024), and StoryDiffusion (Zhou et al., 2024) generate sequences of images or videos from text stories. While related, these methods usually need additional components for coherence across scenes and are not processed end-to-end.

---

## 6. Future Work

**Faster implementation.** The current TTT-MLP kernel is bottlenecked by register spills and suboptimal ordering of asynchronous instructions. Efficiency could be improved by minimizing register pressure and developing a more compiler-aware implementation.

**Better integration.** Bi-direction and learned gates are only one strategy for integrating TTT layers into a pre-trained model. Other video generation backbones (e.g., autoregressive models) might require different integration strategies.

**Longer videos with larger hidden states.** The approach can potentially extend to much longer videos with linear complexity. The key is to instantiate hidden states as much larger neural networks -- for example, $f$ itself can be a Transformer.

---

## Appendix A: Experiment Details

**Diffusion schedule.** Following CogVideoX, fine-tune using v-prediction with a diffusion noise schedule of 1000 steps and ZeroSNR enforced at the final step.

**Training configurations:**
- **Optimizer**: AdamW with $(\beta_1, \beta_2) = (0.9, 0.95)$
- **Learning Rate**: Linear warmup over 2% of training steps
- **Batch Size**: 64
- **Gradient Clipping**: 0.1
- **Weight Decay**: $10^{-4}$ applied to all params except biases and normalization layers
- **VAE Scale Factor**: 1.0
- **Dropout**: Zero-out text prompt with probability 0.1
- **Precision**: Mixed Precision with PyTorch FSDP2

**TTT configurations.** The key hyperparameter is the inner-loop learning rate $\eta$: set to $\eta = 1.0$ for TTT-Linear and $\eta = 0.1$ for TTT-MLP.

**Sampling schedule.** DDIM sampler with 50 steps, applying dynamic classifier-free guidance (CFG) that increases CFG magnitude from 1 to 4 and utilizing negative prompts to further enhance video quality.

---

## Appendix B: On-Chip Tensor Parallel Details

**Hidden state sharding.** Follow the standard strategy for Tensor Parallel: shard the first layer column-wise and the second layer row-wise. The GeLU non-linearity is element-wise, so the forward pass of the TTT-layer requires a single reduction for computing the inner loss used to update the hidden state.

**Further latency optimizations.** Multi-stage pipelining scheme that asynchronously prefetches future mini-batches from HBM, overlapping data transfers with computation on the current mini-batch. This **producer-consumer asynchrony** involves dedicating specialized warpgroups to either data loading (producer) or computation (consumer).

**Gradient checkpointing.** Integrated along the sequence dimension directly into the fused kernel. TMA (Tensor Memory Accelerator) is used for asynchronous memory stores.

---

## Summary
- **TTT layers** use neural networks as hidden states, providing richer expressiveness than linear (matrix) hidden states in Mamba/DeltaNet for compressing long-context video
- Adding TTT-MLP layers to CogVideo-X 5B enables **one-minute video generation** from text storyboards -- the first demonstration of autonomous, multi-scene, story-driven video generation
- TTT-MLP improves over the second-best method by **34 Elo points** in human evaluation (comparable to GPT-4o over GPT-4 Turbo's 29 Elo points in LMSys)
- The largest gains are in **temporal consistency** (+38) and **motion naturalness** (+39) -- precisely the axes that require remembering long-range context
- At shorter contexts (18 seconds, ~100k tokens), Gated DeltaNet performs best; TTT-MLP's advantage emerges at **one-minute scale** (341k tokens)
- **On-chip Tensor Parallel** shards the TTT-MLP hidden state across SMs, keeping computation in fast SMEM and avoiding costly HBM transfers
- TTT-MLP is still $1.4\times$/$2.1\times$ slower than Gated DeltaNet for inference/training -- a key limitation for future work
- The *Tom and Jerry* dataset with human-annotated storyboards provides a controlled testbed for long-context video generation research
- Future directions include faster TTT-MLP kernels, better integration strategies for pre-trained models, and scaling to even longer videos with larger hidden states (e.g., $f$ as a Transformer)

---

## References and Further Reading
- Project website: https://test-time-training.github.io/video-dit
- Sun et al. (2024). Learning to (learn at test time): RNNs with expressive hidden states. *arXiv:2407.04620*.
- Peebles and Xie (2023). Scalable diffusion models with transformers. *CVPR*.
- Hong et al. (2023). CogVideo: Large-scale pretraining for text-to-video generation. *ICLR*.
- Yang et al. (2025). CogVideoX: Text-to-video diffusion models with an expert transformer. *ICLR*.
- Dao (2024). Transformers are SSMs. *ICML*.
- Yang et al. (2025). Gated delta networks. *ICLR*.
- Spector et al. (2025). ThunderKittens: Simple, fast, and adorable AI kernels. *ICLR*.
- Chiang et al. (2024). Chatbot Arena: An open platform for evaluating LLMs by human preference. *ICML*.
