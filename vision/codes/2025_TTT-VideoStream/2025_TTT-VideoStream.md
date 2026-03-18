# Test-Time Training on Video Streams

> **Source**: *Journal of Machine Learning Research 26* (2025), pp. 1–29. Submitted 3/24; Revised 12/24; Published 1/25
> **Authors**: Renhao Wang\*, Yu Sun\*, Arnuv Tandon, Yossi Gandelsman, Xinlei Chen, Alexei A. Efros, Xiaolong Wang
> **Affiliations**: Wang, Sun, Gandelsman, Efros -- UC Berkeley; Tandon -- Stanford University; Chen -- Meta AI; X. Wang -- UC San Diego
> **Editor**: Samy Bengio
> **Project**: https://test-time-training.github.io/video
> **arXiv**: 2307.05014v3 [cs.CV]

---

## This Paper Covers
- Extending Test-Time Training (TTT) from independent images to the **streaming setting** where video frames arrive in temporal order
- Introducing **online TTT** with two forms of memory: *implicit memory* (carried-over parameters) and *explicit memory* (sliding window of recent frames)
- Demonstrating that online TTT outperforms both the fixed-model baseline and the offline TTT oracle on real-world video datasets
- Formalizing **locality** as the advantage of online over offline TTT, with ablations and a bias-variance trade-off theory
- Collecting **COCO Videos**, a new densely annotated video dataset that is orders of magnitude longer than existing ones

---

## Abstract

Prior work established TTT as a general framework to improve a trained model at test time via a self-supervised task such as reconstruction. This paper extends TTT to the streaming setting where video frames arrive in temporal order. The extension is **online TTT**: the current model is initialized from the previous model, then trained on the current frame and a small window of frames immediately before. Online TTT significantly outperforms the fixed-model baseline for four tasks on three real-world datasets (>2.2x and >1.5x for instance and panoptic segmentation). Surprisingly, online TTT also outperforms the offline variant that accesses all frames from the entire test video regardless of temporal order -- **challenging results from prior work using synthetic videos**. The paper formalizes a notion of *locality* as the advantage of online over offline TTT, and analyses its role with ablations and a theory based on bias-variance trade-off.

> **Figure 1**: In the streaming setting, the current model $f_t$ makes a prediction on the current frame before it can see the next one. $f_t$ is obtained through online TTT, initializing from the previous model $f_{t-1}$. Each video is treated as an independent unit. A sliding window of size $k$ contains the current and previous frames as test-time training data for the self-supervised task. Concretely, $k = 16$ gives a window of only 1.6 seconds.

> **Figure 2**: Bar charts for instance/panoptic segmentation on COCO Videos and semantic segmentation on KITTI-STEP. Online TTT (green) performs the best, requiring only the realistic streaming setting. Offline TTT (yellow) requires the unrealistic setting where all frames from the entire test video are available before making predictions. Online TTT outperforms offline by taking advantage of locality.

---

## Overview

Most models in machine learning are fixed during deployment. A trained model must prepare to be robust to all possible futures, but being ready for all futures limits the model's capacity to be good at any particular one, even though only one future actually happens. The basic idea of TTT is to continue training on the future once it arrives in the form of a test instance. Since no ground truth label is available, training is performed with self-supervision.

This paper investigates TTT on video streams, where each "future" (test instance $x_t$) is a frame and each video is an independent unit. The key question is about **locality**: is it better to train on all frames (offline/global) or only on a small, recent window (online/local)?

Besides conceptual interest, the paper has practical motivations: models for many computer vision tasks are trained on large datasets of still images (e.g., COCO for segmentation) but deployed on video streams. The default is to naively run such models frame-by-frame, since temporal smoothing (averaging across a sliding window of predictions) offers little improvement. Online TTT significantly improves prediction quality on four tasks: semantic, instance, and panoptic segmentation, and colorization. **Online TTT beats even the offline oracle.**

The paper also collects **COCO Videos**, a new video dataset with dense annotations. These videos are orders of magnitude longer than in other public datasets and contain much harder scenes from diverse daily-life scenarios. Longer and more challenging videos better showcase the importance of locality.

---

## 1 Introduction

**Key findings:**
- The best performance is achieved through **online TTT**: for each $x_t$, only train on itself and a small sliding window of less than two seconds of frames immediately before $t$. This sliding window is the **explicit memory**.
- The optimal explicit memory is short-term -- **some amount of forgetting is actually beneficial**. This challenges prior work in TTT (Wang et al., 2020; Volpi et al., 2022) and continual learning (Li and Hoiem, 2017; Lopez-Paz and Ranzato, 2017; Kirkpatrick et al., 2017), but is consistent with recent work in neuroscience (Gravitz, 2019).
- Parameters after training on $x_t$ carry over as initialization for training on $x_{t+1}$. This is the **implicit memory**. Most of the benefit comes from just one gradient step per frame.
- Both forms of memory depend on **temporal smoothness** -- that $x_t$ and $x_{t+1}$ are similar.

One of the most popular forms of self-supervision in computer vision is **reconstruction**: removing parts of the input image, then predicting the removed content (Vincent et al., 2008; Pathak et al., 2016; Bao et al., 2021; Xie et al., 2022). A class of deep learning models called masked autoencoders (MAE) (He et al., 2021) uses reconstruction as the self-supervised task and has been highly influential. **TTT-MAE** (Gandelsman et al., 2022) adopts these models for test-time training using reconstruction. The main task in Gandelsman et al. (2022) is object recognition; this paper uses TTT-MAE as a subroutine inside online TTT and extends it to other main tasks such as segmentation.

Prior streaming-setting work (Sun et al., 2020) experimented with online TTT without explicit memory, but each $x_t$ was drawn independently from the same test distribution created by adding synthetic corruption (e.g., Gaussian noise) to a still-image test set (e.g., ImageNet; Hendrycks and Dietterich, 2019). Therefore all $x_t$s belong to the same "future", and locality is meaningless. More recently, Volpi et al. (2022) also experimented in the streaming setting with short clips simulated to contain synthetic corruptions (e.g., CityScapes with Artificial Weather), where each corruption moves all $x_t$s into almost the same "future" (which they call a domain). Their only dataset without corruptions (CityScapes) saw little improvement (1.4% relative to no TTT). There is no mention of locality. TTT on actual video streams is fundamentally different and much more natural.

---

## 2 Related Work

### 2.1 Continual Learning

In continual learning (a.k.a. lifelong learning), a model learns a sequence of tasks defined by distributions $P_t$, with training sets $D_t^{\text{tr}}$ and test sets $D_t^{\text{te}}$ (Van de Ven and Tolias, 2019; Hadsell et al., 2020). At time $t$, the model is evaluated on all test sets $D_1^{\text{te}}, \ldots, D_t^{\text{te}}$ and average performance is reported. The oracle (infinite replay buffer) trains on $D_1^{\text{tr}}, \ldots, D_t^{\text{tr}}$ combined. Due to memory constraints, the model at time $t$ is only allowed to train on $D_t^{\text{tr}}$, so advanced solutions retain past data through model parameters (Santoro et al., 2016; Li and Hoiem, 2017; Lopez-Paz and Ranzato, 2017; Shin et al., 2017; Kirkpatrick et al., 2017; Gidaris and Komodakis, 2018).

Some literature extends beyond the conventional setting: continuous tasks (Aljundi et al., 2019), self-supervised learning on unlabeled training sets (Purushwalkam et al., 2022; Fini et al., 2022), using a labeled set $D_0^{\text{tr}}$ in addition to unlabeled sets (Hoffman et al., 2014; Li and Hospedales, 2020; Panagiotakopoulos et al., 2022), and alternative metrics such as forward transfer to justify forgetting (Diaz-Rodriguez et al., 2018).

Much of continual learning is motivated by the hope to understand human memory and generalization through the lens of artificial intelligence (Hassabis et al., 2017; De Lange et al., 2021). This paper shares the same motivation but focuses on test-time training, without distinct splits of training and test sets.

The TTT streaming setting differs: there are no distinct training/test splits. The model only needs to perform well on the *current* frame, not all past frames.

### 2.2 Test-Time Training

The idea of training at test time dates back to Bottou and Vapnik (1992) under the name *Local Learning*. This approach continues to be effective for SVMs (Zhang et al., 2006) and recently in large language models (Hardt and Sun, 2023). Transductive learning (Joachims, 2002; Collobert et al., 2006; Vapnik, 2013) also emphasizes locality. The principle of transduction (Gammerman et al., 1998; Vapnik and Kotz, 2006): "Try to get the answer that you really need but not a more general one."

In computer vision, TTT has been explored for specific applications (Jain and Learned-Miller, 2011; Shocher et al., 2018; Nitzan et al., 2022; Xie et al., 2023), especially depth estimation (Tonioni et al., 2019a,b; Zhang et al., 2020; Zhong et al., 2018; Luo et al., 2020). This paper extends **TTT-MAE** (Gandelsman et al., 2022). TTT-MAE is inspired by Sun et al. (2020), which proposed the general framework for test-time training with self-supervision. The particular self-supervised task in Sun et al. (2020) is rotation prediction (Gidaris et al., 2018). Many papers have followed this framework (Hansen et al., 2020; Sun et al., 2021; Liu et al., 2021b; Yuan et al., 2023), including Volpi et al. (2022) on videos and Azimi et al. (2022).

In Azimi et al. (2022), each video is treated as a dataset of unordered frames instead of a stream -- there is no concept of past vs. future frames. The same model is used on the entire video. In contrast, this paper emphasizes locality: access only to current and past frames, with the model learning over time. All results are on real-world videos, while Azimi et al. (2022) experiment on videos with artificial corruptions that are i.i.d. across frames.

The paper is very much inspired by **Mullapudi et al. (2018)**: to make video segmentation more efficient, they use a small student model for frame-by-frame predictions, querying an expensive teacher model only when the student is not confident. Thanks to temporal smoothness, the student generalises across many frames without the teacher. This paper uses one model with a self-supervised task instead of a teacher, and focuses on improving inference quality rather than efficiency. Behind their particular algorithm, the authors see the shared idea of locality, regardless of the form of supervision.

---

## 3 Background: TTT-MAE

The paper builds on Test-Time Training with Masked Autoencoders (TTT-MAE, Gandelsman et al., 2022). The general TTT architecture (Sun et al., 2020) is Y-shaped with a stem and two heads:
- Feature extractor $f$ (also called the encoder)
- Self-supervised head $g$ (decoder) for reconstruction
- Main task head $h$ for the downstream task (e.g., segmentation)

The output features of $f$ are shared between $g$ and $h$ as input. For TTT-MAE, the self-supervised task is masked image reconstruction (He et al., 2021). Following standard autoencoder terminology, $f$ is the encoder and $g$ the decoder.

Each input image $x$ is split into non-overlapping patches. To produce the autoencoder input $\bar{x}$, 80% of the patches are masked out. The self-supervised loss compares reconstructed patches to the masked originals via pixel-wise MSE:

$$\ell_s(g \circ f(\bar{x}), x)$$

For the main task (e.g., segmentation), all patches in the original $x$ are given as input to $h \circ f$, during both training and testing.

> **Figure 3**: Training a masked autoencoder to reconstruct each test image at test time. Reconstructed images visualize the progress of gradient descent on this one-sample learning problem. At Step 0: Reconstruction loss = 0.18, Segmentation loss = 18.59. At Step 1: Reconstruction loss = 0.12, Segmentation loss = 22.53 (lower reconstruction is better; higher segmentation is better). The unmasked patches are not shown on the right since they are not part of the reconstruction loss.

### 3.1 Training-Time Training

Three ways to optimize $(f, g, h)$ at training time: joint training, probing, and fine-tuning. Fine-tuning is unsuitable for TTT because it makes $h$ rely too much on features that are used by the main task. **This paper uses joint training** (described in Section 4). Gandelsman et al. (2022) uses probing, described here for completeness.

**Probing**: First train $f$ and $g$ with $\ell_s$ (self-supervised pre-training), producing $f_0$ and $g_0$. Gandelsman et al. (2022) uses the encoder and decoder already pre-trained by He et al. (2021). Then train $h$ separately by optimizing $\ell_m(h \circ f_0(x), y)$ while keeping $f_0$ frozen. Denote the main task head after probing as $h_0$. Since $h_0$ has been trained using features from $f_0$, the fixed model $h_0 \circ f_0$ serves as a baseline without TTT.

### 3.2 Test-Time Training

At test time, TTT-MAE takes gradient steps on the following one-sample learning problem:

$$f', g' = \arg\min_{f,g} \ell_s(g \circ f(\bar{x}'), x'), \tag{1}$$

then predicts $h_0 \circ f'(x')$. Crucially, the optimization always starts from $f_0$ and $g_0$. Gandelsman et al. (2022) always discards $f'$ and $g'$ after each test input and resets to $f_0$ and $g_0$. By test-time training on the test inputs independently, they do not assume that test inputs can help each other.

In the original MAE design (He et al., 2021), $g$ is very small relative to $f$, and only the visible patches (e.g., 20%) are processed by $f$. Therefore the computational cost of the self-supervised task is only a fraction (e.g., 25%) of the main task. In addition to speeding up training-time training for reconstruction, this reduces the extra test-time cost of TTT-MAE. Each gradient step at test time, counting both forward and backward, costs only half the time of forward prediction for the main task.

---

## 4 Test-Time Training on Video Streams

Each test video is a smoothly changing sequence $x_1, \ldots, x_T$; time $T$ is when the video ends. Each video is treated as an independent unit. In the streaming setting, at time $t$, the algorithm makes a prediction on $x_t$ after receiving it from the environment, before seeing any future frame, like how a human would consume it. Past frames $x_1, \ldots, x_{t-1}$ are available if the algorithm chooses to use them. Ground truth labels are never given on test videos.

At a high level, the algorithm simply amounts to a loop over the video frames, wrapped around TTT-MAE. In practice, making it work involves many design choices.

### 4.1 Training-Time Training

**Main Task Only** baseline: optimize $h \circ f$ end-to-end for the main task only; no self-supervised head $g$. But such a model is not suitable for TTT, since then the self-supervised head $g$ would have to be trained from scratch at test time.

To make $g$ **well-initialized for TTT**, at training time all three model components are jointly optimized in a single stage, end-to-end, on both the self-supervised task and the main task. This is called **joint training**. While joint training was also an option for TTT-MAE (Gandelsman et al., 2022), empirical experience at the time indicated that probing performed better. In this paper, joint training has been successfully tuned to be as effective as probing, and is simpler than the two-stage process.

$$g_0, h_0, f_0 = \arg\min_{g,h,f} \frac{1}{n} \sum_{i=1}^{n} \left[\ell_m(h \circ f(x_i), y_i) + \ell_s(g \circ f(\bar{x}_i), x_i)\right]$$

The summation is over the training set with $n$ samples, each consisting of input $x_i$ and label $y_i$. In the case of MAE, $\bar{x}_i$ is obtained by masking 80% of the patches in $x_i$. Note that although the test instances come from video streams, training-time training uses labeled still images, e.g., in the COCO training set, instead of unlabeled videos.

Joint training starts from a model checkpoint already trained for the main task; only $g$ is initialized from scratch.

**MAE Joint Training** baseline: the fixed model $h_0 \circ f_0$ after joint training, applied without any TTT. Empirically similar to *Main Task Only*. Joint training does not hurt or help when only considering the fixed model after training-time training.

### 4.2 Test-Time Training

**TTT-MAE No Memory**: Apply TTT-MAE (Eq. 1) independently to each frame $x_t$, resetting to $h_0$ and $f_0$ each time. This treats the video as a collection of unordered, independent frames. All three baselines (*Main Task Only*, *MAE Joint Training*, *TTT-MAE No Memory*) treat each video this way. None of the three can improve over time, no matter how long a video explores the same environment.

Improvement over time is only possible through some form of **memory**, by retaining information from past frames $x_1, \ldots, x_{t-1}$ to help prediction on $x_t$. Because evaluation is performed at each timestep only on the current frame, memory design should favour past data that are most relevant to the present. With the help of nature, the most recent frames are usually the most relevant due to **temporal smoothness** -- observations close in time tend to be similar.

**Implicit memory.** Do not reset model parameters between timesteps. Initialize test-time training at timestep $t$ with $f_{t-1}$ and $g_{t-1}$ instead of $f_0$ and $g_0$. Information carries over from previous parameters that were already optimized on previous frames. This is biologically plausible: humans do not constantly reset their minds. In prior work (Sun et al., 2020), TTT with implicit memory is called the "online" version, in contrast to the "standard" version with reset.

**Explicit memory.** Keep recent frames in a sliding window. Let $k$ denote the window size. At each timestep $t$, solve:

$$f_t, g_t = \arg\min_{f,g} \frac{1}{k} \sum_{t'=t-k+1}^{t} \ell_s(g \circ f(\bar{x}_{t'}), x_{t'}), \tag{2}$$

before predicting $h_0 \circ f_t(x_t)$. Optimization uses stochastic gradient descent: at each iteration, sample a batch **with replacement**, uniformly from the same window. Masking is applied independently within and across batches.

Only **one iteration** is sufficient, because given temporal smoothness, implicit memory already provides a good initialization for the optimization problem above.

### 4.3 Implementation Details

- In principle, the method is applicable to **any architecture**
- **Architecture**: Mask2Former (Cheng et al., 2021) with **Swin-S** backbone (Liu et al., 2021c) as the shared encoder $f$. Everything following the backbone in the original architecture is taken as the main task head $h$
- The decoder $g$ copies the architecture of $h$ except the last layer that maps into pixel space for reconstruction
- Joint training starts from the Mask2Former checkpoint for the main task; only $g$ is initialized from scratch
- Following He et al. (2021): split each input into patches, mask out 80%
- Unlike Vision Transformers (Dosovitskiy et al., 2020) used in He et al. (2021), Swin Transformers use convolutions. Therefore the entire image must be taken as input with masked patches replaced by black. Following Pathak et al. (2016), a fourth binary channel indicates which pixels are masked
- The fourth channel's parameters are initialized from scratch before joint training
- If a completely transformer-based architecture for segmentation becomes available in the future, the method could become even faster by not encoding the masked patches (He et al., 2021; Gandelsman et al., 2022)

---

## 5 Results

Four applications on three real-world datasets:
1. **Semantic segmentation** on KITTI-STEP (driving videos)
2. **Instance and panoptic segmentation** on COCO Videos (new dataset)
3. **Colorization** on COCO Videos and Lumiere Brothers films

Please visit the project website at https://video-ttt.github.io/ for videos of the results.

### 5.1 Additional Baselines

**Alternative architectures**: The authors of Mask2Former did not evaluate it on KITTI-STEP. Benchmarking against two other models of comparable size: SegFormer B4 (64.1M, 42.0% mIoU) and DeepLabV3+/RN101 (62.7M, 53.1% mIoU). Given that *Main Task Only* in Table 1 has 53.8%, the pre-trained Mask2Former (69M) is indeed state-of-the-art on KITTI-STEP. For COCO segmentation, the Mask2Former authors have already compared with alternative architectures, so those experiments are not repeated.

**Majority vote with augmentation**: Applying the default data augmentation recipe for 100 predictions per frame, then taking the majority vote. Improves *Main Task Only* by 1.2% mIoU on KITTI-STEP. Combining with online TTT yields roughly the same improvement, indicating they are orthogonal. Not used elsewhere in the paper.

**Temporal smoothing**: Averaging predictions across a sliding window. Improves *Main Task Only* by only 0.4% mIoU. Applying temporal smoothing to online TTT also yields 0.3% improvement -- again orthogonal. Not used elsewhere in the paper.

**Alternative TTT techniques**: Self-training (Volpi et al., 2022), layer norm (LN) adaptation (Schneider et al., 2020), and Tent (Wang et al., 2020). For self-training, the authors' implementation significantly improves on Volpi et al. (2022). See Appendix A for in-depth discussion. All perform poorly compared to Online TTT-MAE.

**Class balancing** (Volpi et al., 2022): Record the number of predicted classes for the initial model $h \circ f_0$ and the current model $h \circ f_t$. Reset the model parameters when the difference is large enough, in which case the predictions of the current model have likely collapsed. Evaluated on self-training and Tent. This heuristic cannot be applied to LN Adapt, which does not actually modify the trainable parameters in the model.

| **Setting** | **Method** | **COCO Videos** | | **KITTI-STEP** | | |
|---|---|---|---|---|---|---|
| | | *Instance* | *Panoptic* | *Val.* | *Test* | *Time* |
| Independent frames | Main Task Only | 16.7 | 13.9 | 53.8 | 52.5 | 1.8 |
| | MAE Joint Training | 16.5 | 13.5 | 53.5 | 52.5 | 1.8 |
| | TTT-MAE No Memory | 35.4 | 20.1 | 53.6 | 52.5 | 3.8 |
| Entire video available | Offline TTT-MAE All Frames | 33.6 | 19.6 | 53.2 | 51.2 | 1.8 |
| Frames in a stream | LN Adapt | 16.5 | 14.7 | 53.8 | 52.5 | 2.0 |
| | Tent | 16.6 | 14.6 | 53.8 | 52.2 | 2.8 |
| | Tent with Class Balance | 16.7 | 14.8 | 53.8 | 52.5 | 3.7 |
| | Self-Train | - | - | 54.7 | 54.0 | 6.6 |
| | Self-Train with Class Balance | - | - | 54.1 | 53.6 | 6.9 |
| | **Online TTT-MAE (Ours)** | **37.6** | **21.7** | **55.4** | **54.3** | 4.1 |

> **Table 1**: Metrics: AP (instance), PQ (panoptic), mIoU % (semantic). Time in seconds per frame on a single A100 GPU, averaged over the KITTI-STEP test set. Time costs on COCO Videos are similar, thus omitted. Self-training baselines are not applicable for instance and panoptic segmentation because the model does not return a confidence per object instance. Bars in Figure 2 correspond to: blue = *Main Task Only*, yellow = *Offline TTT-MAE All Frames*, green = *Online TTT-MAE (Ours)*.

### 5.2 Semantic Segmentation on KITTI-STEP

KITTI-STEP (Weber et al., 2021) contains 9 validation and 12 test videos of urban driving scenes at 10 fps -- the longest among public datasets with dense pixel-wise annotations, up to 106 seconds. All hyper-parameters, even for COCO Videos, are selected on the KITTI-STEP validation set. Joint training is performed on CityScapes (Cordts et al., 2016), another driving dataset with exactly the same 19 categories as KITTI-STEP, but containing still images instead of videos.

> **Note**: KITTI-STEP was originally designed to benchmark instance-level tracking, with a separate held-out test set. The official website evaluates only tracking-related metrics on this test set. The authors therefore perform their own evaluation using the segmentation labels. Since they do not perform regular training on KITTI-STEP, the training set is used as the test set.

Table 1 presents the main results. Figure 7 in the appendix visualizes predictions on two frames. *Online TTT-MAE* in the streaming setting performs the best. For semantic segmentation, such an improvement is usually considered highly significant in the community.

Baseline techniques that adapt normalization layers alone do not help at all. This agrees with Volpi et al. (2022): LN Adapt and Tent help significantly on datasets with synthetic corruptions, but do not help on real-world datasets (e.g., CityScapes).

*Online TTT-MAE* optimizes for only one iteration per frame and runs 2.3x slower than the baselines without TTT. Comparing with Gandelsman et al. (2022), which optimizes for 20 iterations per frame (image), the method runs much faster because implicit memory takes advantage of temporal smoothness to get a better initialization for every frame. Resetting parameters is wasteful on videos because adjacent frames are very similar.

### 5.3 COCO Videos

KITTI-STEP videos are still far too short for studying long-term phenomena in locality, and are limited to driving scenarios. The authors collected **COCO Videos**: 3 egocentric videos, each about 5 minutes, annotated by professionals in the same format as COCO instance and panoptic segmentation (Lin et al., 2014). The benchmark metrics are the same as in COCO: average precision (AP) for instance and panoptic quality (PQ) for panoptic.

| **Dataset** | **Length** | **Frames** | **Rate** | **Classes** |
|---|---|---|---|---|
| CityScapes-VPS (Kim et al., 2020) | 1.8 | 3,000 | 17 | 19 |
| DAVIS (Pont-Tuset et al., 2017) | 3.5 | 3,455 | 30 | - |
| YouTube-VOS (Xu et al., 2018) | 4.5 | 123,467 | 30 | 94 |
| KITTI-STEP (Weber et al., 2021) | 40 | 8,008 | 10 | 19 |
| **COCO Videos (Ours)** | **350** | **10,475** | **10** | **134** |

> **Table 2**: Video datasets with annotations for segmentation. Columns: average length per video in seconds, total number of frames, rate in fps, total number of classes. Each of the 3 COCO Videos alone contains more frames than all of the videos combined in the KITTI-STEP validation set.

All videos are egocentric (similar to a walking human). They do not follow any tracked object like in Oxford Long-Term Tracking (Valmadre et al., 2018) or ImageNet-Vid (Shankar et al., 2021). Objects constantly leave and enter the frame. Unlike KITTI-STEP and CityScapes that focus on self-driving scenes, the videos are both indoor and outdoor.

The authors start with the publicly available Mask2Former model pre-trained on still images in the COCO training set. Analogous to KITTI-STEP, joint training for TTT-MAE is also on COCO images, and the 3 videos are only used for evaluation. Mask2Former is state-of-the-art on the COCO validation set (44.9 AP instance, 53.6 PQ panoptic) but drops to 16.7 AP and 13.9 PQ on COCO Videos -- highlighting the challenging nature of COCO Videos and the fragility of still-image models on video.

Exactly the same hyper-parameters as tuned on KITTI-STEP are used for all algorithms. All results for COCO Videos were completed in a single run. As Figure 4 shows, a larger window size would further improve performance, but hyper-parameters for TTT should not be tuned on the test videos.

**Online TTT-MAE achieves >2.2x improvement for instance and >1.5x for panoptic segmentation** relative to *Main Task Only*. Improvements of this magnitude on the state-of-the-art are usually considered dramatic. The self-training baselines are not applicable because the model does not return a confidence per object instance.

Interestingly, even *TTT-MAE No Memory* (the most local method, using only the current frame) outperforms *Offline TTT-MAE All Frames* (the most global method). For COCO Videos, **local is better than global if one has to pick an extreme**.

### 5.4 Video Colorization

The goal is to add realistic RGB colours to gray-scale images, to demonstrate the generality of the method, not to achieve state-of-the-art. Following Zhang et al. (2016), colorization is simply treated as a supervised learning problem. The same Swin Transformer architecture is used with two heads, pre-trained on ImageNet to predict colours given gray-scale images. No domain-specific techniques (perceptual losses, adversarial learning, diffusion) are used. The bare-minimal baseline already achieves results comparable to Zhang et al. (2016). *Online TTT-MAE* uses exactly the same hyper-parameters as for segmentation. All colorization experiments were completed in a single run. Because colorizing COCO Videos is expensive, only *Online TTT-MAE* and *Main Task Only* are evaluated.

| **Method** | **FID** $\downarrow$ | **IS** $\uparrow$ | **LPIPS** $\uparrow$ | **PSNR** $\uparrow$ | **SSIM** $\uparrow$ |
|---|---|---|---|---|---|
| Zhang et al. (2016) | 62.39 | $5.00 \pm 0.19$ | 0.180 | 22.27 | **0.924** |
| Main Task Only (Cheng et al., 2021) | 59.96 | $5.23 \pm 0.12$ | 0.216 | 20.42 | 0.881 |
| **Online TTT-MAE (Ours)** | **56.47** | $\mathbf{5.31 \pm 0.18}$ | **0.237** | **22.97** | 0.901 |

> **Table 3**: Colorization results on COCO Videos. FID: Frechet Inception Distance. IS: Inception Score (standard deviation naturally available). LPIPS: Learned Perceptual Image Patch Similarity. PSNR: Peak Signal-to-Noise Ratio. SSIM: Structural Similarity. Arrows up = higher is better; arrows down = lower is better. PSNR and SSIM often misrepresent actual visual quality because colorization is inherently multi-modal (Zhang et al., 2016, 2017), but included for completeness.

For quantitative results, COCO Videos are first processed into black and white, enabling comparison with the original videos in RGB. For qualitative results, the 10 original black-and-white Lumiere Brothers films from 1895 (each ~40 seconds, at 10 fps) are colorized. The method visually improves quality in all of them, especially consistency across frames.

---

## 6 Analysis on Locality

The two philosophies: training on all possible futures in advance vs. training on the future once it actually happens. In other words, training globally vs. locally.

### 6.1 Empirical Analysis

| **Method** | **COCO Videos** | | **KITTI-STEP** | |
|---|---|---|---|---|
| | *Instance* | *Panoptic* | *Val.* | *Test* |
| TTT-MAE No Memory | 35.4 | 20.1 | 53.6 | 52.5 |
| Implicit Memory Only | 36.1 | 20.7 | 54.3 | 54.4 |
| Explicit Memory Only | 35.7 | 20.2 | 53.6 | 52.5 |
| **Online TTT-MAE (Ours)** | **37.6** | **21.7** | **55.4** | **54.3** |

> **Table 4**: Ablations on the two forms of memory. For ease of reference, *TTT-MAE No Memory* and *Online TTT-MAE (Ours)* values are reproduced from Table 1. Both forms contribute; implicit memory contributes more than explicit memory alone.

**Offline TTT-MAE**: Trains a single model for each test video in a setting where all frames are available for training with the self-supervised task before predictions are made. Strictly more information is provided than the streaming setting. The frames are shuffled into a training set, and gradient iterations are taken on batches sampled the same way as from the sliding window. To give *Offline TTT-MAE All Frames* even more advantage, results are reported from the best iteration on each test video, as measured by actual test performance (which would not be available in the real world). For many videos, this best iteration is around 1000. Even so, online TTT still outperforms.

**Window size $k$**: The choice of explicit memory is not binary. On one end, $k = 1$ is the same as *Implicit Memory Only*. On the other end, $k = \infty$ comes close to *Offline TTT-MAE All Frames*, except that future frames cannot be trained on. Online TTT-MAE prefers very short-term memory. For all three tasks at 10 fps, $k = 16$ covers only 1.6 seconds. Too little memory hurts (high variance), but so does too much (high bias from irrelevant distant frames).

> **Figure 4**: Effect of window size $k$ on performance. For all window sizes, the batch size and computational cost are fixed. KITTI-STEP plot is on the validation set where $k = 16$ was selected. The optimal $k$ on COCO Videos is different for semantic and panoptic segmentation, but Table 1 results still use $k = 16$. The y-values for $k = 1$ match *Implicit Memory Only* in Table 4. The x-axis is in log scale to highlight the effect over a large range; performance is actually not sensitive to $k$ in linear scale.

> **Figure 5**: An illustration of locality. Frame at $t - 10$ (still inside a lecture hall) is relevant and reduces variance. Frame at $t - 200$ (shot before entering the hall) would significantly increase bias.

**Temporal smoothness**: The key assumption that makes both forms of memory effective. Verified by shuffling all frames within each video, destroying temporal order. By construction, all three methods under the setting of independent frames -- *Main Task Only*, *MAE Joint Training*, and *TTT-MAE No Memory* -- are not affected. The same goes for *Offline TTT-MAE All Frames*, which already shuffles. For *Online TTT-MAE*, however, shuffling hurts performance dramatically -- on the KITTI-STEP validation set, performance drops below *Main Task Only*.

### 6.2 Theoretical Analysis

To complement the empirical observation that locality can be beneficial, the effect of window size $k$ for TTT using any self-supervised task is analysed rigorously.

**Notations.** Define the following functions of the shared model parameters $\theta$:

$$\nabla \ell_m^t(\theta) := \nabla_\theta \ell_m(x_t, y_t; \theta), \tag{3}$$

$$\nabla \ell_s^t(\theta) := \nabla_\theta \ell_s(x_t; \theta). \tag{4}$$

These notations have appeared in Sections 3 and 4, where the main task loss $\ell_m$ is defined for object recognition or segmentation, and the self-supervised task loss $\ell_s$ is instantiated as pixel-wise MSE for image reconstruction; $\theta$ refers to parameters of the encoder $f$.

**Problem statement.** Taking gradient steps with $\nabla \ell_m^t$ directly optimizes the test loss, since $y_t$ is the ground truth label of test input $x_t$. However, $y_t$ is not available, so TTT optimizes $\ell_s$ instead. Among the available gradients, $\nabla \ell_s^t$ is the most relevant. But we also have the past inputs $x_1, \ldots, x_{t-1}$. Should we use some, or even all of them?

Consider TTT with gradient-based optimization using:

$$\frac{1}{k} \sum_{t'=t-k+1}^{t} \nabla \ell_s^{t'}, \tag{5}$$

where $k$ is the window size.

**Theorem.** For every timestep $t$, let $\theta_0$ denote the initial condition and $\bar{\theta}$ where optimization converges. Let $\theta^*$ denote the optimal solution of $\ell_m^t$ in the local neighbourhood of $\theta_0$. Then:

$$\mathbb{E}\left[\ell_m(x_t, y_t; \bar{\theta}) - \ell_m(x_t, y_t; \theta^*)\right] \leq \frac{1}{2\alpha}\left(k^2 \beta^2 \eta^2 + \frac{1}{k}\sigma^2\right),$$

under three assumptions:

1. In a local neighbourhood of $\theta^*$, $\ell_m^t$ is $\alpha$-strongly convex in $\theta$, and $\beta$-smooth in $x$.
2. $\|x_{t+1} - x_t\| \leq \eta$ (temporal smoothness in $L^2$ norm; any other norm could be used if the norm in Assumption 1 is changed accordingly).
3. $\nabla \ell_m^t = \nabla \ell_s^t + \delta_t$, where $\delta_t$ is a random variable with mean zero and variance $\sigma^2$ (correlated gradients between main and self-supervised tasks).

The proof is in Appendix C.

**Remark on assumptions.** Assumption 1 (neural networks are strongly convex around their local minima) is widely accepted in the learning theory community (Allen-Zhu et al., 2019; Zhong et al., 2017; Wang et al., 2021). Assumption 2 is simply temporal smoothness. Assumption 3 (main task and self-supervised task have correlated gradients) comes from the theoretical analysis of Sun et al. (2020).

**Bias-variance trade-off.** Disregarding the constant factor of $1/\alpha$, the upper bound is the sum of two terms: $k^2 \beta^2 \eta^2$ and $\sigma^2/k$. The former is the **bias term**, growing with $\eta$. The latter is the **variance term**, growing with $\sigma^2$. More memory (larger sliding window $k$) reduces variance but increases bias. This is consistent with the intuition in Figure 5. Optimizing this upper bound w.r.t. $k$ reveals the theoretical sweet spot:

$$k = \left(\frac{\sigma^2}{\beta^2 \eta^2}\right)^{1/3}$$

This confirms the empirical finding: an intermediate $k$ is optimal -- too small increases variance, too large increases bias.

---

## 7 Discussion

The authors connect their work to other ideas in machine learning.

**Unsupervised domain adaptation (UDA).** *Offline TTT-MAE All Frames*, where the entire unlabeled test video is available at once, is very similar to UDA. Each test video can be viewed as a target domain, and offline MAE practically treats the frames as i.i.d. data drawn from a single distribution. The only difference with UDA is that the unlabeled video serves as both training and test data. In fact, the modified version of UDA above is sometimes called *test-time adaptation*. The results suggest that this perspective of seeing each video as a target domain might be misleading for algorithm design, because it discourages locality.

**Continual learning.** Conventional wisdom holds that forgetting is harmful and the best accuracy is achieved by remembering everything with an infinite replay buffer, given unlimited computer memory. The streaming setting differs from those commonly studied in continual learning, because it does not have distinct splits of training and test sets (as explained in Section 2.1). However, the sliding window can be viewed as a replay buffer, and limiting its size is a form of forgetting. **Forgetting can actually be beneficial** in this context.

**TTT on nearest neighbours.** An alternative heuristic: for each test instance, retrieve its nearest neighbours from a training set and fine-tune on them. This has been explored in Bottou and Vapnik (1992) and Hardt and Sun (2023). Given temporal smoothness -- that proximity in time translates to proximity in the retrieval metric -- the sliding window can be seen as retrieving "neighbours" of the current frame from the unlabeled test video instead of a labelled training set. Two consequences: on one hand, self-supervision must be used; on the other hand, the "neighbours" are still relevant (given temporal smoothness) even when the test instance is not represented by the training set.

**In-context learning.** Each test video could be used as the context of a Transformer or RNN, both of which often exhibit "in-context learning" (Brown et al., 2020). As long as the model is autoregressive, the video is still processed as a stream. But in practice, this requires the model to be already trained on videos. This paper's approach does not use videos as training data; the very goal is to study the generalization from still images to videos.

**Sequence modelling.** The method as shown in Figure 1 closely resembles an RNN, where the model parameters of $f$ serve as the hidden state $W$. From this perspective, online TTT can be regarded as compressing frames $x_1, \ldots, x_t$ into $W_t$, and gradient descent is simply a particular update rule. Following earlier versions of this paper, Sun et al. (2023) and Sun et al. (2024) program TTT into sequence modelling layers as an alternative to self-attention, and apply it to language modelling.

> **Figure 6** (from Sun et al., 2024): **Top**: A generic sequence modelling layer expressed as a hidden state that transitions according to an update rule. All sequence modelling layers can be viewed as different instantiations of three components: initial state, update rule, and output rule. **Bottom**: Examples of sequence modelling layers. RNN layers compress the growing context into a fixed-size hidden state, so their cost per token stays constant. Self-attention has a hidden state growing with context, therefore growing cost per token. The TTT layer, introduced in Sun et al. (2024), is a particular RNN layer with gradient descent as the update rule, following the approach in this paper.

| | **Initial State** | **Update Rule** | **Output Rule** | **Cost** |
|---|---|---|---|---|
| Naive RNN | $s_0 = \text{vector}()$ | $s_t = \sigma(\theta_{ss} s_{t-1} + \theta_{sx} x_t)$ | $z_t = \theta_{zs} s_t + \theta_{zx} x_t$ | $O(1)$ |
| Self-attention | $s_0 = \text{list}()$ | $s_t = s_{t-1}.\text{append}(k_t, v_t)$ | $z_t = V_t \text{softmax}(K_t^T q_t)$ | $O(t)$ |
| Naive TTT | $W_0 = f.\text{params}()$ | $W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)$ | $z_t = f(x_t; W_t)$ | $O(1)$ |

---

## Appendix A: Baseline Techniques for TTT

### A.1 Self-Training

Self-training is a popular technique in semi-supervised learning (Radosavovic et al., 2018; Rosenberg et al., 2005; Zoph et al., 2020; Asano et al., 2019, 2020) and domain adaptation (Kumar et al., 2020; Zou et al., 2018; Mei et al., 2020; Liu et al., 2021a; Spadotto et al., 2021). Volpi et al. (2022) evaluated it but found inferior performance. This paper experiments with self-training both in its original form and with design improvements.

**Setup**: For each test image $x$, the prediction $\hat{y}$ is of the same shape in 2D (satisfied for semantic segmentation and colorization). $F$ also outputs an estimated confidence map $\hat{c}$ of the same shape. For pixel $x[i,j]$: $\hat{y}[i,j]$ is the predicted class, $\hat{c}[i,j]$ is the estimated confidence. Self-training repeats many iterations of:

1. Start with an empty set of labels $D$ for this iteration
2. Loop over every $[i,j]$ location, add pseudo-label $\hat{y}[i,j]$ to $D$ if $\hat{c}[i,j] > \lambda$, for a fixed threshold $\lambda$
3. Train $F$ to fit this iteration's set $D$, as if the selected pseudo-labels are ground truth labels

**Improvement 1 -- Confidence threshold $\lambda$**: In Volpi et al. (2022), all predictions are used as pseudo-labels regardless of confidence. For low $\lambda$ or $\lambda = 0$, self-training is noisy and unstable. For high $\lambda$, there is limited learning signal (little gradient, since $f$ is already very confident about the pseudo-label). An intermediate threshold works best.

**Improvement 2 -- Strong masking** (inspired by Sohn et al., 2020): Make learning more challenging with already-confident predictions by masking image patches in $x$. In Sohn et al. (2020), masking is applied sparingly on 2.5% of pixels on average. This paper masks 80% of the pixels, inspired by He et al. (2021).

### A.2 Layer Norm Adapt

Prior work (Schneider et al., 2020) shows that recalculating batch normalization (BN) statistics works well for unsupervised domain adaptation. Volpi et al. (2022) applies this to video streams by accumulating the statistics with a forward pass on each frame once it is revealed. Since modern transformers use layer normalization (LN) instead, the same technique is applied to LN.

### A.3 Tent

Tent (Wang et al., 2020) is an objective for learning only the trainable parameters of normalization layers (BN and LN) at test time, by minimizing the softmax entropy of the predicted distribution over classes. The LN statistics and parameters are updated with Tent in the same loop as the main method, also using implicit and explicit memory. Hyper-parameters are searched on the KITTI-STEP validation set to be optimal for Tent.

---

## Appendix B: Visualization and Colorization Dataset

> **Figure 7**: Semantic segmentation predictions for adjacent frames ($t = 221$, $t = 222$) from a video in KITTI-STEP. **Top**: Fixed model baseline without TTT -- predictions are inconsistent between the two frames; terrain on the right is incompletely segmented, and terrain on the left is incorrectly classified as a wall on the first frame. **Bottom**: *Online TTT-MAE* on the same frames -- predictions are now consistent and correct.

> **Figure 8**: Panoptic segmentation predictions for adjacent frames ($t = 286$, $t = 287$) from COCO Videos. **Top**: Fixed model baseline without TTT -- predictions are inconsistent. **Bottom**: *Online TTT-MAE* -- predictions are now consistent and correct.

### Lumiere Brothers Films

10 public-domain films from 1895, each approximately 40 seconds:

1. Workers Leaving the Lumiere Factory (46 s)
2. The Gardener (49 s)
3. The Disembarkment of the Congress of Photographers in Lyon (48 s)
4. Horse Trick Riders (46 s)
5. Fishing for Goldfish (42 s)
6. Blacksmiths (49 s)
7. Baby's Meal (41 s)
8. Jumping Onto the Blanket (41 s)
9. Cordeliers Square in Lyon (44 s)
10. The Sea (38 s)

> **Figure 9**: Colorization results on the Lumiere Brothers films. **Top**: Zhang et al. (2016). **Middle**: Mask2Former *Main Task Only* baseline, which is already comparable, if not superior to Zhang et al. (2016). **Bottom**: After applying *Online TTT-MAE* on top of the baseline. Colours are more vibrant and consistent within regions.

---

## Appendix C: Proof of Theorem 1

**Lemma.** Let $f: \mathbb{R}^n \to \mathbb{R}$ be $\alpha$-strongly convex and continuously differentiable, with optimal solution $x^*$. Let

$$\tilde{f}(x) = f(x) + v^T x, \tag{6}$$

with optimal solution $\tilde{x}^*$. Then:

$$f(\tilde{x}^*) - f(x^*) \leq \frac{1}{2\alpha} \|v\|^2. \tag{7}$$

*Proof of Lemma.* It is a well known fact in convex optimization (Bubeck et al., 2015) that for $f$ $\alpha$-strongly convex and continuously differentiable:

$$\alpha(f(x) - f(x^*)) \leq \frac{1}{2}\|\nabla f(x)\|^2, \tag{8}$$

for all $x$. Since $\tilde{x}^*$ is the optimal solution of $\tilde{f}$ and $\tilde{f}$ is also convex, we have $\nabla \tilde{f}(\tilde{x}^*) = 0$. But

$$\nabla \tilde{f}(x) = \nabla f(x) + v, \tag{9}$$

so we then have

$$\nabla f(\tilde{x}^*) = \nabla \tilde{f}(\tilde{x}^*) - v = -v. \tag{10}$$

Making $x = \tilde{x}^*$ in Equation 8, we finish the proof. $\square$

**Proof of Theorem.** By Assumptions 1 and 2, we have

$$\|\nabla \ell_m^t(\theta) - \nabla \ell_m^{t-1}(\theta)\| \leq \beta \eta. \tag{11}$$

Using Assumption 3, decompose the averaged self-supervised gradient:

$$\frac{1}{k}\sum_{t'=t-k+1}^{t} \nabla \ell_s^{t'} = \frac{1}{k}\sum_{t'=t-k+1}^{t} \nabla \ell_m^{t'} + \frac{1}{k}\sum_{t'=t-k+1}^{t} \delta_{t'} \tag{12}$$

$$= \frac{1}{k}\sum_{t'=t-k+1}^{t} \left[\nabla \ell_m^t + \sum_{t''=t'}^{t-1}\left(\nabla \ell_m^{t''} - \nabla \ell_m^{t''+1}\right)\right] + \frac{1}{k}\sum_{t'=t-k+1}^{t}\delta_{t'} \tag{13}$$

$$= \nabla \ell_m^t + \frac{1}{k}\left[\underbrace{\sum_{t'=t-k+1}^{t}\sum_{t''=t'}^{t-1}\left(\nabla \ell_m^{t''} - \nabla \ell_m^{t''+1}\right)}_{A}\right] + \frac{1}{k}\underbrace{\sum_{t'=t-k+1}^{t}\delta_{t'}}_{B} \tag{14}$$

To simplify notation, define

$$A = \sum_{t'=t-k+1}^{t}\sum_{t''=t'}^{t-1}\left(\nabla \ell_m^{t''} - \nabla \ell_m^{t''+1}\right), \tag{15}$$

$$B = \sum_{t'=t-k+1}^{t}\delta_{t'}. \tag{16}$$

So

$$\frac{1}{k}\sum_{t'=t-k+1}^{t} \nabla \ell_s^{t'} - \nabla \ell_m^t = (A + B)/k. \tag{17}$$

Because $\ell_m^t$ is convex in $\theta$, taking gradient steps with $\nabla \ell_m^t$ would eventually reach the local optima of $\ell_m^t$. Because $\frac{1}{k}\sum_{t'=t-k+1}^{t}\nabla \ell_s^{t'}$ differs from $\nabla \ell_m^t$ by $(A+B)/k$, taking gradient steps with the former reaches the local optima of $\ell_m^t + (A+B)\theta/2$. Now we can invoke our lemma. To do so, we first calculate

$$\mathbb{E}\left\|\frac{1}{k}\sum_{t'=t-k+1}^{t}\nabla \ell_s^{t'} - \nabla \ell_m^t\right\|^2 = \frac{1}{k^2}\mathbb{E}\|A + B\|^2 \tag{18}$$

$$= \frac{1}{k^2}\left(\|A\|^2 + \mathbb{E}\|B\|^2 + \mathbb{E}\,A^T B\right) \tag{19}$$

$$\leq \frac{1}{k^2}\left(k^4 \beta^2 \eta^2 + k\sigma^2\right) \tag{20}$$

$$= k^2 \beta^2 \eta^2 + \frac{1}{k}\sigma^2. \tag{21}$$

Then by our lemma, we have

$$\mathbb{E}\left[\ell_m(x_t, y_t; \bar{\theta}) - \ell_m^*\right] \leq \frac{1}{2\alpha}\,\mathbb{E}\left\|\frac{1}{k}\sum_{t'=t-k+1}^{t}\nabla \ell_s^{t'} - \nabla \ell_m^t\right\|^2 \leq \frac{1}{2\alpha}\left(k^2\beta^2\eta^2 + \frac{1}{k}\sigma^2\right). \tag{22}$$

This finishes the proof. $\square$

---

## Summary
- **Online TTT** extends TTT to the streaming video setting with two forms of memory: *implicit* (parameter carry-over) and *explicit* (sliding window of recent frames)
- Online TTT with only **one gradient step per frame** and a window of **~1.6 seconds** (16 frames at 10 fps) significantly outperforms the fixed model
- **Online TTT beats even the offline oracle** that accesses all frames from the entire video, challenging prior assumptions that more data is always better
- The key concept is **locality**: for prediction on the current frame, only nearby (recent) frames provide useful self-supervised training signal
- Forgetting is beneficial: a limited sliding window outperforms using the entire video history
- A **bias-variance trade-off theory** formalizes the optimal window size as $k = (\sigma^2 / \beta^2 \eta^2)^{1/3}$
- The method generalizes across **four tasks** (semantic, instance, panoptic segmentation, colorization) and **three datasets** (KITTI-STEP, COCO Videos, Lumiere Brothers films)
- Online TTT resembles an **RNN** where model parameters are the hidden state and gradient descent is the update rule -- this perspective later inspired TTT layers for language modelling (Sun et al., 2023, 2024)

---

## References and Further Reading
- Project website: https://test-time-training.github.io/video (also: https://video-ttt.github.io/)
- Sun et al. (2020). Test-time training with self-supervision for generalization under distribution shifts. *ICML*.
- Gandelsman et al. (2022). Test-time training with masked autoencoders. *NeurIPS*.
- Sun et al. (2024). Learning to (learn at test time): RNNs with expressive hidden states. *arXiv:2407.04620*.
- Volpi et al. (2022). On the road to online adaptation for semantic image segmentation. *CVPR*.
- Mullapudi et al. (2018). Online model distillation for efficient video inference. *arXiv:1812.02699*.
- Bottou and Vapnik (1992). Local learning algorithms. *Neural Computation*.
- Hardt and Sun (2023). Test-time training on nearest neighbors for large language models. *arXiv:2305.18466*.
