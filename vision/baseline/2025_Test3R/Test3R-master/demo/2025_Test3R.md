# Test3R: Learning to Reconstruct 3D at Test Time

**Yuheng Yuan, Qiuhong Shen, Shizun Wang, Xingyi Yang, Xinchao Wang** *National University of Singapore* {yuhengyuan, qiuhong.shen, shizun.wang, xyang}@u.nus.edu, xinchao@nus.edu.sg  
Project page: [https://test3r-nop.github.io/](https://test3r-nop.github.io/)

---

## Abstract

Dense matching methods like DUSt3R regress pairwise pointmaps for 3D reconstruction. However, the reliance on pairwise prediction and the limited generalization capability inherently restrict the global geometric consistency. In this work, we introduce **Test3R**, a surprisingly simple test-time learning technique that significantly boosts geometric accuracy. Using image triplets $(I_{1}, I_{2}, I_{3})$, Test3R generates reconstructions from pairs $(I_{1}, I_{2})$ and $(I_{1}, I_{3})$. The core idea is to optimize the network at test time via a self-supervised objective: maximizing the geometric consistency between these two reconstructions relative to the common image $I_{1}$. This ensures the model produces cross-pair consistent outputs, regardless of the inputs. Extensive experiments demonstrate that our technique significantly outperforms previous state-of-the-art methods on the 3D reconstruction and multi-view depth estimation tasks. Moreover, it is universally applicable and nearly cost-free, making it easily applied to other models and implemented with minimal test-time training overhead and parameter footprint. Code is available at [https://github.com/nopQAQ/Test3R](https://github.com/nopQAQ/Test3R).

---

## 1. Introduction

3D reconstruction from multi-view images is a cornerstone task in computer vision. Traditionally, this process has been achieved by assembling classical techniques such as keypoint detection [1-3] and matching [4, 5], robust camera estimation [4, 6], Structure-from-Motion (SfM), Bundle Adjustment (BA) [7-9], and dense Multi-View Stereo [10, 11]. Although effective, these multi-stage methods require significant engineering effort to manage the entire process. This complexity inherently constrains their scalability and efficiency.

Recently, dense matching methods, such as DUSt3R [12] and MAST3R [13], have emerged as compelling alternatives. At its core, DUSt3R utilizes a deep neural network trained to predict dense correspondences between image pairs in an end-to-end fashion. Specifically, DUSt3R takes in two images and, for each, predicts a pointmap. Each pointmap represents the 3D coordinates of every pixel, as projected into a common reference view's coordinate system. Once pointmaps are generated from multiple views, DUSt3R aligns them by optimizing the registration of these 3D points. This process recovers the camera pose for each view and reconstructs the overall 3D geometry.

**Figure 1:** Given a set of images of a specific scene, our Test3R improves the quality of reconstruction by maximizing the consistency between the pointmaps generated from multiple image pairs.

**Figure 2 (Inconsistency Study):** On the left are two image pairs sharing the same reference view $I_{1}$ but with different source views $I_{2}$ and $I_{3}$. On the right are the corresponding point maps, with each color indicating the respective image pair.

Despite its huge success, this pair-wise prediction paradigm is inherently problematic. Under such a design, the model considers only two images at a time. Such a constraint leads to several issues. To investigate this, we compare the pointmaps of image $I_{1}$ but with different views $I_{2}$ and $I_{3}$ in Figure 2. It demonstrates that the predicted pointmaps are imprecise and inconsistent.

Firstly, the precision of geometric predictions can suffer because the model is restricted to inferring scene geometry from just one image pair. This is especially true for short-baseline cases [14], where small camera movement leads to poor triangulation and thus inaccurate geometry. Second, reconstructing an entire scene requires pointmaps from multiple image pairs. Unfortunately, these individual pairwise predictions may not be mutually consistent. For example, the pointmap predicted from $(I_{1}, I_{2}, X_{3})$ may not align with the prediction from $(I_{1}, X_{3}, I_{3})$ as highlighted by the color difference in Figure 2. This local inconsistency further leads to discrepancies in the overall reconstruction. What makes things worse, the model, like many deep learning systems, struggles to generalize to new or diverse scenes. Such limitations directly exacerbate the previously discussed problems of precision and inter-pair consistency. Consequently, even with a final global refinement stage, inaccurate pointmaps lead to persistent errors.

To address these problems, in this paper, we present **Test3R**, a novel yet strikingly simple solution for 3D reconstruction, operating entirely at test time. Its core idea is straightforward: **Maximizing the consistency between the reconstructions generated from multiple image pairs.** This principle is realized through two basic steps:

1.  Given image triplets $(I_{1}, I_{2}, I_{3})$, Test3R first estimates two initial pointmaps with respect to $I_{1}$: $X_{1}$ from pairs $(I_{1}, I_{2})$ and $X_{2}$ from $(I_{1}, I_{3})$.
2.  Test3R optimizes the network, so that the two pointmaps are cross-pair consistent, i.e., $X_{1} \approx X_{2}$. Critically, this optimization is performed at test time via prompt tuning [15].

Despite its simplicity, Test3R offers a robust solution to all challenges mentioned above. It ensures consistency by aligning local two-view predictions, which resolves inconsistencies. This same mechanism also improves geometric precision: if a pointmap from short-baseline images is imprecise, Test3R pushes it closer to an overall global prediction, which reduces errors. Finally, Test3R adapts to new, unseen scenes, minimizing its errors on unfamiliar data.

We evaluated Test3R on the DUSt3R for 3D reconstruction and multi-view depth estimation. Test3R performs exceptionally well across diverse datasets, improving upon vanilla DUSt3R to achieve competitive or state-of-the-art results in both tasks. Surprisingly, for multi-view depth estimation, Test3R even surpasses baselines requiring camera poses and intrinsics, as well as those trained on the same domain. This further validates our model's robustness and efficacy.

The best part is that Test3R is universally applicable and nearly cost-free. This means it can easily be applied to other models sharing a similar pipeline. We validated this by incorporating our design into MAST3R [13] and MonST3R [16]. Experimental results confirmed substantial performance improvements for both models.

**The contributions of this work are as follows:**
* We introduce Test3R, a novel yet simple solution to learn the reconstruction at test time. It optimizes the model via visual prompts to maximize the cross-pair consistency. It provides a robust solution to the challenges of the pairwise prediction paradigm and limited generalization capability.
* We conducted comprehensive experiments across several downstream tasks on the DUSt3R. Experiment results demonstrate that Test3R not only improves the reconstruction performance compared to vanilla DUSt3R but also outperforms a wide range of baselines.
* Our design is universally applicable and nearly cost-free. It can easily be applied to other models and implemented with minimal test-time training overhead and parameter footprint.

---

## 2. Related Work

### 2.1 Multi-view Stereo
Multi-view Stereo (MVS) aims to densely reconstruct the geometry of a scene from multiple overlapping images. Traditionally, all camera parameters are often estimated with SfM [17], as the given input. Existing MVS approaches can generally be classified into three categories: traditional handcrafted [11, 18-20], global optimization [21-24], and learning-based methods [10, 25-28]. Recently, DUSt3R [12] has attracted significant attention as a representative of learning-based methods. It attempts to estimate dense pointmaps from a pair of views without any explicit knowledge of the camera parameters. Subsequent tremendous works focus on improving its efficiency [29-31], quality [13, 29, 32], and broadening its applicability to dynamic reconstruction [16, 33-36] and 3D perception [37]. The majority employ the pairwise prediction strategy introduced by DUSt3R [12]. However, the pair-wise prediction paradigm is inherently problematic. It leads to low precision and mutually inconsistent pointmaps. Furthermore, the limited generalization capability of the model exacerbates these issues. This challenge continues even with the latest models [29, 38], which can process multiple images in a single forward pass. While potentially more robust, these newer approaches demand significantly larger resources for training and, importantly, still face challenges in generalizing to unseen environments. To this end, we introduce a novel test-time training technique. This simple design ensures the cross-pairs consistency by aligning local two-view predictions to push the pointmaps closer to an overall global prediction, which addresses all challenges mentioned above.

### 2.2 Test-time Training
The idea of training on unlabeled test data dates back to the 1990s [39], called transductive learning. As Vladimir Vapnik [40] famously stated, "Try to get the answer that you really need but not a more general one", this principle has been widely applied to SVMs [41, 42] and recently in large language models [43]. Another early line of work is local learning [44, 45]: for each test input, a "local" model is trained on the nearest neighbors before a prediction is made. Recently, Test-time training (TTT) [46] proposes a general framework for test-time training with self-supervised learning, which produces a different model for every single test input through the self-supervision task. This strategy allows the model trained on the large-scale datasets to adapt to the target domain at test time. Many other works have followed this framework since then [47-50]. Inspired by these studies, we introduce Test3R, a novel yet simple technique that extends the test-time training paradigm to the 3D reconstruction domain. Our model exploits the cross-pairs consistency as a strong self-supervised objective to optimize the model parameters at test time, thereby improving the final quality of reconstruction.

### 2.3 Prompt tuning
Prompt tuning was first proposed as a technique that appends learnable textual prompts to the input sequence, allowing pre-trained language models to adapt to downstream tasks without modifying the backbone parameters [51]. In follow-up research, a portion of studies [52, 53] explored strategies for crafting more effective prompt texts, whereas others [54-56] proposed treating prompts as learnable, task-specific continuous embeddings, which are optimized via gradient descent during fine-tuning referred to as Prompt Tuning. In recent years, prompt tuning has also received considerable attention in the 2D vision domain. Among these, Visual Prompt Tuning (VPT) [15] has gained significant attention as an efficient approach specifically tailored for vision tasks. It introduces a set of learnable prompt tokens into the pretrained model and optimizes them using the downstream task's supervision while keeping the backbone frozen. This strategy enables the model to transfer effectively to downstream tasks. In our study, we leverage the efficient fine-tuning capability of VPT to optimize the model to ensure the pointmaps are cross-view consistent. This design makes our model nearly cost-free, requiring minimal test-time training overhead and a small parameter footprint.

---

## 3. Preliminary of DUSt3R

Given a set of images $\{I_{t}^{i}\}_{i=1}^{N_{t}}$ of a specific scene, DUSt3R [12] achieves high precision 3D reconstruction by predicting pairwise pointmaps of all views and global alignment.

**Pairwise prediction.** Briefly, DUSt3R takes a pair of images, $I^{1}, I^{2} \in \mathbb{R}^{W \times H \times 3}$ as input and outputs the corresponding pointmaps $X^{1,1}, X^{2,1} \in \mathbb{R}^{W \times H \times 3}$ which are expressed in the same coordinate frame of $I^{1}$. In our paper, we refer to the viewpoint of $I^{1}$ as the reference view, while the other is the source view. Therefore, the pointmaps $X^{1,1}, X^{2,1}$ can be denoted as $X^{ref,ref}, X^{src,ref}$, respectively.

In more detail, these two input images $I^{ref}, I^{src}$ are first encoded by the same weight-sharing ViT-based model [57] with $N_{e}$ layers to yield two token representations $F^{ref}$ and $F^{src}$:
$$F^{ref} = Encoder(I^{ref}), \quad F^{src} = Encoder(I^{src}) \quad (1)$$

After encoding, the network reasons over both of them jointly in the decoder. Each decoder block also attends to tokens from the other branch:
$$G_{i}^{ref} = DecoderBlock_{i}^{ref}(G_{i-1}^{ref}, G_{i-1}^{src}) \quad (2)$$
$$G_{i}^{src} = DecoderBlock_{i}^{src}(G_{i-1}^{src}, G_{i-1}^{ref}) \quad (3)$$

where $i=1, ..., N_{d}$ for a decoder with $N_{d}$ decoder layers and initialized with encoder tokens $G_{0}^{ref} = F^{ref}$ and $G_{0}^{src} = F^{src}$. Finally, in each branch, a separate regression head takes the set of decoder tokens and outputs a pointmap and an associated confidence map:
$$X^{ref,ref}, C^{ref,ref} = Head^{ref}(G_{0}^{ref}, ..., G_{N_{d}}^{ref}) \quad (4)$$
$$X^{src,ref}, C^{src,ref} = Head^{src}(G_{0}^{src}, ..., G_{N_{d}}^{src}) \quad (5)$$

**Global alignment.** After predicting all the pairwise pointmaps, DUSt3R introduces a global alignment to handle pointmaps predicted from multiple images. For the given image set $\{I_{t}^{i}\}_{i=1}^{\tilde{N}_{t}}$, DUSt3R first constructs a connectivity graph $\mathcal{G}(\mathcal{V}, \mathcal{E})$ for selecting pairwise images, where the vertices $\mathcal{V}$ represent $N_{t}$ images and each edge $e \in \mathcal{E}$ is an image pair. Then, it estimates the depth maps $D:=\{D_{k}\}$ and camera pose $\pi:=\{\pi_{k}\}$ by:

$$\operatorname*{arg\,min}_{D,\pi,\sigma} \sum_{e\in\mathcal{E}}\sum_{v\in e} C_{v}^{e} ||D_{v} - \sigma_{e} P_{e}(\pi_{v}, X_{v}^{e})||_{2}^{2} \quad (6)$$

where $\sigma=\{\sigma_{e}\}$ are the scale factors defined on the edges, $P_{e}(\pi_{v}, X_{v}^{e})$ means projecting the predicted pointmap $X_{v}^{e}$ to view using poses $\pi_{v}$ to get a depth map. The objective function in eq. (6) explicitly constrains the geometry alignment between frame pairs, aiming to preserve cross-view consistency in the depth maps.

---

## 4. Methods

Test3R is a test-time training technique that adapts DUSt3R [12] to challenging test scenes. It improves reconstruction by maximizing cross-pair consistency. We begin by analyzing the root cause of inconsistency in Sec. 4.1. In Sec. 4.2, we establish the core problem and define the test-time training objective. Finally, we employ prompt tuning for efficient test-time adaptation in Sec. 4.3.

**Figure 3 (Overview of Test3R):** The primary goal of Test3R is to adapt a pretrained reconstruction model $f_{s}$ to the specific distribution of test scenes $f_{t}$. It achieves this goal by optimizing a set of visual prompts at test time through a self-supervised training objective that maximizes cross-pair consistency between $X_{1}^{ref,ref}$ and $X_{2}^{ref,ref}$.

### 4.1 Cross-pair Inconsistency
DUSt3R [12] aims to achieve consistency through global alignment; however, the inaccurate and inconsistent pointmaps lead to persistent errors, significantly compromising the effectiveness of global alignment. Therefore, we show a qualitative analysis of the pointmaps on the DTU [58] and ETH3D [59] datasets. Specifically, we compare the pointmap for the same reference view but paired with two different source views, and align these two pointmaps to the same coordinate system using Iterative Closest Point (ICP). The result is shown in Figure 2. On the left are two image pairs sharing the same reference view but with different source views. On the right are the corresponding pointmaps, with each color indicating the respective image pair.

**Observations.** These two predicted pointmaps of the reference view exhibit inconsistencies, as highlighted by the presence of large regions with inconsistent colors in 3D space. Ideally, if these pointmaps are consistent, they should be accurate enough to align perfectly in 3D space, resulting in a single, unified color (either blue or red). This result indicates that DUSt3R may produce different pointmaps for the same reference view when paired with different source views. In our view, this phenomenon stems from the problematic pair-wise prediction paradigm. First, since only two views are provided as input at each prediction step, the scene geometry is estimated solely based on visual correspondences between a single image pair. Therefore, the model produces inaccurate pointmaps. Second, all predicted pointmaps are mutually inconsistent individual pairs. For different image pairs, their visual correspondences are also different. As a result, DUSt3R may produce inconsistent pointmaps for the same reference view when paired with different source views due to the different correspondences. This issue significantly hinders the effectiveness of subsequent global alignment and further leads to discrepancies in the overall reconstruction. What's worse, the limited generalization capability of DUSt3R further exacerbates the above issues of low precision and cross-pair inconsistency.

### 4.2 Triplet Objective Made Consistent
The inconsistencies observed above highlight a core limitation of the pairwise prediction paradigm. Specifically, DUSt3R may produce different pointmaps for the same reference view when paired with different source views. This motivates a simple but effective idea: enforce triplet consistency across these pointmaps directly at test time, as shown in Figure 3.

**Definition.** We first describe the definition of test-time training on the 3D reconstruction task, where only images $\{I_{t}^{i}\}_{i=1}^{N_{t}}$ from the test scene are available. During training time training phase, $N_{s}$ labeled samples $\{I_{s}^{i}, \overline{X}_{s}^{i}\}_{i=1}^{N_{s}}$ collected from various scenes are given, where $I_{s}^{i} \in \mathcal{I}_{s}$ and $\overline{X}_{s}^{i} \in \overline{\mathcal{X}}_{s}$ are images and the corresponding pointmaps derived from the ground-truth depth $\overline{D}_{s} \in \overline{\mathcal{D}}_{s}$. Furthermore, we denote DUSt3R [12], parameterized by $\Theta$, as the model trained to learn the reconstruction function $f_{s}: \mathcal{I}_{s} \rightarrow \overline{\mathcal{X}}_{s}$. Subsequently, during test time training phase, only unlabeled images $\{I_{t}^{i}\}_{i=1}^{N_{t}}$ from test scene are available, where $I_{t}^{i} \in \mathcal{I}_{t}$. Our goal is to optimize the model $f_{s}$ to the specific scene $f_{t}: \mathcal{I}_{t} \rightarrow \overline{\mathcal{X}}_{t}$ at test time. This is achieved by minimizing the self-supervised training objective $l$.

Specifically, our core training objective is to maximize the geometric consistency by aligning the pointmaps of the reference view when paired with different source views. For a set of images $\{I_{t}^{i}\}_{i=1}^{N_{t}}$ from the specific scene, we consider a triplet consisting of one reference view and two different source views, denoted as $(I^{ref}, I^{src1}, I^{src2})$. Subsequently, Test3R forms two reference-source view pairs $(I^{ref}, I^{src1})$ and $(I^{ref}, I^{src2})$ from this triplet. These reference-source view pairs are then fed into the Test3R independently to predict pointmaps of reference views under different source view conditions in the same coordinate frame of $I^{ref}$, denoted as $X_{1}^{ref,ref}$ and $X_{2}^{ref,ref}$. Finally, we construct the training objective by aligning these two inconsistent pointmaps, formulated as:
$$l = ||X_{1}^{ref,ref} - X_{2}^{ref,ref}|| \quad (7)$$

With this objective, we can collectively compose triplets from a large number of views of an unseen 3D scene at test time. It guides the model to successfully resolve the limitations mentioned in Section 4.1. For inconsistencies, it ensures consistency by aligning the local two-view predictions. Meanwhile, it also pushes the predicted pointmap closer to an overall global prediction to mitigate the inaccuracy. Moreover, by optimizing for the specific scene at test time, it enables the model to adapt to the distribution of that scene.

### 4.3 Visual Prompt Tuning for Test Time Training
After the self-supervised training objective is defined, effectively modulating the model during test-time training for specific scenes remains a non-trivial challenge. During the test-time training phase, it only relies on unsupervised training objectives. However, these objectives are often noisy and unreliable, which makes the model prone to overfitting and may lead to training collapse, especially when only a limited number of images are available for the current scene. Fortunately, similar issues has been partially explored in the 2D vision community. In these works, visual prompt tuning [15] has demonstrated strong effectiveness in domain adaptation in 2D classification tasks [60]. It utilizes a set of learnable continuous parameters to learn the specific knowledge while retaining the knowledge learned from large-scale pretraining. Motivated by this, we explore the use of visual prompts as a carrier to learn the geometric consistency for specific scenes.

Specifically, we incorporate a set of learnable prompts into the encoder of DUSt3R [12]. Consider an encoder of DUSt3R with $N_{e}$ standard Vision Transformer (ViT) [57] layers, an input image is first divided into fixed-sized patches and then embedded into d-dimensional tokens $E_{0}=\{e_{0}^{k} \in \mathbb{R}^{D} | k \in N, 1 \le k \le N_{t}\}$, where $N_{t}$ is the length of image patch tokens. Subsequently, to optimize the model, we introduce a set of learnable prompt tokens $\{P_{i-1}\}_{i=1}^{N_{e}}$ into each Transformer layer. For $i$-th transformer layer, the prompt tokens are denoted as $P_{i-1}=\{p_{i-1}^{k} \in \mathbb{R}^{D} | k \in \mathbb{N}, 1 \le k \le N_{p}\}$, where $N_{p}$ is the length of prompt tokens. Therefore, the encoder layer augmented by visual prompts is formulated as:
$$[\_, E_{i}] = L_{i}([P_{i-1}, E_{i-1}]) \quad (8)$$

where $P_{i-1}$ and $E_{i-1}$ are learnable prompt tokens and image patch tokens at $i-1$-th Transformer layer.

**Test-time training.** We only fine-tune the parameters of the prompts, while all other parameters are fixed. This strategy enables our model to maximize geometric consistency by optimizing the prompts at test time while retaining the reconstruction knowledge acquired from large-scale datasets training within the unchanged backbone.

---

## 5. Experiment

We evaluate our method across a range of 3D tasks, including 3D Reconstruction (Section 5.1) and Multi-view Depth (Section 5.2). Moreover, we discuss the generality of Test3R and the prompt design (Section 5.3). Additional experiments and detailed model information, including parameter settings, test-time training overhead, and memory consumption, are provided in the appendix.

**Baselines.** Our primary baseline is DUSt3R [12], which serves as the backbone of our technique in the experiment. Subsequently, we select different baselines for the specific tasks to comprehensively evaluate the performance of our proposed method. For the 3D reconstruction task, which is the primary focus of the majority of 3R-series models, we compared our method with current mainstream approaches to evaluate its effectiveness. It includes MAST3R [13], MonST3R [16], CUT3R [35] and Spann3R [31]. All of these models are follow-up works building on the foundation established by DUSt3R [12]. Furthermore, for the multi-view task, we not only compare our model with baselines [61, 62] that do not require camera parameters but also evaluate our model against methods [9, 11, 27, 61-64, 64, 65] that rely on camera parameters or trained on datasets from the same domain to demonstrate the effectiveness of our technique.

### 5.1 3D Reconstruction
We utilize two scene-level datasets, 7Scenes [66] and NRGBD [67] datasets. We follow the experiment setting on the CUT3R [35], and employ several commonly used metrics: accuracy (Acc), completion (Comp), and normal consistency (NC) metrics. Each scene has only 3 to 5 views available for the 7Scenes [66] dataset and 2 to 4 views for NRGBD [67] dataset. This is a highly challenging experimental setup, as the overlap between images in each scene is minimal, demanding a strong scene reconstruction capability.

**Quantitative Results.** The quantitative evaluation is shown in Table 1. Compared to vanilla DUSt3R [12], our model demonstrates superior performance, outperforming DUSt3R on the majority of evaluation metrics, particularly in terms of mean accuracy and completion. Moreover, our approach achieves comparable or even superior results compared to mainstream methods. Only CUT3R [35] and MAST3R [13] outperform our approach on several metrics. This demonstrates the effectiveness of our test-time training strategy.

**Table 1:** 3D reconstruction comparison on 7Scenes and NRGBD datasets.

| Method | 7Scenes Acc (Mean) | 7Scenes Comp (Mean) | 7Scenes NC (Mean) | NRGBD Acc (Mean) | NRGBD Comp (Mean) | NRGBD NC (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MAST3R [13] | 0.189 | 0.211 | 0.766 | 0.085 | 0.028 | 0.794 |
| MonST3R [16] | 0.240 | 0.268 | 0.758 | 0.114 | 0.287 | 0.758 |
| Spann3R [31] | 0.298 | 0.112 | 0.650 | 0.416 | 0.285 | 0.684 |
| CUT3R [35] | 0.126 | 0.031 | 0.727 | 0.031 | 0.076 | 0.837 |
| DUSt3R [12] | 0.146 | 0.067 | 0.839 | 0.144 | 0.018 | 0.871 |
| **Ours** | **0.105** | **0.035** | **0.746** | **0.083** | **0.079** | **0.870** |

**Qualitative Results.** The qualitative results are shown in Figure 4. We compare our method with CUT3R [35] and DUSt3R [12] on the Office and Kitchen scenes from the 7Scenes [66] and NRGBD [67] datasets, respectively. We observe that DUSt3R incorrectly regresses the positions of scene views, leading to errors in the final scene reconstruction. In contrast, our model achieves more reliable scene reconstructions. This improvement is particularly evident in the statue in the Office scene and the wall in the Kitchen scene. For these two objects, the reconstruction results from DUSt3R are drastically different from the ground truth. Compared to CUT3R, the current state-of-the-art in 3D reconstruction, we achieve better reconstruction results. Specifically, we effectively avoid the generation of outliers, resulting in more accurate pointmaps. Details can be seen in the red bounding boxes as shown in Figure 4.

### 5.2 Multi-view Depth
Following RobustMVD [63], performances are measured on the object-centric dataset DTU [58] and scene-centric dataset ETH3D [59]. To evaluate the depth map, we report the Absolute Relative Error (rel) and the Inlier Ratio ($\tau$) at a threshold of 3% on each test set and the averages across all test sets.

**Quantitative Results.** The quantitative evaluation is shown in Table 2. On the DTU dataset, our model significantly improves upon the performance of vanilla DUSt3R, reducing the Absolute Relative Error by 1.3 and increasing the Inlier Ratio by 14.2. Similarly, on the ETH3D dataset, our model also demonstrates comparable improvements, achieving state-of-the-art performance on this challenging benchmark as well. Notably, our model surpasses the majority of methods that rely on camera poses and intrinsic parameters, and the models trained on the dataset from the same domain. This indicates that our approach effectively captures scene-specific global information and enables to adaptation of the distribution of test scenes, thereby significantly improving the quality of the depth maps.

**Table 2:** Multi-view depth evaluation. (Parentheses) denote training on data from the same domain.

| Method | GT Pose | GT Range | GT Intrinsics | Align | DTU rel $\downarrow$ | DTU $\tau$ $\uparrow$ | ETH3D rel $\downarrow$ | ETH3D $\tau$ $\uparrow$ | AVG rel $\downarrow$ | AVG $\tau$ $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| COLMAP [9, 11] | ✓ | ✓ | ✓ | - | 0.7 | 96.5 | 55.1 | 16.4 | 27.9 | 56.4 |
| COLMAP Dense [9, 11] | ✓ | ✓ | ✓ | - | 20.8 | 69.3 | 23.2 | 89.8 | 22.0 | 79.5 |
| MVSNet [27] | ✓ | ✓ | ✓ | - | (1.8) | (86.0) | 31.4 | 35.4 | 16.6 | 60.7 |
| Vis-MVSNet [64] | ✓ | ✓ | ✓ | - | (1.8) | (87.4) | 10.8 | 43.3 | 6.3 | 65.4 |
| MVS2D ScanNet [65] | ✓ | ✓ | ✓ | - | 17.2 | 9.8 | 4.8 | 27.4 | 11.0 | 18.6 |
| MVS2D DTU [65] | ✓ | ✓ | ✓ | - | (3.6) | (64.2) | 11.6 | 99.0 | 7.6 | 81.6 |
| DeMoN [61] | X | X | X | med | 21.8 | 16.6 | 15.4 | 10.1 | 18.6 | 13.3 |
| DeepV2D KITTI [62] | X | X | X | med | 24.8 | 8.1 | 17.4 | 27.1 | 21.1 | 17.6 |
| DeepV2D ScanNet [62] | X | X | ✓ | med | 7.7 | 33.0 | 11.8 | 29.3 | 9.8 | 31.1 |
| DUSt3R [12] | X | X | X | med | 3.3 | 69.9 | 73.0 | 3.3 | 3.3 | 71.5 |
| **Test3R** | **X** | **X** | **X** | **med** | **2.0** | **84.1** | **74.0** | **3.2** | **2.6** | **79.1** |

**Qualitative Results.** The qualitative result is shown in Figure 5. We present the depth map on the key view, following RobustMVD [63]. We observe that Test3R effectively improves the accuracy of depth estimation compared to DUSt3R and RobustMVD [63] with camera parameters. Specifically, Test3R captures more fine-grained details, including the computer chassis and table. Additionally, on the white-background DTU dataset, Test3R effectively understands scene context, allowing it to accurately estimate the depth of background regions.

### 5.3 Ablation Study and Analysis

#### 5.3.1 Framework Generalization.
To demonstrate the generalization ability of our proposed technique, we applied Test3R to MAST3R [13] and MonST3R [16], and evaluated the performances on the 7Scenes [66] dataset. As shown in Table 3, Test3R effectively improves the performance of MAST3R and MonST3R on 3D reconstruction task. This demonstrates the generalization ability of our technique, which can be applied to other models sharing a similar pipeline.

**Table 3:** Generalization Study.

| Method | 7Scenes Acc (Mean) | 7Scenes Comp (Mean) | 7Scenes NC (Mean) |
| :--- | :--- | :--- | :--- |
| MAST3R [13] | 0.189 | 0.211 | 0.687 |
| MAST3R (w. Test3R) | 0.179 | 0.177 | 0.702 |
| MonST3R [16] | 0.240 | 0.268 | 0.672 |
| MonST3R (w. Test3R) | 0.218 | 0.251 | 0.687 |

#### 5.3.2 Ablation on Visual Prompt.
We introduce a model variant, Test3R-S, and conduct an ablation study to evaluate the impact of visual prompts. For Test3R-S, the prompts are only inserted into the first Transformer layer, accompany the image tokens through the encoding process, and are then discarded.

**Table 4:** Ablation study on Visual Prompt.

| Variants | Prompts Length 16 (Acc/Comp) | Prompts Length 32 (Acc/Comp) | Prompts Length 64 (Acc/Comp) |
| :--- | :--- | :--- | :--- |
| Test3R-S | 0.133 / 0.142 | 0.125 / 0.150 | 0.120 / 0.158 |
| Test3R | 0.118 / 0.131 | 0.122 / 0.155 | 0.105 / 0.136 |

The result is shown in Table 4. Both Test3R-S and Test3R effectively improve model performance, compared to vanilla DUSt3R. For prompt length, we observe that when the number of prompts is small, increasing the prompt length can enhance the ability of Test3R to improve reconstruction quality. However, as the prompt length increases, the number of trainable parameters also grows, making it more challenging to converge within the same number of iterations, thereby reducing their overall effectiveness. For prompt insertion depth, we observe that Test3R, which uses distinct prompts at each layer, demonstrates superior performance. This is because the feature distributions vary across each layer of the encoder of DUSt3R, making layer-specific prompts more effective for fine-tuning. However, as the number of prompt parameters increases, Test3R becomes more susceptible to optimization challenges compared to Test3R-S, leading to a faster performance decline.

---

## 6. Conclusion

In this paper, we present Test3R, a novel yet strikingly simple solution that learns to reconstruct at test time. It maximizes the cross-pair consistency via optimizing a set of visual prompts at test time. This design successfully mitigates the reconstruction quality degradation caused by the pairwise predictions paradigm and limited generalization capability. Extensive experiments show that our simple design not only effectively improves model performance but also achieves state-of-the-art performance across various tasks. Moreover, our technique is universally applicable and nearly cost-free, which can be widely applied to different models and implemented with minimal test-time training overhead and parameter footprint.

---

## Appendices

### A. Implement details
We use PyTorch for all implementations, and our method is tested in a single RTX 4090 GPU. For the visual prompts, the prompt length is 32 for each transformer layer. For constructing triplets, we consider all available images. Considering a scene with $n$ images, the total number of triplets is $n^3$. For computational efficiency, if the total number of triplets exceeds 165, we randomly sample 165 triplets as the test-time training set. For test-time training, we adopt the Adam [68] optimizer. Given the varying number of available views in each dataset, we select different learning rates accordingly. We set the learning rate to 0.00001, 0.00008, 0.00004, and 0.00001 for 7Scenes [66], NRGBD [67], DTU [58], and ETH3D [59], respectively. We only fine-tune Test3R for 1 epoch at the specific test scene.

### B. Consumption
This section discusses the time consumption, parameter footprint, and memory allocation of Test3R. We report our result on the scene Office-seq-09 from the 7Scenes [66]. The result is shown in Table 5. Compared to the vanilla DUSt3R, as inference and parameter optimization are required for each triplet, this leads to increased test-time latency. However, only prompts are introduced in each transformer layer, resulting in negligible overhead in terms of both parameter footprint and memory consumption for Test3R. By fine-tuning these additional parameters, our model can effectively enhance the final reconstruction quality.

**Table 5:** Comparison of time consumption, number of parameters, and memory allocation.

| Metrics | Acc | Comp | NC | Time Consumption (Total / TTT) | Parameter (Prompts / Total) | Memory (Prompts / Total) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DUSt3R | 0.62 | 0.51 | 0.54 | ~10s / - | - / 571.17M | - / 2178.85M |
| Test3R | 0.14 | 0.20 | 0.69 | ~30s / ~40s | 0.79M / 571.96M | 3M / 2181.85M |

### C. Discussion and Analysis
#### C.1 Cross-pair Consistency
Our self-supervised training objective is designed to maximize cross-pair consistency of pointmap. In this section, we demonstrate the effectiveness of this objective. Specifically, we visualize the pointmaps of the same reference view but paired with different source views. The pointmaps are visualized in 2D space by projecting them onto the corresponding depthmaps, which provides a clearer representation while avoiding interference caused by viewpoint variations. The relationship between the pointmap and the depthmap is defined by the following equation:

$$X_{i,j} = K^{-1}[iD_{i,j}, jD_{i,j}, D_{i,j}] \quad (9)$$

where $(i, j) \in \{1...W\} \times \{1...H\}$ is the pixel coordinates and $K \in \mathbb{R}^{3 \times 3}$ is the camera intrinsics. $X_{i,j}$ and $D_{i,j}$ are the corresponding pointmap and depthmap. We visualize the depthmap on the scan1 from the DTU [58] dataset, as shown in Figure 6. Compared to vanilla DUSt3R, Test3R demonstrates superior consistency across different pairs. The depthmaps predicted by DUSt3R exhibit significant inconsistencies in regions with limited overlap. After optimizing by cross-pairs consistency objective, Test3R generates consistent and reliable depth predictions in these regions. Moreover, even in the $(I^{ref}, I^{ref})$ pair case, Test3R can still predict relatively consistent depth maps.

**Figure 6:** Comparison on cross-pair consistency. The depth map of the same reference view but paired with different source views. Test3R demonstrates superior cross-pair consistency compared to vanilla DUSt3R.

#### C.2 Compared to single forward-based model
We compare DUSt3R and Test3R with the current single forward-based models, Fast3R [38] and VGGT [29]. These models can process multiple images in a single forward pass, which may enhance the model's robustness and accuracy. Therefore, we report the result on the NRGBD [67] dataset, as shown in Table 6. It demonstrates that these models still struggle to generalize to unseen scenes. Fast3R shows significantly inferior reconstruction quality compared to DUSt3R and Test3R on the NRGBD dataset. Meanwhile, although VGGT achieves relatively strong performance on this dataset, it requires substantial computational resources for training and still underperforms Test3R on several metrics. These results validate the effectiveness and robustness of our model across diverse scenes. By maximizing cross-pair consistency, our model can adapt to previously unseen scenes, thereby enabling more accurate reconstruction of challenging scenes with minimal test-time training overhead and parameter footprint.

**Table 6:** Comparison with Single forward-based model.

| Method | Acc (Mean) | Comp (Mean) | NC (Mean) |
| :--- | :--- | :--- | :--- |
| Fast3R [38] | 0.215 | 0.076 | 0.695 |
| VGGT [29] | 0.377 | 0.043 | 0.901 |
| DUSt3R [12] | 0.144 | 0.018 | 0.871 |
| **Ours** | **0.083** | **0.079** | **0.870** |

### D. Limitations
While Test3R significantly improves the quality of the reconstruction on the DUSt3R, there are still some limitations. Firstly, the final reconstruction quality still heavily depends on the input images. It still struggles with in-the-wild data, which is often characterized by occlusions, dynamic objects, and varying illumination. Secondly, Test3R lacks efficient utilization of inference results. It only considers the pointmaps from the reference views, without leveraging the pointmaps from the source views. Many current baselines incorporate a camera head into the prediction stage. Therefore, using camera poses to align different viewpoints is a promising direction for future research. Thirdly, we focus on scenarios with sparse viewpoints in our study, where Test3R can consider each view. However, when the number of viewpoints increases, considering every viewpoint is computationally expensive. Therefore, how to effectively sample these views when forming triplets remains an open question.

### E. More reconstruction result
We provide more reconstruction results, as shown in Figure 7. We observe that Test3R achieves more detailed and consistent reconstructions than DUSt3R, as specifically illustrated within the red boxes. The objects, like fences and stone pillars, remain consistent under different viewpoints, demonstrating improved cross-view consistency. Furthermore, Test3R produces fewer outliers in ambiguous or low-texture regions, such as the distant trees and sky, highlighting its robustness.

---

## References

[1] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-net: A trainable cnn for joint description and detection of local features. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 8092-8101, 2019.

[2] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 224-236, 2018.

[3] David G Lowe. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60:91-110, 2004.

[4] Eric Brachmann and Carsten Rother. Neural-guided ransac: Learning where to sample model hypotheses. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4322-4331, 2019.

[5] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching at light speed. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17627-17638, 2023.

[6] Chen Zhao, Yixiao Ge, Feng Zhu, Rui Zhao, Hongsheng Li, and Mathieu Salzmann. Progressive correspondence pruning by consensus learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6464-6473, 2021.

[7] David J Crandall, Andrew Owens, Noah Snavely, and Daniel P Huttenlocher. Sfm with mrfs: Discrete-continuous optimization for large-scale structure from motion. IEEE transactions on pattern analysis and machine intelligence, 35(12):2841-2853, 2012.

[8] Philipp Lindenberger, Paul-Edouard Sarlin, Viktor Larsson, and Marc Pollefeys. Pixel-perfect structure-from-motion with featuremetric refinement. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5987-5997, 2021.

[9] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104-4113, 2016.

[10] Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong Tan, and Ping Tan. Cascade cost volume for high-resolution multi-view stereo and stereo matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2495-2504, 2020.

[11] Johannes L Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In Computer Vision-ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, pages 501-518. Springer, 2016.

[12] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697-20709, 2024.

[13] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71-91. Springer, 2024.

[14] Masatoshi Okutomi and Takeo Kanade. A multiple-baseline stereo. IEEE Transactions on pattern analysis and machine intelligence, 15(4):353-363, 1993.

[15] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European conference on computer vision, pages 709-727. Springer, 2022.

[16] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825, 2024.

[17] Shimon Ullman. The interpretation of structure from motion. Proceedings of the Royal Society of London. Series B. Biological Sciences, 203(1153):405-426, 1979.

[18] Yasutaka Furukawa, Carlos Hernández, et al. Multi-view stereo: A tutorial. Foundations and trends® in Computer Graphics and Vision, 9(1-2):1-148, 2015.

[19] Silvano Galliani, Katrin Lasinger, and Konrad Schindler. Massively parallel multiview stereopsis by surface normal diffusion. In Proceedings of the IEEE international conference on computer vision, pages 873-881, 2015.

[20] Yuesong Wang, Zhaojie Zeng, Tao Guan, Wei Yang, Zhuo Chen, Wenkai Liu, Luoyuan Xu, and Yawei Luo. Adaptive patch deformation for textureless-resilient multi-view stereo. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1621-1630, 2023.

[21] Qiancheng Fu, Qingshan Xu, Yew Soon Ong, and Wenbing Tao. Geo-neus: Geometry-consistent neural implicit surfaces learning for multi-view reconstruction. Advances in Neural Information Processing Systems, 35:3403-3416, 2022.

[22] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3504-3515, 2020.

[23] Yi Wei, Shaohui Liu, Yongming Rao, Wang Zhao, Jiwen Lu, and Jie Zhou. Nerfingmvs: Guided optimization of neural radiance fields for indoor multi-view stereo. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5610-5619, 2021.

[24] Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, and Yaron Lipman. Multiview neural surface reconstruction by disentangling geometry and appearance. Advances in Neural Information Processing Systems, 33:2492-2502, 2020.

[25] Zeyu Ma, Zachary Teed, and Jia Deng. Multiview stereo with cascaded epipolar raft. In European Conference on Computer Vision, pages 734-750. Springer, 2022.

[26] Rui Peng, Rongjie Wang, Zhenyu Wang, Yawen Lai, and Ronggang Wang. Rethinking depth estimation for multi-view stereo: A unified representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8645-8654, 2022.

[27] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pages 767-783, 2018.

[28] Zhe Zhang, Rui Peng, Yuxi Hu, and Ronggang Wang. Geomvsnet: Learning multi-view stereo with geometry perception. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21508-21518, 2023.

[29] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. arXiv preprint arXiv:2503.11651, 2025.

[30] Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnan Fan, and Baoquan Chen. Slam3r: Real-time dense scene reconstruction from monocular rgb videos. arXiv preprint arXiv:2412.09401, 2024.

[31] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061, 2024.

[32] Wonbong Jang, Philippe Weinzaepfel, Vincent Leroy, Lourdes Agapito, and Jerome Revaud. Pow3r: Empowering unconstrained 3d reconstruction with camera and scene priors. arXiv preprint arXiv:2503.17316, 2025.

[33] Jiahao Lu, Tianyu Huang, Peng Li, Zhiyang Dou, Cheng Lin, Zhiming Cui, Zhen Dong, Sai-Kit Yeung, Wenping Wang, and Yuan Liu. Align3r: Aligned monocular depth estimation for dynamic videos. arXiv preprint arXiv:2412.03079, 2024.

[34] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3r: Estimating disentangled motion from dust3r without training. arXiv preprint arXiv:2503.24391, 2025.

[35] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. arXiv preprint arXiv:2501.12387, 2025.

[36] Shizun Wang, Xingyi Yang, Qiuhong Shen, Zhenxiang Jiang, and Xinchao Wang. Gflow: Recovering 4d world from monocular video. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7862-7870, 2025.

[37] Jie Hu, Shizun Wang, and Xinchao Wang. Pe3r: Perception-efficient 3d reconstruction. arXiv preprint arXiv: 2503.07507, 2025.

[38] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. arXiv preprint arXiv:2501.13928, 2025.

[39] Alex Gammerman, Volodya Vovk, and Vladimir Vapnik. Learning by transduction. arXiv preprint arXiv: 1301.7375, 2013.

[40] Vladimir Vapnik. Estimation of dependences based on empirical data. Springer Science & Business Media, 2006.

[41] Ronan Collobert, Fabian Sinz, Jason Weston, Léon Bottou, and Thorsten Joachims. Large scale transductive svms. Journal of Machine Learning Research, 7(8), 2006.

[42] Thorsten Joachims. Learning to classify text using support vector machines, volume 668. Springer Science & Business Media, 2002.

[43] Moritz Hardt and Yu Sun. Test-time training on nearest neighbors for large language models. arXiv preprint arXiv:2305.18466, 2023.

[44] Léon Bottou and Vladimir Vapnik. Local learning algorithms. Neural computation, 4(6):888-900, 1992.

[45] Hao Zhang, Alexander C Berg, Michael Maire, and Jitendra Malik. Svm-knn: Discriminative nearest neighbor classification for visual category recognition. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06), volume 2, pages 2126-2136. IEEE, 2006.

[46] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with self-supervision for generalization under distribution shifts. In International conference on machine learning, pages 9229-9248. PMLR, 2020.

[47] Nicklas Hansen, Rishabh Jangir, Yu Sun, Guillem Alenyà, Pieter Abbeel, Alexei A Efros, Lerrel Pinto, and Xiaolong Wang. Self-supervised policy adaptation during deployment. arXiv preprint arXiv:2007.04309, 2020.

[48] Yu Sun, Wyatt L Ubellacker, Wen-Loong Ma, Xiang Zhang, Changhao Wang, Noel V Csomay-Shanklin, Masayoshi Tomizuka, Koushil Sreenath, and Aaron D Ames. Online learning of unknown dynamics for model-based controllers in legged locomotion. IEEE Robotics and Automation Letters, 6(4):8442-8449, 2021.

[49] Yuejiang Liu, Parth Kothari, Bastien Van Delft, Baptiste Bellot-Gurlet, Taylor Mordan, and Alexandre Alahi. Ttt++: When does self-supervised test-time training fail or thrive? Advances in Neural Information Processing Systems, 34:21808-21820, 2021.

[50] Longhui Yuan, Binhui Xie, and Shuang Li. Robust test-time adaptation in dynamic scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15922-15932, 2023.

[51] Ben Mann, N Ryder, M Subbiah, J Kaplan, P Dhariwal, A Neelakantan, P Shyam, G Sastry, A Askell, S Agarwal, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 1:3, 2020.

[52] Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423-438, 2020.

[53] Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. arXiv preprint arXiv: 2010.15980, 2020.

[54] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv: 2104.08691, 2021.

[55] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.

[56] Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks. arXiv preprint arXiv:2110.07602, 2021.

[57] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[58] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis, Engin Tola, and Anders Bjorholm Dahl. Large-scale data for multiple-view stereopsis. International Journal of Computer Vision, 120:153-168, 2016.

[59] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3260-3269, 2017.

[60] Yunhe Gao, Xingjian Shi, Yi Zhu, Hao Wang, Zhiqiang Tang, Xiong Zhou, Mu Li, and Dimitris N Metaxas. Visual prompt tuning for test-time domain adaptation. arXiv preprint arXiv:2210.04831, 2022.

[61] Benjamin Ummenhofer, Huizhong Zhou, Jonas Uhrig, Nikolaus Mayer, Eddy Ilg, Alexey Dosovitskiy, and Thomas Brox. Demon: Depth and motion network for learning monocular stereo. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5038-5047, 2017.

[62] Zachary Teed and Jia Deng. Deepv2d: Video to depth with differentiable structure from motion. arXiv preprint arXiv:1812.04605, 2018.

[63] Philipp Schröppel, Jan Bechtold, Artemij Amiranashvili, and Thomas Brox. A benchmark and a baseline for robust multi-view depth estimation. In 2022 International Conference on 3D Vision (3DV), pages 637-645. IEEE, 2022.

[64] Jingyang Zhang, Shiwei Li, Zixin Luo, Tian Fang, and Yao Yao. Vis-mvsnet: Visibility-aware multi-view stereo network. International Journal of Computer Vision, 131(1):199-214, 2023.

[65] Zhenpei Yang, Zhile Ren, Qi Shan, and Qixing Huang. Mvs2d: Efficient multi-view stereo via attention-driven 2d convolutions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8574-8584, 2022.

[66] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgb-d images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2930-2937, 2013.

[67] Dejan Azinović, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. Neural rgb-d surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6290-6301, 2022.

[68] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.