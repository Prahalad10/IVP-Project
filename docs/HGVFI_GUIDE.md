# Technical foundations of HGVFI: a comprehensive reference

HGVFI (Hint-Guided Video Frame Interpolation for Video Compression) builds on a rich stack of architectural components — from the EMA-VFI backbone that extracts motion and appearance simultaneously via inter-frame attention, to PixelShuffle-based upsampling, residual attention blocks, U-Net feature extractors, and a compression paradigm that competes with the DCVC family of neural codecs. This report provides the implementation-level technical details a researcher needs to replicate and extend every component of the HGVFI system.

---

## 1. EMA-VFI: unified motion-appearance extraction through inter-frame attention

EMA-VFI (Zhang et al., CVPR 2023) is the backbone VFI network in HGVFI. Its central contribution is a **parallel extraction paradigm** that uses a single inter-frame attention operation to simultaneously produce both motion and appearance features — avoiding the redundancy of sequential designs (separate flow network + context network) and the ambiguity of mixed designs (concatenate-and-convolve).

### Architecture overview

The pipeline has five hierarchical stages in a hybrid CNN + Transformer design. Given input frames $I_0, I_1$, a shared convolutional feature extractor $F$ generates multi-scale features independently for each frame:

$$L_i^0, L_i^1, L_i^2 = F(I_i)$$

**Stages 0–2 (CNN):** Each stage uses a ConvBlock with `depths[k]=2` convolutional layers and PReLU activations, followed by stride-2 convolution for downsampling. Channel dimensions double at each level. For the large model (C=32): **32 → 64 → 128 → 256 → 512** across all five stages. For the compact model (C=16): **16 → 32 → 64 → 128 → 256**.

**CrossScalePatchEmbed:** Before entering Transformer stages, multi-scale CNN features are fused through multi-scale dilated convolutions (with dilation rates from 1 to $2^{2-k}$ per scale $k$), concatenated, and projected via a linear layer into cross-scale appearance features $C_i$.

**Stages 3–4 (Transformer):** These stages apply Swin-Transformer-style shifted window attention with the novel Inter-Frame Attention (IFA) mechanism. The large model uses 4 blocks per stage; the compact model uses 2. Window size is **7×7** (ablated: 5 and 9 perform worse). Attention heads: 8 at stage 3, 16 at stage 4. MLP expansion ratio: 4. Position encoding uses depth-wise convolution in the MLP (following PvT v2), not learned positional embeddings.

### The inter-frame attention mechanism

For appearance features $A_0, A_1 \in \mathbb{R}^{\hat{H} \times \hat{W} \times C}$ from the two frames, the IFA computes attention within local windows:

$$Q_0^{i,j} = A_0^{i,j} W_Q, \quad K_1^{n_{i,j}} = A_1^{n_{i,j}} W_K, \quad V_1^{n_{i,j}} = A_1^{n_{i,j}} W_V$$

$$S_{0 \to 1}^{i,j} = \text{softmax}\left(\frac{Q_0^{i,j} \cdot (K_1^{n_{i,j}})^\top}{\sqrt{\hat{C}}}\right) \in \mathbb{R}^{N \times N}$$

where $n_{i,j}$ denotes the $N \times N$ neighborhood in frame 1 around position $(i,j)$. The attention map $S$ serves a dual purpose:

**Appearance enhancement** (standard attention output with residual):
$$\hat{A}_0^{i,j} = A_0^{i,j} + S_{0 \to 1}^{i,j} \cdot V_1^{n_{i,j}}$$

**Motion extraction** (the novel contribution): A coordinate map $B \in \mathbb{R}^{\hat{H} \times \hat{W} \times 2}$ stores normalized spatial coordinates. The motion vector is the attention-weighted displacement:

$$M_{0 \to 1}^{i,j} = S_{0 \to 1}^{i,j} \cdot B^{n_{i,j}} - B^{i,j} \in \mathbb{R}^2$$

This computes a soft correspondence: the weighted average of neighbor positions minus the query's own position yields an approximate displacement vector. For arbitrary timestep $t$, motion scales linearly: $M_{0 \to t} = t \cdot M_{0 \to 1}$, enabling single-computation multi-timestep interpolation.

### Motion estimation and frame synthesis

Motion features from each Transformer stage are concatenated with appearance features, upsampled via **two sequential PixelShuffle(2) operations** (4× total), and combined with warped original images plus previous flow/mask estimates. Only **3 convolution layers** per iteration produce residual updates to bidirectional flows $F_{t \to 0}, F_{t \to 1}$ and a fusion mask $O$. The final blended frame is:

$$\tilde{I}_t = O \odot \text{BW}(I_0, F_{t \to 0}) + (1-O) \odot \text{BW}(I_1, F_{t \to 1})$$

A simplified U-Net **RefineNet** then produces a residual correction: $\hat{I}_t = \tilde{I}_t + \text{RefineNet}(\tilde{I}_t, L, A)$, injecting low-level CNN features $L$ and inter-frame appearance features $A$ at corresponding scales.

### Compact variant (EMA-VFI-S)

The compact model halves initial channels (C=16 vs 32) and Transformer depth (2 blocks vs 4 per stage) while keeping the same attention mechanism and window size. The performance-speed tradeoff is significant:

| Config | Vimeo90K PSNR | Runtime (256²) | Memory (256²) |
|--------|--------------|-----------------|---------------|
| Large (C=32, N=4/4) | **36.50 dB** | 56 ms / 1.49 GB | — |
| Compact (C=16, N=2/2) | 36.07 dB | **13 ms / 1.14 GB** | — |

The compact variant runs at **4.3× the speed** with only 0.43 dB quality loss, making it practical for compression pipelines where the VFI model runs at decode time. Training uses AdamW with cosine annealing (2e-4 → 2e-5 over 300 epochs), batch size 32, 256×256 crops, and a combined Laplacian loss: $\mathcal{L} = \mathcal{L}_{rec} + 0.5 \sum_i \mathcal{L}_{warp}^i$.

---

## 2. PixelShuffle: learned upsampling without artifacts

PixelShuffle (Shi et al., CVPR 2016) rearranges a low-resolution tensor with many channels into a high-resolution tensor with fewer channels. It is the dominant upsampling method in modern learned image/video processing and appears throughout HGVFI's pipeline.

### Mathematical formulation

Given an input tensor $T$ of shape $(B, C \cdot r^2, H, W)$ and upscaling factor $r$, PixelShuffle produces output of shape $(B, C, rH, rW)$. The exact index mapping is:

$$\text{PS}(T)_{b,c,y,x} = T_{b,\; C \cdot r \cdot (y \bmod r) + C \cdot (x \bmod r) + c,\; \lfloor y/r \rfloor,\; \lfloor x/r \rfloor}$$

In PyTorch terms: `x.view(B, C, r, r, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C, H*r, W*r)`. Every $r \times r$ block of output pixels is drawn from $r^2$ different channels at the same LR spatial location. A complete sub-pixel convolution layer first applies a standard convolution with $C_{out} \cdot r^2$ output channels, then reshuffles.

### Why PixelShuffle beats the alternatives

**Transposed convolutions** suffer from checkerboard artifacts caused by uneven output-pixel overlap when kernel size is not divisible by stride — different output pixels receive contributions from different numbers of input elements, producing a characteristic grid pattern. PixelShuffle operates as stride-1 convolution followed by pure data reorganization, so **every output pixel receives identical computational treatment**, eliminating overlap-induced artifacts entirely.

**Bilinear/bicubic upsampling** uses fixed, handcrafted filters. PixelShuffle learns $r^2$ distinct upsampling filters per output channel, optimized end-to-end for the task. This yields +0.15 dB on images and +0.39 dB on video over prior CNN approaches in the original ESPCN paper.

**Computational efficiency** is the third advantage. Previous super-resolution approaches (SRCNN, VDSR) first upsampled to HR resolution via bicubic interpolation, then applied all convolutions in HR space. ESPCN performs all feature extraction in **LR space** and upsamples only at the final step, reducing computation by approximately $r^2$. With ICNR initialization (repeating sub-kernels), even initialization-related artifacts vanish.

### Usage in HGVFI

In HGVFI, PixelShuffle appears in multiple roles. The **learnable upsampling module** in the hint branch progressively upsamples hint features — for 4× upsampling, two sequential PixelShuffle(2) stages with intermediate convolutions are used rather than a single PixelShuffle(4), which provides richer intermediate representations. In EMA-VFI's motion estimation, two PixelShuffle(2) operations achieve 4× upsampling from the Transformer's low-resolution features to the flow estimation resolution. The pattern `Conv2d(C_in, C_out·4, 3×3) → PixelShuffle(2)` serves as the standard upsampling block throughout.

---

## 3. Residual Attention Block: global context for feature refinement

The Residual Attention Block (RAB), as used in image restoration networks like DRANet (Wu et al., 2024), combines residual learning with attention-based feature reweighting. In HGVFI, RABs refine upsampled features in the hint branch and decoder path.

### Architecture

A RAB consists of a **residual convolutional path** followed by an **attention module**. The convolutional path stacks two or more `Conv(3×3) → ReLU → Conv(3×3)` pairs with short skip connections (element-wise addition) between them. The attention module then gates the output:

**Spatial Attention (SAM)** variant (DRANet): Global Average Pooling and Global Max Pooling are applied along the channel dimension, their outputs concatenated, passed through `Conv(7×7) → Sigmoid`, producing a spatial attention map $M \in [0,1]^{H \times W}$. The output is $F_{out} = F_{res} \odot M$.

**Channel Attention (SE/RCAB)** variant (RCAN, Zhang et al. 2018): Global Average Pooling compresses spatial dimensions to a $C$-dimensional vector, which passes through a bottleneck MLP: `FC(C → C/r) → ReLU → FC(C/r → C) → Sigmoid`, producing channel attention weights $s \in [0,1]^C$. The output is $\hat{X} = X \odot s$. The reduction ratio $r$ is typically 16.

### Why attention captures global context

A plain residual block with two 3×3 convolutions has a **5×5 local receptive field**. The attention mechanism's Global Average Pooling compresses the entire $H \times W$ spatial extent into a single statistic per channel — giving the block **global receptive field in one operation**, regardless of network depth. This allows the block to make globally-informed decisions about which features to emphasize and suppress. DRANet's paper states explicitly: "convolutional layers can only extract local information… the attention mechanism is applied to capture and learn global context information."

### Role in upsampling pipelines

RABs serve two critical functions in decoder paths. **Pre-upsampling refinement**: RABs before PixelShuffle ensure LR features are maximally informative before being reshuffled to HR space (the RCAN pattern: RCAB body → PixelShuffle tail). **Post-upsampling cleanup**: RABs after upsampling suppress artifacts and emphasize high-frequency details that the upsampling operation may have introduced or lost. In HGVFI specifically, RABs refine hint features at each scale of the U-Net decoder and perform post-fusion refinement after combining features from the main VFI branch and hint branch. A typical pipeline: `[RAB × N] → Conv → PixelShuffle(2) → [RAB × M] → Conv → PixelShuffle(2) → Conv(C, 3, 3×3) → RGB output`.

---

## 4. U-Net: encoder-decoder feature extraction for VFI

The U-Net architecture (Ronneberger et al., 2015) provides the encoder-decoder structure used in HGVFI's context-aware feature extraction module, adapted significantly from its segmentation origins.

### Classic structure

The original U-Net has a symmetric encoder-decoder with skip connections. The **encoder** (contracting path) applies 4 blocks of `[Conv(3×3) → ReLU] × 2 → MaxPool(2×2)`, doubling channels at each level: **64 → 128 → 256 → 512 → 1024** (bottleneck). The **decoder** (expansive path) mirrors this with transposed convolutions for upsampling and channel-wise concatenation of encoder features via skip connections. Skip connections are the key innovation: they pass fine-grained spatial features from encoder to decoder, combining **"what"** (semantic context from deep layers) with **"where"** (spatial precision from shallow layers).

### VFI-specific adaptations

Video frame interpolation U-Nets differ from segmentation U-Nets in several important ways. **Inputs** are multi-channel: concatenation of two frames, warped frames, flow fields, and masks (6–20+ channels vs. 3). **Outputs** are optical flows, fusion masks, and residual images rather than class probabilities. **Normalization is typically omitted** — RIFE explicitly removes BatchNorm because it distorts motion statistics. **LeakyReLU or PReLU** replaces ReLU for better gradient flow. **Channel counts are lighter** (32→64→128→256 vs. 64→1024) for real-time inference. **Downsampling** uses strided convolutions or average pooling rather than max pooling.

Super SloMo (Jiang et al., 2018) is the canonical VFI U-Net: it uses two U-Nets — one for bidirectional flow estimation, one for flow refinement and occlusion handling — with 6-level encoders, LeakyReLU (α=0.1), and average pooling downsampling. RIFE's IFNet takes a coarse-to-fine approach reminiscent of U-Net decoders: 3 stacked IFBlocks at 1/4, 1/2, and full resolution, each refining flow residuals with only 3×3 convolutions.

### HGVFI's context-aware feature extraction module

In HGVFI, the hint branch employs a **U-Net-based Context-aware Feature Extraction Module**. The encoder processes the hint frame (a downsampled, compressed version of the target) at progressively lower resolutions, extracting features at each scale. Progressively upsampled hint features from a learnable upsampling module (PixelShuffle-based) are **injected into both encoder and decoder paths via skip connections**, enabling multi-scale contextual feature reuse. The decoder progressively refines these features using three information sources: bottleneck features (high-level semantics), encoder skip connections (fine spatial details), and injected hint features at each scale (structural guidance from the compressed target). This multi-source fusion enables robust guidance extraction even when hints are heavily compressed.

---

## 5. The DCVC family: conditional coding surpasses residual coding

The DCVC (Deep Contextual Video Compression) family from Microsoft Research Asia represents the state of the art in neural video compression. Understanding it contextualizes HGVFI's position in the compression landscape.

### The conditional coding paradigm

DCVC's foundational insight is information-theoretic: **$H(x_t - \tilde{x}_t) \geq H(x_t | \tilde{x}_t)$**. Encoding the residual between predicted and actual frames (as DVC does) is provably suboptimal compared to encoding the frame conditioned on temporal context. In DCVC, the current frame $x_t$ passes through a **contextual encoder** that takes both $x_t$ and temporal context $\bar{x}_t$ as input:

$$\hat{x}_t = f_{dec}\left(\lfloor f_{enc}(x_t | \bar{x}_t) \rceil \mid \bar{x}_t\right)$$

The context $\bar{x}_t$ is a **high-dimensional feature representation** (e.g., 64 channels at full resolution) extracted from the previously decoded frame via motion-compensated feature warping: $\bar{x}_t = f_{cr}(\text{warp}(f_{fe}(\hat{x}_{t-1}), \hat{m}_t))$. Unlike a 3-channel pixel prediction, these multi-channel features carry far richer information — different channels specialize in textures, edges, colors, and structural patterns.

The entropy model fuses three priors to estimate latent probabilities: a **hyperprior** (side information), a **spatial/autoregressive prior** (from already-decoded positions), and a **temporal prior** (from context $\bar{x}_t$). The temporal prior is fully parallel and unique to DCVC.

### Why conditional coding wins

Ablation evidence from the original DCVC paper is striking: residual coding requires **12.9% more bitrate** than the full DCVC system. Even adding a temporal prior to residual coding still costs 11.2% more. The reasons are threefold. First, residual coding discards the prediction signal after subtraction, while conditional coding can adaptively weight its use. Second, for occluded or newly appearing content, the residual can be enormous — conditional coding learns to fall back to near-intra coding in these regions. Third, feature-domain context carries richer, multi-dimensional information than pixel-domain prediction.

### Evolution of the family

**DCVC** (NeurIPS 2021) established conditional coding, saving 26% bitrate over x265 veryslow. **DCVC-TCM** (2022) introduced multi-scale temporal context mining and feature propagation across frames (not just pixel-domain references), dropping the slow autoregressive prior. **DCVC-HEM** (2022) added a latent prior chain ($\hat{y}_{t-1}$ feeds into the entropy model for $\hat{y}_t$), content-adaptive quantization, and became the **first NVC to exceed H.266/VTM**, saving 18.2% bitrate over VTM on UVG. **DCVC-DC** (CVPR 2023) introduced hierarchical quality structures across frames, group-based offset diversity for complex motion, and quadtree spatial partitioning, achieving the **first NVC to exceed ECM** (the next-generation traditional codec prototype). **DCVC-FM** (CVPR 2024) added learned quantization scalers supporting an **11.4 dB PSNR range** in a single model, periodic temporal feature refresh every 32 frames to break error accumulation, and 29.7% bitrate savings over DCVC-DC. The latest **DCVC-RT** (CVPR 2025) achieves real-time 1080p encoding at 125 fps.

### Where HGVFI fits relative to DCVC

The two approaches occupy fundamentally different design spaces. DCVC encodes **every frame** through a learned latent bottleneck with explicit motion coding and sophisticated entropy models — a general-purpose solution. HGVFI transmits only **keyframes** (full resolution, traditional codec) and **hint frames** (low resolution, traditional codec), reconstructing intermediate frames via VFI. HGVFI saves bits by avoiding latent representations for intermediate frames entirely but relies on the VFI model's ability to synthesize content, with hints as a safety net. For simple motion, VFI may be more bit-efficient; for complex motion and occlusions, DCVC's explicit conditional coding is more robust. HGVFI is inherently a **B-frame (bidirectional)** strategy, while DCVC is primarily P-frame (unidirectional) — a potential complementary relationship in a hybrid GOP structure.

---

## 6. From hierarchical interpolation to hint-guided compression

Wu, Singhal, and Krähenbühl (ECCV 2018) first demonstrated that video compression can be reframed as repeated image interpolation, achieving performance on par with H.264. HGVFI builds on this foundation but solves its critical weakness.

### The hierarchical interpolation structure

Wu et al.'s pipeline designates every 12th frame as a **keyframe** (I-frame), encoded using a progressive recurrent image codec with Conv-LSTMs. Intermediate frames are reconstructed via a dyadic hierarchy:

- **Level 1**: Frame 6 is interpolated from keyframes {0, 12} using model $M_{6,6}$
- **Level 2**: Frames 3 and 9 from {0, 6} and {6, 12} using $M_{3,3}$
- **Level 3**: Remaining frames from nearest decoded neighbors using $M_{1,2}$/$M_{2,1}$

Each level uses all previously decompressed frames as references. The interpolation network uses a **U-Net context network** extracting multi-resolution features, motion-compensated warping, and a progressive residual encoder/decoder with binary bottleneck codes ($L=8$ bits for short distances, $L=16$ for longer). Motion is stored as lossless WebP images of block motion estimates. Rate allocation uses a beam-search heuristic across hierarchy levels.

### The compounding error problem

Wu et al. acknowledge a fundamental limitation: **"every level in our hierarchical interpolation compounds error. Error propagation for more than three levels significantly reduces performance."** Frames interpolated at higher levels serve as references for lower levels, so any interpolation error cascades. This limits GOP size and quality floor.

### HGVFI's hint-frame solution

HGVFI eliminates hierarchical error propagation entirely. Instead of relying on previously interpolated (potentially error-corrupted) frames, it transmits a **low-resolution, compressed hint of each target frame** — an actual downsampled version of the ground truth, encoded at very low bitrate with a traditional codec. The VFI model receives both neighboring decoded keyframes and the hint frame, using the hint's structural cues to resolve motion ambiguities and occlusions that pure interpolation cannot handle. Key advantages: no error compounding across hierarchy levels, a single trained VFI model works across all configurations (vs. separate $M_{d_1,d_2}$ models per temporal offset), natural integration with existing codec infrastructure, and robust handling of complex motion and occlusions where the hint "effectively compensate[s] for reduced keyframe quality."

---

## 7. Rate-distortion evaluation: CRF sweeps, BD-Rate, and the UVG dataset

Rigorous evaluation methodology is essential for comparing HGVFI against traditional and neural codecs.

### CRF sweep generates rate-distortion curves

**Constant Rate Factor (CRF)** is the default quality-control mode in x264/x265. It targets constant perceptual quality by dynamically adjusting the Quantization Parameter (QP) per frame and macroblock — complex frames get higher QP, simple frames get lower QP. Range is **0–51** (lower = higher quality); x264 defaults to CRF 23, x265 to CRF 28. A ±6 change roughly halves or doubles file size. A **CRF sweep** encodes the same video at multiple CRF values (e.g., {15, 19, 23, 27, 31}), producing multiple (bitrate, PSNR) points that form the RD curve. For standardized academic benchmarking, **Constant QP** mode is often preferred over CRF because it provides deterministic quantization without rate-control algorithms affecting fair comparison. JVET Common Test Conditions specify QP ∈ {22, 27, 32, 37}.

### BD-Rate computation

**Bjøntegaard Delta Rate (BD-Rate)** summarizes the average bitrate difference between two RD curves as a single percentage. The computation follows these steps:

1. Obtain ≥4 RD points $(R_i, D_i)$ for both anchor and test codec
2. Convert rates to log domain: $R \to \log_{10}(R)$
3. Fit piecewise cubic interpolation (PCHIP or Akima recommended over the original cubic spline) through each codec's points
4. Determine overlapping distortion range: $D_{min} = \max(D_{A,1}, D_{B,1})$, $D_{max} = \min(D_{A,N}, D_{B,N})$
5. Integrate and compute:

$$\text{BD-Rate (\%)} = \left[10^{\frac{\int_{D_{min}}^{D_{max}} \log R_B(D) \, dD - \int_{D_{min}}^{D_{max}} \log R_A(D) \, dD}{D_{max} - D_{min}}} - 1\right] \times 100$$

A **negative BD-Rate** means the test codec saves bitrate (is better); e.g., BD-Rate = −20% means 20% fewer bits at equivalent quality. **BD-PSNR** is the dual metric: average quality gain (dB) at equivalent bitrate. Open-source implementations include the Python `bjontegaard` package (supporting cubic, PCHIP, and Akima interpolation).

### The UVG dataset

The **Ultra Video Group (UVG) dataset** is the standard benchmark for neural video compression research. It contains **16 sequences** at **3840×2160 (4K)** resolution, also available downsampled to 1920×1080, captured with a Sony F65 cinema camera in raw 8-bit and 10-bit 4:2:0 YUV. The commonly used **7-sequence subset** (120 fps) includes Beauty, Bosphorus, HoneyBee, Jockey, ReadySetGo, ShakeNDry, and YachtRide — this is the subset Wu et al. 2018 used and that most neural compression papers report on. An additional 9 sequences at 50 fps (CityAlley, FlowerFocus, FlowerKids, FlowerPan, Lips, RaceNight, RiverBank, SunBath, Twilight) complete the full set. UVG is preferred because it is openly licensed (CC-BY-NC), provides diverse content characteristics (static to high-motion, indoor to outdoor), and complements the HEVC/VVC Common Test Conditions test sets.

RD curves plot **bpp** (bits per pixel per frame: $\text{bpp} = \text{total bits} / (N_{frames} \times W \times H)$) on the x-axis against **PSNR** ($10 \log_{10}(\text{MAX}^2 / \text{MSE})$ dB) or **MS-SSIM** (often converted to dB as $-10 \log_{10}(1 - \text{MS-SSIM})$) on the y-axis. Standard anchor codecs include **x265** (practical HEVC encoder, medium or veryslow preset), **HM** (HEVC reference software), and **VTM** (VVC/H.266 reference software, the strongest traditional anchor). HGVFI reports results against H.264 and HEVC, showing consistent gains on UVG sequences particularly for complex-motion content like YachtRide and ReadySetGo.

---

## Conclusion

The HGVFI system integrates components spanning five years of rapid progress in learned video processing. **EMA-VFI's inter-frame attention** provides the critical insight that a single attention map can yield both motion vectors (via coordinate-weighted averaging) and appearance features (via value aggregation) — eliminating the need for separate optical flow networks. **PixelShuffle** enables artifact-free learned upsampling throughout the pipeline at $r^2 \times$ lower computational cost than processing in HR space. **Residual Attention Blocks** inject global context into local convolutions through a single pooling operation, essential for refining features at each decoder scale. The **U-Net** encoder-decoder with skip connections provides the multi-scale feature extraction backbone for the hint branch, adapted from segmentation with lighter channels, no batch normalization, and multi-source feature injection.

The most important design insight for extending HGVFI is the fundamental tradeoff it navigates: the DCVC family shows that conditional coding in learned latent spaces achieves state-of-the-art compression (DCVC-FM saves 25.5% over VTM), but requires encoding every frame through a neural bottleneck. HGVFI's hint-guided approach avoids this by leveraging VFI to synthesize intermediate frames "for free" and transmitting only lightweight structural hints. The compounding-error problem that limited Wu et al.'s hierarchical approach to 3 levels is solved not by better interpolation but by better side information — a lesson that extends to future work on combining VFI with learned codecs in hybrid architectures.
