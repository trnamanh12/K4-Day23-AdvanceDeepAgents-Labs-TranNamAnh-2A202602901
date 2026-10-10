# Comprehensive Survey on Video and Multimodal Generation

## TL;DR
Video and multimodal generation has transitioned from simple 2D image inflation and small-scale UNet architectures to highly scalable Spatio-Temporal Diffusion Transformers (DiTs), unified MM-DiT frameworks, and advanced autoregressive (AR) world models [1]. Modern research focuses on high-resolution latent spaces via specialized video VAEs [2][3], compositional camera-trajectory and human-motion controls [4][5], physical simulator in-the-loop alignment [6][7], and closed-loop agentic memory structures for long-horizon consistency [8][9]. Despite these breakthroughs, challenges remain in mitigating exposure bias during streaming rollouts, balancing gradient flows across autoregressive blocks [10][11], and achieving real-time interaction without sacrificing temporal coherence.

## Background and Evolution of Video Generative Foundations
The field of video and multimodal generation builds directly upon advances in large-scale image diffusion models and autoregressive sequence modeling. Early approaches attempted to adapt 2D image foundations by inflating convolutional filters or appending rudimentary temporal layers, but frequently suffered from high computational overhead, high-frequency flickering, and limited temporal receptive fields [1]. 

To overcome these constraints, modern architectures leverage spatio-temporal factorized Diffusion Transformers (DiTs) (such as VDT, Latte, and W.A.L.T) which decouple spatial modeling from temporal attention to optimize computational efficiency [1][12]. Concurrently, multimodal Diffusion Transformers (MM-DiT) like CogVideoX, HunyuanVideo, and FullDiT integrate rich textual and visual conditioning via unified stream or dual-to-single stream designs [1]. Autoregressive (AR) diffusion models factorize video generation into sequential conditional probabilities across temporal chunks, though they historically suffered from exposure bias during long rollouts—a limitation actively addressed by recent training paradigms like CausVid, Diffusion Forcing, and Self-Forcing [1].

## Latent Spaces, Video VAEs, and Spatio-Temporal Compression
Operating directly in pixel space for high-resolution video generation is computationally prohibitive. Consequently, specialized video variational autoencoders (Video-VAEs) have become foundational components of state-of-the-art video generation pipelines. 

The Compatible Video VAE (CV-VAE) framework enables seamless integration with existing pre-trained image VAEs, ensuring robust spatio-temporal compression without retraining the entire generative backbone from scratch [2]. Similarly, architectures like LTX-Video combine customized Video-VAEs with high-throughput denoising transformer backbones to achieve real-time latent diffusion while preserving fine-grained temporal consistency [3]. Furthermore, mask-modeling paradigms such as VDT capture complex temporal dependencies through general-purpose spatio-temporal attention blocks operating across compressed latent spaces [13].

## Compositional Control: Camera Trajectories and Human Dynamics
Controllability remains a critical requirement for practical video generation and interactive world simulation. Recent advances target precise 3D camera control and articulated human motion through dedicated conditioning networks and large-scale datasets.

For instance, the Directing World framework establishes a fast autoregressive world model featuring compositional human-motion and camera-trajectory control, utilizing a Fast-Slow Memory training strategy and timestep-guided dynamic projections to align features across denoising steps [4]. Complementing this, the Vision-Language-Camera (VLC) model CT-1 transfers spatial reasoning to video generation by estimating accurate camera trajectories and employing a wavelet-based regularization loss (WavReg) on SE(3) manifolds to eliminate high-frequency jitter [5]. In parallel, AC3D investigates optimized pose conditioning schedules and selective camera conditioning to enhance 3D camera control in video diffusion transformers without degrading structural fidelity [14].

## Physical Simulation Alignment and Dynamic World Simulation
Ensuring that generated video content obeys real-world physical laws and maintains structural consistency is essential for advanced simulation. Emerging techniques integrate physical simulators directly into the training and refinement loops.

Physical Simulator In-the-Loop Video Generation (PSIVG) couples Material Point Method (MPM) simulations with Mitsuba rendering to generate physically consistent 3D trajectories, applying Test-Time Texture-Consistency Optimization (TTCO) to penalize pixel-level drift during complex rotations [6]. Similarly, the ProPhy framework introduces a Progressive Physical Alignment approach utilizing a two-stage Mixture-of-Physics-Experts (MoPE) consisting of Semantic Expert and Refinement Expert blocks to extract and refine hierarchical physical priors dynamically [7].

## Long-Video Generation, Memory Models, and Autoregressive Scaling
Extending video generation beyond short clips into coherent long-duration narratives requires sophisticated memory management and hierarchical generation strategies. 

Agentic Auto-Regressive Diffusion (A2RD) models frame long video synthesis as a closed-loop Retrieve-Synthesize-Refine-Update cycle supported by Multimodal Video Memory (MVMem) and Hierarchical Test-Time Self-Improvement (HITS) to prevent semantic drift and narrative collapse [8]. Architectural hybrids like MovieDreamer decompose complex narratives into manageable scenes via script conditioning, coupling global autoregressive script planners with diffusion rendering [15]. Furthermore, VideoSSM integrates hybrid state-space memory models with autoregressive diffusion to harmonize short- and long-term context [9], while HiAR implements hierarchical denoising across matched noise levels to boost inference efficiency and temporal stability [16].

## Optimization Dynamics: Gradient Flows and Self-Forcing
Addressing the fundamental limitations of streaming autoregressive generation requires rethinking gradient propagation and training objectives. 

Connected Self Forcing reconnects backward gradient paths across autoregressive chunks, allowing feedback from later predictions to guide earlier context generation and mitigate exposure bias during long rollouts [10]. Concurrently, SGF+ introduces gradient flow decoupling by assigning separate parameters to context writing and frame denoising, eliminating negative gradient interference during joint optimization while maintaining causal interaction [11].

## Trends and Open Problems
The field of video and multimodal generation is advancing rapidly along several key vectors:
- **Unified Multimodal Architectures:** Moving toward fully native multimodal transformers that ingest text, audio, image, and video streams within a single unified training objective.
- **Physical Plausibility:** Integrating real-time physics simulation engines (e.g., MPM and fluid dynamics) directly into test-time optimization loops to eliminate physical artifacts.
- **Exposure Bias Mitigation:** Overcoming error accumulation in long-horizon autoregressive video generation via advanced self-forcing and decoupled gradient optimization [10][11].
- **Real-Time Interactive Generation:** Scaling real-time latent diffusion [3] to support interactive world models for gaming, robotics, and immersive simulation.

## References
[1] Evolution of Video Generative Foundations. web. https://arxiv.org/html/2604.06339v1 (n.d.)
[2] CV-VAE: A Compatible Video VAE for Latent Generative Video Models. hf-search. https://huggingface.co/papers/2405.20279 (2024-05-30)
[3] LTX-Video: Realtime Video Latent Diffusion. hf-search. https://huggingface.co/papers/2501.00103 (2024-12-30)
[4] Directing the World: Fast Autoregressive Video Generation with Compositional Human-Camera Control. web. https://arxiv.org/html/2606.27964 (2026-06-26)
[5] CT-1: Vision-Language-Camera for Video Generation. web. https://www.emergentmind.com/papers/2604.09201 (2026-04-10)
[6] Physical Simulator In-the-Loop Video Generation (PSIVG). web. https://arxiv.org/html/2603.06408 (n.d.)
[7] ProPhy: Progressive Physical Alignment for Dynamic World Simulation. web. https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_ProPhy_Progressive_Physical_Alignment_for_Dynamic_World_Simulation_CVPR_2026_paper.pdf (2026-n.d.)
[8] Agentic Auto-Regressive Diffusion (A2RD) for Long Video Consistency. web. https://arxiv.org/html/2605.06924 (n.d.)
[9] VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory. hf-search. https://huggingface.co/papers/2512.04519 (2025-12-04)
[10] Connected Self Forcing: Beyond Local Learning in Video Autoregression. arxiv. https://arxiv.org/abs/2610.12156 (2026-10-08)
[11] SGF+: Decoupling Gradient Flows for Autoregressive Video Generation. arxiv. https://arxiv.org/abs/2610.10429 (2026-10-07)
[12] Latte: Latent Diffusion Transformer for Video Generation. web. https://arxiv.org/html/2401.03048v3 (n.d.)
[13] VDT: General-purpose Video Diffusion Transformers via Mask Modeling. hf-search. https://huggingface.co/papers/2305.13311 (2023-05-22)
[14] AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers. hf-search. https://huggingface.co/papers/2411.18673 (2024-11-27)
[15] MovieDreamer: Hierarchical Long-Duration Video Generation. web. https://proceedings.iclr.cc/paper_files/paper/2025/file/7d4647db960780c8b5a4e7d9e4a58d68-Paper-Conference.pdf (2025)
[16] HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising. hf-search. https://huggingface.co/papers/2603.08703 (2026-03-09)
