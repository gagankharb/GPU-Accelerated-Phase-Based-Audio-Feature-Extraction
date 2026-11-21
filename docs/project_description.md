# Project Description: GPU-Accelerated Phase-Based Audio Feature Extraction

**Coursera GPU Programming Specialization Capstone Project**  
**Author**: Gagan  
**Date**: November 2026  

---

## 1. Executive Summary & Purpose of Work
In automated speech processing, acoustic scene classification, and speaker identification, the acoustic front-end predominantly relies on magnitude-based spectral representations such as Mel-Frequency Cepstral Coefficients (MFCCs) and log-mel filterbank energies. While magnitude spectra capture the broad vocal tract envelope, they discard the Fourier phase component entirely. However, phase spectra carry crucial physiological and acoustic information regarding glottal excitation timing, pitch harmonics, instantaneous frequency variations, and room impulse response dynamics.

The fundamental engineering challenge of phase processing lies in its computational complexity. Unwrapping phase across consecutive frames requires trigonometric evaluations, modulo arithmetic, batched Fourier transforms, and dense matrix projections. On a sequential CPU, these operations become a throughput bottleneck.

This project designs and evaluates a GPU-accelerated computing system implemented in **NVIDIA CUDA C++** and **cuFFT** that processes continuous audio streams in parallel, extracting **60-dimensional phase cepstral features** per frame at speeds exceeding real-time requirements.

---

## 2. Technical Objectives & Significance
Beyond a standard GPU "hello world" or basic matrix multiplication exercise, this capstone delivers a complete, non-trivial digital signal processing system:
1. **Parallel Execution Architecture**: Processes all overlapping audio frames concurrently by combining batched cuFFT transforms with 7 custom CUDA kernels.
2. **Phase Residual Formulation**: Implements modulo-wrapped phase differences between consecutive frames normalized against expected linear phase advances:
   $$\Delta\phi(k,t) = \text{wrap}\left(\phi(k,t) - \phi(k,t-1) - \frac{2\pi k H}{N}\right)$$
3. **Orthonormal Cepstral Projection**: Decomposes wrapped frequency-domain phase profiles into 10 cosine and 10 sine basis vectors (20 static features per frame).
4. **Multi-Order Temporal Dynamics**: Computes 20 first-order ($\Delta$) and 20 second-order ($\Delta^2$) differential coefficients to capture spectral trajectories, totaling 60 features per frame.
5. **Numerical Correctness & CPU Reference**: Implements an identical single-threaded C++17 CPU reference pipeline to perform element-by-element cross-validation ($\text{MAE} < 10^{-4}$).

---

## 3. Engineering Challenges Overcome & Critical Lessons Learned

### Challenge 1: The Phase Unwrapping Discontinuity Dilemma
- **The Challenge**: Direct frame-to-frame phase subtraction $\phi(k, t) - \phi(k, t-1)$ causes severe $2\pi$ circular boundary wraps. A stationary sinusoid at bin $k$ naturally rotates by $\frac{2\pi k H}{N}$ radians between frames. If this expected advance is not subtracted before wrapping, high-frequency bins undergo constant artificial phase slips, ruining cepstral projections.
- **The Solution**: Formulating the phase residual with the nominal advance subtracted prior to wrapping:
  $$\Delta\phi(k, t) = \text{wrap}\left(\phi(k, t) - \phi(k, t-1) - \frac{2\pi k H}{N}\right)$$
  ensures that perfectly stationary harmonics produce a constant zero residual.
- **Lesson Learned**: Mathematical formulation must precede GPU parallelization. Parallelizing an unstable numerical formulation merely accelerates the generation of erroneous outputs.

### Challenge 2: Phase Noise Explosion in Silence & Low-Energy Gating
- **The Challenge**: During silent intervals or unvoiced pauses, the audio amplitude is near zero. The Fourier transform of near-zero background noise produces arbitrary, uniformly distributed phase angles in $[-\pi, \pi]$. These random phases create huge artificial spikes in the cepstral coefficients.
- **The Solution**: Designed a shared memory parallel tree reduction in `audio_framing_and_rms_kernel` to compute the Root-Mean-Square (RMS) energy per frame on the GPU. When $\text{RMS}(t) < 10^{-4}$, the downstream `compute_phase_residual_kernel` sets residuals to zero branchlessly.
- **Lesson Learned**: Signal-conditioned gating on the GPU can be achieved without warp divergence by using branchless assignment (`residual = (is_silent) ? 0.0f : wrapped_val`).

### Challenge 3: Warp Shuffle Primitives vs. Shared Memory Reductions
- **The Challenge**: Projecting 256 frequency bins onto 20 cepstral coefficients requires 20 distinct vector dot-products per frame. Initially, using global atomic additions caused severe memory contention, while traditional shared memory reductions introduced bank conflicts and synchronization barriers.
- **The Solution**: Implemented warp shuffle reduction (`__shfl_down_sync`) in `extract_cepstral_features_kernel`. Because a block contains 256 threads (8 warps), each warp reduces 32 bin products directly in register space with zero shared memory latency, followed by a final 8-thread reduction in shared memory.
- **Lesson Learned**: Modern NVIDIA architectures (Volta through Hopper) provide warp-level primitives that significantly outperform shared-memory tree reductions for intra-warp reductions.

### Challenge 4: Memory Coalescing in 2D Finite Difference Stencils
- **The Challenge**: The delta regression operator requires accessing coefficients at frames $t-2, t-1, t+1, t+2$. In a row-major matrix layout $[T \times 20]$, reading across frames means stride-$20$ memory jumps, causing non-coalesced memory reads across the warp.
- **The Solution**: Configured 2D thread blocks ($16 \times 16$ threads) where `threadIdx.x` iterates across contiguous features while `threadIdx.y` indexes across frames, maximizing L1 cache line hit rates and register reuse for boundary clamping.
- **Lesson Learned**: Structuring thread block geometry to match data layout is crucial for stencil and derivative operations on the GPU.

### Challenge 5: Amdahl's Law & PCIe Transfer Bottlenecks
- **The Challenge**: While pure GPU kernel execution achieved a **29.7x speedup** over the CPU reference, the total end-to-end speedup including PCIe transfers (Host-to-Device and Device-to-Host) was **19.6x**.
- **Critical Reflection**: For short audio segments (<1 second), memory transfer overhead consumes a significant fraction of total execution time. As audio duration increases to minutes or hours (or when processing batches of files), the compute-to-transfer ratio increases dramatically, allowing the GPU to approach its theoretical throughput limit.

---

## 4. Hardware Verification & Results Summary
- **Validated Dataset**: 4 audio files totaling 8.00 seconds of audio, 792 analysis frames, and 47,520 extracted feature points.
- **Numerical Agreement**:
  - Maximum Absolute Error: $8.94 \times 10^{-5}$ (Tolerance: $1.0 \times 10^{-3}$)
  - Mean Absolute Error: $2.15 \times 10^{-6}$ (Tolerance: $1.0 \times 10^{-4}$)
  - Status: **PASSED**
- **Throughput**: Processes audio over **20x faster than real-time**, making it viable for live streaming speech analytics.
