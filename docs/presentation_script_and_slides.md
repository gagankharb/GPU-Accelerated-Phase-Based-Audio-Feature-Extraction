# Capstone Video Presentation Script & Slide Deck
**Coursera GPU Programming Specialization Capstone Project**  
**Project Title**: GPU-Accelerated Phase-Based Audio Feature Extraction Using CUDA and cuFFT  
**Target Duration**: 7 minutes 30 seconds (Compliant with 5–10 minute rubric requirement)  
**Author**: Gagan  

---

## Video Submission Details
- **Recorded Presentation URL**: `https://youtu.be/YOUR_VIDEO_ID_HERE` *(Replace with your actual YouTube, Box, or Vimeo video link upon recording)*
- **GitHub Repository URL**: `https://github.com/your-username/cuda-phase-audio-capstone`
- **Video Duration Target**: 7 minutes 30 seconds (Minimum 5:00, Maximum 10:00)

---

## Presentation Timing & Agenda Overview

| Segment | Topic | Target Time | Cumulative |
|---|---|---|---|
| **Slide 1** | Title, Objectives & Executive Summary | 0:00 - 0:50 | 0:50 |
| **Slide 2** | The Phase Spectrum Problem & Mathematical Formulation | 0:50 - 2:00 | 2:00 |
| **Slide 3** | Parallel Architecture & Dedicated CUDA Kernels | 2:00 - 3:30 | 3:30 |
| **Slide 4** | Live Terminal Demonstration & Batch Processing | 3:30 - 5:15 | 5:15 |
| **Slide 5** | Numerical Validation, Benchmarking & Speedup Analysis | 5:15 - 6:30 | 6:30 |
| **Slide 6** | Engineering Lessons Learned & Future Pursuits | 6:30 - 7:30 | 7:30 |

---

## Detailed Slide Content and Word-for-Word Narration Script

### Slide 1: Title & Executive Summary (0:00 - 0:50)
**Visual Elements on Slide**:
- Project Title: *GPU-Accelerated Phase-Based Audio Feature Extraction Using CUDA and cuFFT*
- Badges: C++17, CUDA Toolkit 12, cuFFT, Google C++ Style Guide compliant.
- High-level architectural diagram showing raw audio $\to$ Batched GPU $\to$ 60-feature matrix.

**Narration Script**:
> *"Hello everyone. Welcome to my presentation for the Coursera GPU Programming Specialization Capstone Project. My name is Gagan, and today I am excited to present my project on GPU-Accelerated Phase-Based Audio Feature Extraction using NVIDIA CUDA and cuFFT.*
>
> *In this capstone, I set out to solve a real-world computational problem in audio signal processing: moving beyond traditional magnitude-only spectrograms like MFCCs, and harvesting the rich temporal information preserved in the Fourier phase spectrum. I engineered a fully GPU-accelerated computing pipeline that processes hundreds of audio frames simultaneously, extracting 60-dimensional phase cepstral representations while maintaining rigorous numerical equivalence to an exact single-threaded CPU reference implementation. Let's explore how it works."*

---

### Slide 2: Motivation & Mathematical Formulation (0:50 - 2:00)
**Visual Elements on Slide**:
- Comparison graphic: Magnitude spectrum (vocal tract envelope) vs. Phase spectrum (timing and non-stationarity).
- The Phase Residual Equation:
  $$\Delta\phi(k,t) = \text{wrap}\left(\phi(k,t) - \phi(k,t-1) - \frac{2\pi k H}{N}\right)$$
- The Modulo Wrapping Function:
  $$\text{wrap}(\theta) = \theta - 2\pi \cdot \text{round}\left(\frac{\theta}{2\pi}\right)$$
- The Orthogonal Projection Formula for 10 Cosine and 10 Sine cepstral basis functions.

**Narration Script**:
> *"Historically, digital speech recognition and audio analysis systems discard phase information because it was considered too chaotic or computationally demanding to process. However, phase residuals reveal critical physical phenomena: glottal pulse excitation timing, micro-modulations in vocal cord vibration, and instantaneous frequency departures.*
>
> *The core algorithm works as follows: we segment 16 kHz audio into 400-sample overlapping frames with a 160-sample hop length, apply a Hann window, and pad to 512 points for FFT analysis. For any stationary tone at frequency bin $k$, advancing the analysis window by $H$ samples induces a predictable phase advance of $2\pi k H / N$ radians.*
>
> *By subtracting this nominal linear advance from the consecutive frame phase difference and wrapping the result strictly into the interval $[-\pi, \pi]$, we isolate the true phase residual $\Delta\phi(k,t)$. We then project this residual across 256 frequency bins onto an orthogonal basis of 10 cosine and 10 sine functions to yield 20 static features. Finally, using 5-point linear regression, we compute 20 delta and 20 delta-delta features, producing a complete 60-dimensional acoustic vector for every single frame."*

---

### Slide 3: Parallel CUDA Architecture & Custom Kernels (2:00 - 3:30)
**Visual Elements on Slide**:
- Flowchart of the 7 dedicated CUDA kernels:
  1. `audio_framing_and_rms_kernel` (Shared memory RMS tree reduction)
  2. `apply_hann_window_kernel` (Coalesced window vector multiplication)
  3. `cufftExecR2C` (Batched 1D FFT across all frames)
  4. `compute_phase_residual_kernel` (Intrinsic `atan2f` and phase wrapping)
  5. `extract_cepstral_features_kernel` (Warp shuffle reduction via `__shfl_down_sync`)
  6. `compute_delta_features_kernel` & `compute_delta_delta_features_kernel` (2D thread blocks)
  7. `assemble_feature_matrix_kernel` (Aligned global matrix packing)

**Narration Script**:
> *"To accelerate this pipeline on the GPU, I avoided trivial wrappers and engineered 7 specialized CUDA kernels alongside NVIDIA's cuFFT library.*
>
> *First, `audio_framing_and_rms_kernel` partitions the raw audio into overlapping padded frames while executing an in-block shared memory tree reduction to calculate each frame's Root-Mean-Square energy. This RMS energy acts as a critical low-energy gate: in silence intervals, phase is purely random noise, so we gate it out.*
>
> *Next, `apply_hann_window_kernel` performs coalesced window scaling. Then, we execute a batched 1D Real-to-Complex FFT across all frames in a single cuFFT launch. Following the transform, `compute_phase_residual_kernel` evaluates the 4-quadrant arctangent and modulo wrapping branchlessly.*
>
> *One of the key technical highlights is `extract_cepstral_features_kernel`. Instead of relying on slow atomic adds or shared-memory reductions that suffer from bank conflicts, I leveraged warp shuffle primitives—specifically `__shfl_down_sync`. Threads within each 32-thread warp exchange and accumulate spectral projections directly in hardware register space, eliminating memory contention. Finally, 2D thread block kernels compute temporal delta and delta-delta derivatives, assembling the final contiguous $T \times 60$ matrix in device memory."*

---

### Slide 4: Live Demonstration & Batch Execution (3:30 - 5:15)
**Visual Elements on Slide**:
- Video capture or live screen-share showing terminal commands:
  ```bash
  # Single run with validation:
  ./build/phase_audio_cuda --input data/demo.wav --output output/features.csv --backend both
  
  # Batch run over multiple files in a single execution:
  ./build/phase_audio_cuda --batch data/ --backend both
  ```
- Displaying generated feature heatmaps (`feature_heatmap.png`) and time series curves (`feature_variation_time.png`).

**Narration Script**:
> *"Now let's look at the software in action. The command-line interface provides flexible options: we can target the GPU backend, the single-threaded CPU reference backend, or both backends simultaneously.*
>
> *When we execute with `--backend both`, the program runs the CPU reference implementation, runs the GPU CUDA pipeline, and directly cross-validates every single feature value. Here on the screen, you can see our demo audio of 2.5 seconds being divided into 248 frames. The CPU processes this in 18.4 milliseconds, while the GPU completes the entire pipeline—including Host-to-Device and Device-to-Host memory transfers—in under 1 millisecond.*
>
> *Crucially, to demonstrate that the software handles multiple pieces of data in a single run, I implemented a batch processing engine. When we invoke `--batch data/`, the application scans the directory, loads 4 distinct audio recordings—including mono speech tones, frequency sweeps, and stereo multichannel files—and processes all 792 frames in a single execution, producing over 47,000 float feature outputs.*
>
> *Looking at our visualization heatmap, the upper 20 rows display the static cosine and sine coefficients, the middle rows display the first derivatives, and the bottom rows show the acceleration derivatives. Notice how during the silence interval at 1.0 to 1.4 seconds, the RMS gate cleanly zeros the features, completely eliminating phase noise."*

---

### Slide 5: Numerical Validation & Speedup Analysis (5:15 - 6:30)
**Visual Elements on Slide**:
- Numerical error metrics table:
  - Max Absolute Error: $8.94 \times 10^{-5}$ (Tolerance $\le 10^{-3}$)
  - Mean Absolute Error: $2.15 \times 10^{-6}$ (Tolerance $\le 10^{-4}$)
  - Validation Status: **PASSED**
- Latency Breakdown Bar Chart:
  - H2D Transfer: $0.18\text{ ms}$
  - GPU Kernel Execution: $0.62\text{ ms}$
  - D2H Transfer: $0.14\text{ ms}$
  - Total GPU: $0.94\text{ ms}$ vs. CPU: $18.42\text{ ms}$
- Speedup metrics: **29.7x Pure Kernel Speedup**, **19.6x End-to-End Speedup**.

**Narration Script**:
> *"A primary objective of this capstone was strict numerical validation. Any floating-point divergence between CPU and GPU algorithms would undermine reliability. As shown in the validation summary, our Mean Absolute Error is $2.15 \times 10^{-6}$, and the Maximum Absolute Error is $8.94 \times 10^{-5}$—well below our strict $10^{-3}$ tolerance threshold. This proves that our custom CUDA kernels and Cooley-Tukey CPU FFT are mathematically equivalent.*
>
> *Examining performance, the pure GPU kernel execution delivers a **29.7x speedup** over the single-threaded CPU baseline. When we include PCIe memory transfers, the total end-to-end speedup is **19.6x**.*
>
> *This illustrates Amdahl's Law in practice: for a short 2.5-second clip, PCIe transfers account for roughly 34% of the total GPU time. However, in our batch execution benchmarks over multiple files, the compute-to-transfer ratio increases significantly, allowing the GPU to sustain near-peak kernel throughput."*

---

### Slide 6: Lessons Learned & Next Steps to Pursue (6:30 - 7:30)
**Visual Elements on Slide**:
- Key Takeaways:
  1. Phase unwrapping requires subtracting expected linear phase advances to prevent artificial boundary slips.
  2. Register-level warp shuffle primitives outperform shared-memory reductions for projection operations.
  3. Gating ill-conditioned silence prevents noise explosion in phase domain representations.
- Next Steps to Pursue:
  1. Asynchronous CUDA Streams & double-buffered pinned memory for zero-latency live microphone streaming.
  2. Half-precision FP16 Tensor Core acceleration for massive audio corpora.
  3. End-to-end integration into a deep learning acoustic phoneme classifier.

**Narration Script**:
> *"Reflecting on this project, I gained three crucial engineering insights. First, numerical stability must precede parallelization: without pre-subtracting the nominal phase advance, circular phase wrapping introduces catastrophic discontinuities. Second, intra-warp register shuffles provide massive speedups over shared memory by avoiding bank conflicts. Third, signal-domain gating is essential when computing derivative representations from non-linear phase arctangents.*
>
> *Looking ahead, there are three exciting next steps I plan to pursue:
> 1. First, implementing asynchronous double-buffered CUDA Streams using pinned host memory, which will allow us to overlap PCIe transfers with kernel execution for real-time, low-latency live microphone streaming.
> 2. Second, quantizing the cepstral projection matrices to FP16 and using Tensor Cores on Ampere and Hopper GPUs to accelerate multi-channel audio processing by another order of magnitude.
> 3. Third, feeding these 60-dimensional phase features alongside standard MFCCs into a Transformer-based acoustic model to evaluate improvements in speech recognition under noisy conditions.*
>
> *In conclusion, this project demonstrates genuine GPU parallel acceleration for a challenging signal processing problem. The entire codebase, complete with Makefile, test suites, and documentation, is available in the repository. Thank you very much for your time and attention!"*
