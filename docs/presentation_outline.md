# Capstone Presentation Outline: 5-10 Minute Technical Demonstration

## Slide 1: Title & Overview (0:00 - 1:00)
- **Title**: GPU-Accelerated Phase-Based Audio Feature Extraction using CUDA and cuFFT
- **Context**: Coursera GPU Programming Specialization Capstone Project
- **Key Message**: Moving beyond magnitude-only spectrograms (MFCCs) by exploiting phase residuals to characterize speech acoustics at high GPU throughput.

## Slide 2: Motivation & Phase Signal Processing (1:00 - 2:30)
- **The Problem**: Traditional audio representations discard Fourier phase. However, phase residuals reveal fine-grained temporal structure, glottal pulse timing, and non-stationary sound transitions.
- **The Mathematical Formulation**:
  $$\Delta\phi(k,t) = \text{wrap}\left(\phi(k,t) - \phi(k,t-1) - \frac{2\pi k H}{N}\right)$$
- **Orthogonal Cepstral Basis**: Decomposing wrapped phase across 256 frequency bins into 10 cosine and 10 sine basis vectors (20 static features).
- **Temporal Derivatives**: 20 delta and 20 delta-delta features, creating a 60-dimensional representation per frame.

## Slide 3: Parallel CUDA & cuFFT Architecture (2:30 - 4:30)
- **Batched cuFFT Execution**: Transforming all overlapping audio frames simultaneously in GPU memory.
- **Dedicated Custom Kernels**:
  1. `audio_framing_and_rms_kernel`: Parallel frame division with shared memory RMS reduction.
  2. `apply_hann_window_kernel`: Coalesced window multiplication.
  3. `compute_phase_residual_kernel`: GPU-accelerated 4-quadrant arctangent and phase wrapping.
  4. `extract_cepstral_features_kernel`: Warp shuffle (`__shfl_down_sync`) reduction for fast matrix-vector basis projection.
  5. `compute_delta_features_kernel` & `compute_delta_delta_features_kernel`: 2D thread block temporal regression.
  6. `assemble_feature_matrix_kernel`: Output matrix consolidation.

## Slide 4: Live Demonstration & Numerical Correctness (4:30 - 6:30)
- **Execution Modes**: Command-line demonstration showing `--backend gpu`, `--backend cpu`, and `--backend both`.
- **Validation Threshold**:
  - Max Absolute Error: $\le 10^{-3}$
  - Mean Absolute Error: $\le 10^{-4}$
- **Visual Inspection**: Displaying generated feature heatmaps and time-series dynamic plots.

## Slide 5: Performance Benchmarks & Speedup (6:30 - 8:00)
- Comparison of CPU reference vs NVIDIA GPU execution.
- Breakdown of latency: H2D transfer, GPU kernel processing, and D2H transfer.
- Observed speedup scaling on multi-second audio streams.

## Slide 6: Summary & Future Extensions (8:00 - 9:00)
- **Summary**: Delivered a robust, zero-dependency C++17/CUDA application featuring full mathematical equivalence between CPU and GPU.
- **Future Directions**: Multi-GPU streaming via CUDA Streams, integration with real-time microphone input, and downstream deep neural network training.
- **Q&A Session**: (1 minute).
