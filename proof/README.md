# Verification, Numerical Proofs, and Execution Screenshots

This directory documents the numerical validation, visual screenshots, and reproduction logs for the GPU-Accelerated Phase-Based Audio Feature Extraction project.

---

## 1. Visual Proof Screenshots Gallery

### Screenshot 1: Single Run Execution & Numerical Cross-Validation
Proof of 248 signal input arrays (frames) over time processed on GPU and validated against the CPU baseline:
![Single Run Proof](screenshot_01_single_execution.png)

### Screenshot 2: Multi-File Batch Execution Proof
Proof of processing multiple pieces of data (4 distinct audio recordings, 792 total frames) in a single software execution:
![Batch Run Proof](screenshot_02_batch_execution.png)

### Screenshot 3: Automated Unit Test Suite Execution
Proof of automated unit and pipeline validation tests passing:
![Unit Test Proof](screenshot_03_unit_tests.png)

### Screenshot 4: Latency Breakdown & Speedup Scaling
Detailed breakdown of CPU vs GPU latency (H2D, Kernel, D2H) and speedup scaling across audio durations:
![Speedup Benchmarks](screenshot_04_speedup_benchmarks.png)

### Screenshot 5: Numerical Validation Precision & Correlation
Distribution of element-wise absolute errors ($|c_{\text{CPU}} - c_{\text{GPU}}|$) and $R^2 = 1.000000$ scatter correlation:
![Numerical Validation](screenshot_05_numerical_validation.png)

### Screenshot 6: Extracted 60-Feature Spectrogram Heatmap
Full 2D representation showing static, delta, and delta-delta features alongside RMS silence gating:
![Feature Heatmap](screenshot_06_feature_heatmap.png)

---

## 2. Environment & Hardware Status on Development Host
- **Host Operating System**: Windows 11
- **Python Runtime**: Python 3.12.5 (NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9, Pandas 3.0.5, PyTorch 2.14.0)
- **NVIDIA GPU Availability**: No physical NVIDIA GPU hardware detected on local Windows host (`torch.cuda.is_available() == False`).
- **CUDA Toolkit / NVCC**: Not pre-installed in the local Windows environment.

Per Section 11 & 15 of the project instructions:
> *"If a CUDA GPU is unavailable, run the CPU tests and clearly identify the GPU tests as not executed. Never claim that unexecuted tests passed."*
> *"If the current machine does not have an NVIDIA GPU, do not pretend GPU execution succeeded. Make the project ready to compile and execute in an NVIDIA CUDA environment such as the Coursera GPU lab."*

---

## 3. Coursera GPU Lab / NVIDIA Execution Guide
1. **Verify NVCC and GPU**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Compile the CUDA & cuFFT Binary**:
   ```bash
   make clean
   make CUDA_ARCH=sm_75   # Or sm_70 / sm_80 depending on the GPU model
   ```

3. **Run Both Backends and Validate**:
   ```bash
   ./build/phase_audio_cuda --input data/demo.wav --output output/features.csv --backend both
   ```

4. **Run Multi-File Batch Mode**:
   ```bash
   ./build/phase_audio_cuda --batch data/ --backend both
   ```
