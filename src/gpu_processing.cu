#include "gpu_processing.h"
#include "feature_extraction.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>

// ============================================================================
// CUDA Device Query & Compatibility Check
// ============================================================================
bool check_cuda_device_available(std::string* device_name,
                                 int* compute_major,
                                 int* compute_minor) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return false;
    }

    if (device_name) *device_name = prop.name;
    if (compute_major) *compute_major = prop.major;
    if (compute_minor) *compute_minor = prop.minor;
    return true;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Kernel 1: Audio Frame Preparation & RMS Energy Calculation
// Divides the raw 1D audio sample array into overlapping frames, zero-pads each
// frame to FFT_SIZE (512), and computes the RMS energy for low-energy gating.
__global__ void audio_framing_and_rms_kernel(const float* __restrict__ d_audio_samples,
                                             float* __restrict__ d_padded_frames,
                                             float* __restrict__ d_rms_energies,
                                             int total_samples,
                                             int num_frames) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x; // 0 .. 511

    if (frame_idx >= num_frames) return;

    int sample_offset = frame_idx * HOP_SIZE;
    float sample_val = 0.0f;

    // Load audio sample or zero-pad if outside FRAME_SIZE
    if (tid < FRAME_SIZE) {
        int idx = sample_offset + tid;
        sample_val = (idx < total_samples) ? d_audio_samples[idx] : 0.0f;
    }
    d_padded_frames[frame_idx * FFT_SIZE + tid] = sample_val;

    // Parallel reduction in shared memory to compute RMS energy over FRAME_SIZE (400)
    __shared__ float s_sq_sum[512];
    s_sq_sum[tid] = (tid < FRAME_SIZE) ? (sample_val * sample_val) : 0.0f;
    __syncthreads();

    for (int s = 256; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < 512) {
            s_sq_sum[tid] += s_sq_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_rms_energies[frame_idx] = sqrtf(s_sq_sum[0] / static_cast<float>(FRAME_SIZE));
    }
}

// Kernel 2: Hann Window Application
// Multiplies the first FRAME_SIZE (400) samples of each frame by the Hann window.
__global__ void apply_hann_window_kernel(float* __restrict__ d_padded_frames,
                                         const float* __restrict__ d_hann_window,
                                         int num_frames) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x; // 0 .. 511

    if (frame_idx >= num_frames) return;

    if (tid < FRAME_SIZE) {
        int idx = frame_idx * FFT_SIZE + tid;
        d_padded_frames[idx] *= d_hann_window[tid];
    }
}

// Kernel 3: Phase Residual Calculation with Phase Wrapping and RMS Gating
// Δφ(k,t) = wrap(φ(k,t) - φ(k,t-1) - 2πkH/N)
__global__ void compute_phase_residual_kernel(const cufftComplex* __restrict__ d_fft_complex,
                                              const float* __restrict__ d_rms_energies,
                                              const float* __restrict__ d_expected_advances,
                                              float* __restrict__ d_phase_residuals,
                                              int num_frames,
                                              float rms_gate) {
    int frame_idx = blockIdx.x;
    int k = threadIdx.x; // 0 .. 255 (NUM_FREQ_BINS)

    if (frame_idx >= num_frames || k >= NUM_FREQ_BINS) return;

    // FFT R2C output has pitch of (FFT_SIZE / 2 + 1) = 257 complex bins per frame
    int complex_pitch = FFT_SIZE / 2 + 1;
    cufftComplex curr_c = d_fft_complex[frame_idx * complex_pitch + k];
    float curr_phase = atan2f(curr_c.y, curr_c.x);

    float frame_rms = d_rms_energies[frame_idx];
    bool is_silent = (frame_rms < rms_gate);
    bool prev_is_silent = (frame_idx == 0) || (d_rms_energies[frame_idx - 1] < rms_gate);

    float residual = 0.0f;
    if (frame_idx > 0 && !is_silent && !prev_is_silent) {
        cufftComplex prev_c = d_fft_complex[(frame_idx - 1) * complex_pitch + k];
        float prev_phase = atan2f(prev_c.y, prev_c.x);
        float raw_diff = curr_phase - prev_phase - d_expected_advances[k];
        residual = wrap_phase(raw_diff);
    }

    d_phase_residuals[frame_idx * NUM_FREQ_BINS + k] = residual;
}

// Kernel 4: Phase Cepstral Feature Extraction
// Calculates 10 Cosine and 10 Sine cepstral coefficients per frame (20 static features total)
// c_n = (1 / K) * sum_{k=0}^{K-1} phase_residual(k) * basis(n, k)
__global__ void extract_cepstral_features_kernel(const float* __restrict__ d_phase_residuals,
                                                 const float* __restrict__ d_cepstral_basis,
                                                 float* __restrict__ d_static_features,
                                                 int num_frames) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x; // 0 .. 255 (NUM_FREQ_BINS)

    if (frame_idx >= num_frames) return;

    __shared__ float s_residual[NUM_FREQ_BINS];
    s_residual[tid] = d_phase_residuals[frame_idx * NUM_FREQ_BINS + tid];
    __syncthreads();

    // Loop through each of the 20 static features
    for (int feat = 0; feat < NUM_STATIC_FEATURES; ++feat) {
        float term = s_residual[tid] * d_cepstral_basis[feat * NUM_FREQ_BINS + tid];

        // Warp shuffle reduction
        unsigned int mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset >>= 1) {
            term += __shfl_down_sync(mask, term, offset);
        }

        // Store each warp's sum in shared memory
        __shared__ float s_warp_sums[8];
        int lane = tid & 31;
        int warp_id = tid >> 5;
        if (lane == 0) {
            s_warp_sums[warp_id] = term;
        }
        __syncthreads();

        // Final reduction across the 8 warps in the first warp
        if (warp_id == 0) {
            float block_sum = (tid < 8) ? s_warp_sums[tid] : 0.0f;
            for (int offset = 4; offset > 0; offset >>= 1) {
                block_sum += __shfl_down_sync(mask, block_sum, offset);
            }
            if (tid == 0) {
                d_static_features[frame_idx * NUM_STATIC_FEATURES + feat] =
                    block_sum / static_cast<float>(NUM_FREQ_BINS);
            }
        }
        __syncthreads();
    }
}

// Kernel 5: Delta Feature Calculation
// Computes 20 first-order temporal derivatives using 5-point linear regression
__global__ void compute_delta_features_kernel(const float* __restrict__ d_static_features,
                                              float* __restrict__ d_delta_features,
                                              int num_frames) {
    int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feat_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (frame_idx < num_frames && feat_idx < NUM_STATIC_FEATURES) {
        d_delta_features[frame_idx * NUM_DELTA_FEATURES + feat_idx] =
            compute_delta_sample(d_static_features, frame_idx, feat_idx, num_frames, NUM_STATIC_FEATURES);
    }
}

// Kernel 6: Delta-Delta Feature Calculation
// Computes 20 second-order temporal derivatives by applying delta operator to delta features
__global__ void compute_delta_delta_features_kernel(const float* __restrict__ d_delta_features,
                                                    float* __restrict__ d_ddelta_features,
                                                    int num_frames) {
    int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feat_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (frame_idx < num_frames && feat_idx < NUM_DELTA_FEATURES) {
        d_ddelta_features[frame_idx * NUM_DDELTA_FEATURES + feat_idx] =
            compute_delta_sample(d_delta_features, frame_idx, feat_idx, num_frames, NUM_DELTA_FEATURES);
    }
}

// Kernel 7: Feature Matrix Assembly
// Packs static (20), delta (20), and delta-delta (20) features into contiguous 60-element frame vectors
__global__ void assemble_feature_matrix_kernel(const float* __restrict__ d_static,
                                               const float* __restrict__ d_delta,
                                               const float* __restrict__ d_ddelta,
                                               float* __restrict__ d_final_features,
                                               int num_frames) {
    int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feat_col = blockIdx.y * blockDim.y + threadIdx.y;

    if (frame_idx < num_frames && feat_col < TOTAL_FEATURES) {
        float val = 0.0f;
        if (feat_col < NUM_STATIC_FEATURES) {
            val = d_static[frame_idx * NUM_STATIC_FEATURES + feat_col];
        } else if (feat_col < NUM_STATIC_FEATURES + NUM_DELTA_FEATURES) {
            int sub_idx = feat_col - NUM_STATIC_FEATURES;
            val = d_delta[frame_idx * NUM_DELTA_FEATURES + sub_idx];
        } else {
            int sub_idx = feat_col - (NUM_STATIC_FEATURES + NUM_DELTA_FEATURES);
            val = d_ddelta[frame_idx * NUM_DDELTA_FEATURES + sub_idx];
        }
        d_final_features[frame_idx * TOTAL_FEATURES + feat_col] = val;
    }
}

// ============================================================================
// High-Level GPU Execution Pipeline
// ============================================================================
FeatureMatrix process_audio_gpu(const AudioData& audio,
                                float rms_gate,
                                ExecutionTimings* timings) {
    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    if (dev_err != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No CUDA-capable GPU detected on this system.");
    }

    const int num_frames = audio.num_frames;
    const int total_samples = audio.total_samples;
    FeatureMatrix features;
    features.allocate(num_frames);

    if (num_frames <= 0) return features;

    // Precomputed coefficients on Host
    std::vector<float> h_hann = generate_hann_window(FRAME_SIZE);
    std::vector<float> h_expected_advances = generate_expected_phase_advances(NUM_FREQ_BINS, HOP_SIZE, FFT_SIZE);
    std::vector<float> h_cepstral_basis = generate_cepstral_basis(NUM_STATIC_FEATURES, NUM_FREQ_BINS);

    // CUDA Events for precise timing
    cudaEvent_t ev_start_h2d, ev_end_h2d;
    cudaEvent_t ev_start_kernel, ev_end_kernel;
    cudaEvent_t ev_start_d2h, ev_end_d2h;

    CUDA_CHECK(cudaEventCreate(&ev_start_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_end_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_start_kernel));
    CUDA_CHECK(cudaEventCreate(&ev_end_kernel));
    CUDA_CHECK(cudaEventCreate(&ev_start_d2h));
    CUDA_CHECK(cudaEventCreate(&ev_end_d2h));

    // Device memory pointers
    float* d_audio_samples = nullptr;
    float* d_hann_window = nullptr;
    float* d_expected_advances = nullptr;
    float* d_cepstral_basis = nullptr;
    float* d_padded_frames = nullptr;
    float* d_rms_energies = nullptr;
    cufftComplex* d_fft_complex = nullptr;
    float* d_phase_residuals = nullptr;
    float* d_static_features = nullptr;
    float* d_delta_features = nullptr;
    float* d_ddelta_features = nullptr;
    float* d_final_features = nullptr;
    cufftHandle cufft_plan;

    try {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_audio_samples, total_samples * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hann_window, FRAME_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_expected_advances, NUM_FREQ_BINS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cepstral_basis, NUM_STATIC_FEATURES * NUM_FREQ_BINS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_padded_frames, num_frames * FFT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rms_energies, num_frames * sizeof(float)));

        // FFT complex output size: num_frames * (FFT_SIZE / 2 + 1)
        int complex_pitch = FFT_SIZE / 2 + 1;
        CUDA_CHECK(cudaMalloc(&d_fft_complex, num_frames * complex_pitch * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_phase_residuals, num_frames * NUM_FREQ_BINS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_static_features, num_frames * NUM_STATIC_FEATURES * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_delta_features, num_frames * NUM_DELTA_FEATURES * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ddelta_features, num_frames * NUM_DDELTA_FEATURES * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_final_features, num_frames * TOTAL_FEATURES * sizeof(float)));

        // Create cuFFT Batched 1D Real-to-Complex Plan
        int rank = 1;
        int n[1] = { FFT_SIZE };
        int inembed[1] = { FFT_SIZE };
        int onembed[1] = { complex_pitch };
        int istride = 1, idist = FFT_SIZE;
        int ostride = 1, odist = complex_pitch;

        CUFFT_CHECK(cufftPlanMany(&cufft_plan, rank, n,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  CUFFT_R2C, num_frames));

        // --------------------------------------------------------------------
        // Stage 1: Host to Device Memory Transfers
        // --------------------------------------------------------------------
        CUDA_CHECK(cudaEventRecord(ev_start_h2d));
        CUDA_CHECK(cudaMemcpy(d_audio_samples, audio.samples.data(), total_samples * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_hann_window, h_hann.data(), FRAME_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_expected_advances, h_expected_advances.data(), NUM_FREQ_BINS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cepstral_basis, h_cepstral_basis.data(), NUM_STATIC_FEATURES * NUM_FREQ_BINS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev_end_h2d));

        // --------------------------------------------------------------------
        // Stage 2: GPU Kernel Execution Pipeline
        // --------------------------------------------------------------------
        CUDA_CHECK(cudaEventRecord(ev_start_kernel));

        // Kernel 1: Audio Framing and RMS Energy Calculation
        audio_framing_and_rms_kernel<<<num_frames, 512>>>(d_audio_samples,
                                                          d_padded_frames,
                                                          d_rms_energies,
                                                          total_samples,
                                                          num_frames);
        CUDA_CHECK(cudaGetLastError());

        // Kernel 2: Hann Window Application
        apply_hann_window_kernel<<<num_frames, 512>>>(d_padded_frames,
                                                      d_hann_window,
                                                      num_frames);
        CUDA_CHECK(cudaGetLastError());

        // Batched cuFFT Execution: transforms all num_frames in parallel
        CUFFT_CHECK(cufftExecR2C(cufft_plan, d_padded_frames, d_fft_complex));

        // Kernel 3: Phase Residual Calculation
        compute_phase_residual_kernel<<<num_frames, NUM_FREQ_BINS>>>(d_fft_complex,
                                                                     d_rms_energies,
                                                                     d_expected_advances,
                                                                     d_phase_residuals,
                                                                     num_frames,
                                                                     rms_gate);
        CUDA_CHECK(cudaGetLastError());

        // Kernel 4: Phase Cepstral Feature Extraction
        extract_cepstral_features_kernel<<<num_frames, NUM_FREQ_BINS>>>(d_phase_residuals,
                                                                        d_cepstral_basis,
                                                                        d_static_features,
                                                                        num_frames);
        CUDA_CHECK(cudaGetLastError());

        // Kernel 5: Delta Feature Calculation
        dim3 block_2d(16, 16);
        dim3 grid_delta((num_frames + 15) / 16, (NUM_STATIC_FEATURES + 15) / 16);
        compute_delta_features_kernel<<<grid_delta, block_2d>>>(d_static_features,
                                                                d_delta_features,
                                                                num_frames);
        CUDA_CHECK(cudaGetLastError());

        // Kernel 6: Delta-Delta Feature Calculation
        dim3 grid_ddelta((num_frames + 15) / 16, (NUM_DELTA_FEATURES + 15) / 16);
        compute_delta_delta_features_kernel<<<grid_ddelta, block_2d>>>(d_delta_features,
                                                                       d_ddelta_features,
                                                                       num_frames);
        CUDA_CHECK(cudaGetLastError());

        // Kernel 7: Feature Matrix Assembly
        dim3 grid_assemble((num_frames + 15) / 16, (TOTAL_FEATURES + 15) / 16);
        assemble_feature_matrix_kernel<<<grid_assemble, block_2d>>>(d_static_features,
                                                                    d_delta_features,
                                                                    d_ddelta_features,
                                                                    d_final_features,
                                                                    num_frames);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev_end_kernel));

        // --------------------------------------------------------------------
        // Stage 3: Device to Host Memory Transfers
        // --------------------------------------------------------------------
        CUDA_CHECK(cudaEventRecord(ev_start_d2h));
        CUDA_CHECK(cudaMemcpy(features.data.data(), d_final_features, num_frames * TOTAL_FEATURES * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(features.rms_energies.data(), d_rms_energies, num_frames * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev_end_d2h));

        // Synchronize device events
        CUDA_CHECK(cudaEventSynchronize(ev_end_d2h));

        // Record timing metrics
        if (timings) {
            float ms_h2d = 0.0f, ms_kernel = 0.0f, ms_d2h = 0.0f;
            cudaEventElapsedTime(&ms_h2d, ev_start_h2d, ev_end_h2d);
            cudaEventElapsedTime(&ms_kernel, ev_start_kernel, ev_end_kernel);
            cudaEventElapsedTime(&ms_d2h, ev_start_d2h, ev_end_d2h);

            timings->gpu_h2d_ms = static_cast<double>(ms_h2d);
            timings->gpu_kernel_ms = static_cast<double>(ms_kernel);
            timings->gpu_d2h_ms = static_cast<double>(ms_d2h);
            timings->gpu_total_ms = static_cast<double>(ms_h2d + ms_kernel + ms_d2h);
        }

    } catch (...) {
        // Cleanup resources before re-throwing
        if (d_audio_samples) cudaFree(d_audio_samples);
        if (d_hann_window) cudaFree(d_hann_window);
        if (d_expected_advances) cudaFree(d_expected_advances);
        if (d_cepstral_basis) cudaFree(d_cepstral_basis);
        if (d_padded_frames) cudaFree(d_padded_frames);
        if (d_rms_energies) cudaFree(d_rms_energies);
        if (d_fft_complex) cudaFree(d_fft_complex);
        if (d_phase_residuals) cudaFree(d_phase_residuals);
        if (d_static_features) cudaFree(d_static_features);
        if (d_delta_features) cudaFree(d_delta_features);
        if (d_ddelta_features) cudaFree(d_ddelta_features);
        if (d_final_features) cudaFree(d_final_features);
        cufftDestroy(cufft_plan);
        cudaEventDestroy(ev_start_h2d);
        cudaEventDestroy(ev_end_h2d);
        cudaEventDestroy(ev_start_kernel);
        cudaEventDestroy(ev_end_kernel);
        cudaEventDestroy(ev_start_d2h);
        cudaEventDestroy(ev_end_d2h);
        throw;
    }

    // Free device memory
    cudaFree(d_audio_samples);
    cudaFree(d_hann_window);
    cudaFree(d_expected_advances);
    cudaFree(d_cepstral_basis);
    cudaFree(d_padded_frames);
    cudaFree(d_rms_energies);
    cudaFree(d_fft_complex);
    cudaFree(d_phase_residuals);
    cudaFree(d_static_features);
    cudaFree(d_delta_features);
    cudaFree(d_ddelta_features);
    cudaFree(d_final_features);
    cufftDestroy(cufft_plan);
    cudaEventDestroy(ev_start_h2d);
    cudaEventDestroy(ev_end_h2d);
    cudaEventDestroy(ev_start_kernel);
    cudaEventDestroy(ev_end_kernel);
    cudaEventDestroy(ev_start_d2h);
    cudaEventDestroy(ev_end_d2h);

    return features;
}
