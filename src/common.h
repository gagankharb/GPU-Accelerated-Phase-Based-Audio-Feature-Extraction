// Copyright 2026 Coursera GPU Specialization Capstone
// Author: Gagan
// Project: GPU-Accelerated Phase-Based Audio Feature Extraction

#ifndef SRC_COMMON_H_
#define SRC_COMMON_H_

#include <cmath>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace audio_phase {

// Audio signal processing pipeline constants (Google C++ Style Guide naming)
constexpr int32_t kSampleRate         = 16000;  // 16 kHz audio sampling rate
constexpr int32_t kFrameSize          = 400;    // 25 ms analysis window length
constexpr int32_t kHopSize            = 160;    // 10 ms frame advance (hop length)
constexpr int32_t kFftSize            = 512;    // cuFFT batched real-to-complex FFT size
constexpr int32_t kNumFreqBins        = 256;    // Analyzed frequency bins (k = 0 .. 255)
constexpr int32_t kNumCosCoeffs       = 10;     // Cosine phase cepstral coefficients
constexpr int32_t kNumSinCoeffs       = 10;     // Sine phase cepstral coefficients
constexpr int32_t kNumStaticFeatures  = 20;     // 10 Cosine + 10 Sine static features
constexpr int32_t kNumDeltaFeatures   = 20;     // First-order time derivatives
constexpr int32_t kNumDDeltaFeatures  = 20;     // Second-order time derivatives
constexpr int32_t kTotalFeatures      = 60;     // Final features per audio frame
constexpr int32_t kDeltaWindow        = 2;      // Window radius for delta regression (D = 2)
constexpr float   kDefaultRmsGate     = 1e-4f;  // Silence / low-energy threshold
constexpr float   kPi                 = 3.14159265358979323846f;
constexpr float   kTwoPi              = 6.28318530717958647692f;

// Backward-compatibility aliases
constexpr int SAMPLE_RATE         = kSampleRate;
constexpr int FRAME_SIZE          = kFrameSize;
constexpr int HOP_SIZE            = kHopSize;
constexpr int FFT_SIZE            = kFftSize;
constexpr int NUM_FREQ_BINS       = kNumFreqBins;
constexpr int NUM_COS_COEFFS      = kNumCosCoeffs;
constexpr int NUM_SIN_COEFFS      = kNumSinCoeffs;
constexpr int NUM_STATIC_FEATURES = kNumStaticFeatures;
constexpr int NUM_DELTA_FEATURES  = kNumDeltaFeatures;
constexpr int NUM_DDELTA_FEATURES = kNumDDeltaFeatures;
constexpr int TOTAL_FEATURES      = kTotalFeatures;
constexpr int DELTA_WINDOW        = kDeltaWindow;
constexpr float DEFAULT_RMS_GATE  = kDefaultRmsGate;
constexpr float PI_F              = kPi;
constexpr float TWO_PI_F          = kTwoPi;

// Error checking macros for CUDA and cuFFT
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            throw std::runtime_error("CUDA API Call Failed");                 \
        }                                                                     \
    } while (0)

#define CUFFT_CHECK(call)                                                     \
    do {                                                                      \
        cufftResult err = call;                                               \
        if (err != CUFFT_SUCCESS) {                                           \
            std::cerr << "[cuFFT Error] code: " << err                        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            throw std::runtime_error("cuFFT API Call Failed");                \
        }                                                                     \
    } while (0)
#endif

// Representation of audio signal loaded from WAV
struct AudioData {
    std::string file_path;
    std::vector<float> samples;     // Normalized float samples in [-1.0, 1.0]
    int32_t sample_rate = kSampleRate;
    int32_t num_channels = 1;      // Channels in source file (downmixed to 1)
    int32_t total_samples = 0;     // Total mono samples
    int32_t num_frames = 0;        // Total overlapping frames
    float duration_seconds = 0.0f; // Total duration in seconds
};

// Representation of extracted audio feature matrix
struct FeatureMatrix {
    std::string file_name;
    int32_t num_frames = 0;
    int32_t num_features = kTotalFeatures; // Always 60
    std::vector<float> data;               // Row-major: [num_frames * 60]
    std::vector<float> rms_energies;       // [num_frames]
    std::vector<float> frame_times;        // [num_frames] in seconds

    void allocate(int32_t frames) {
        num_frames = frames;
        num_features = kTotalFeatures;
        data.assign(static_cast<size_t>(frames) * kTotalFeatures, 0.0f);
        rms_energies.assign(frames, 0.0f);
        frame_times.assign(frames, 0.0f);
        for (int32_t t = 0; t < frames; ++t) {
            frame_times[t] = static_cast<float>(t * kHopSize) / static_cast<float>(kSampleRate);
        }
    }

    float& at(int32_t frame, int32_t feature) {
        return data[frame * kTotalFeatures + feature];
    }

    float at(int32_t frame, int32_t feature) const {
        return data[frame * kTotalFeatures + feature];
    }
};

// Execution timing measurement structure
struct ExecutionTimings {
    double cpu_total_ms = 0.0;
    double gpu_h2d_ms = 0.0;
    double gpu_kernel_ms = 0.0;
    double gpu_d2h_ms = 0.0;
    double gpu_total_ms = 0.0;
    double speedup_total = 0.0;
    double speedup_kernel = 0.0;
};

// Numerical validation structure for CPU vs GPU comparison
struct ValidationResults {
    float max_abs_error = 0.0f;
    float mean_abs_error = 0.0f;
    float rmse = 0.0f;
    int32_t max_err_frame = 0;
    int32_t max_err_feature = 0;
    bool passed = false;
    float tolerance_mae = 1e-4f;
    float tolerance_max = 1e-3f;
};

}  // namespace audio_phase

// Expose commonly used structures in global namespace
using audio_phase::AudioData;
using audio_phase::FeatureMatrix;
using audio_phase::ExecutionTimings;
using audio_phase::ValidationResults;
using audio_phase::SAMPLE_RATE;
using audio_phase::FRAME_SIZE;
using audio_phase::HOP_SIZE;
using audio_phase::FFT_SIZE;
using audio_phase::NUM_FREQ_BINS;
using audio_phase::NUM_STATIC_FEATURES;
using audio_phase::NUM_DELTA_FEATURES;
using audio_phase::NUM_DDELTA_FEATURES;
using audio_phase::TOTAL_FEATURES;
using audio_phase::DEFAULT_RMS_GATE;

#endif  // SRC_COMMON_H_
