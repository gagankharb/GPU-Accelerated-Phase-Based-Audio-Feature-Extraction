#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include "common.h"
#include <cmath>
#include <vector>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

// Fast inline phase wrapping function: wrap(theta) = theta - 2*pi * round(theta / (2*pi))
// Confines angles strictly to [-pi, pi]
__host__ __device__ inline float wrap_phase(float angle) {
    constexpr float TWO_PI = 6.28318530717958647692f;
    return angle - TWO_PI * roundf(angle / TWO_PI);
}

// Precomputes the symmetric Hann analysis window of length L = FRAME_SIZE (400)
// w(m) = 0.5 * (1 - cos(2*pi*m / (L - 1)))
inline std::vector<float> generate_hann_window(int length = FRAME_SIZE) {
    std::vector<float> window(length);
    for (int m = 0; m < length; ++m) {
        window[m] = 0.5f * (1.0f - std::cos(TWO_PI_F * static_cast<float>(m) / static_cast<float>(length - 1)));
    }
    return window;
}

// Precomputes the nominal expected phase advance for bin k:
// delta_theta_expected(k) = 2 * pi * k * H / N
inline std::vector<float> generate_expected_phase_advances(int num_bins = NUM_FREQ_BINS,
                                                          int hop = HOP_SIZE,
                                                          int nfft = FFT_SIZE) {
    std::vector<float> expected(num_bins);
    for (int k = 0; k < num_bins; ++k) {
        expected[k] = (TWO_PI_F * static_cast<float>(k) * static_cast<float>(hop)) / static_cast<float>(nfft);
    }
    return expected;
}

// Precomputes the Phase Cepstral transform basis matrix:
// Matrix dimensions: [NUM_STATIC_FEATURES x NUM_FREQ_BINS] = [20 x 256]
// Rows 0..9:   Cosine basis: cos(pi * n * (k + 0.5) / K) for n = 1 .. 10
// Rows 10..19: Sine basis:   sin(pi * n * (k + 0.5) / K) for n = 1 .. 10
inline std::vector<float> generate_cepstral_basis(int num_static = NUM_STATIC_FEATURES,
                                                  int num_bins = NUM_FREQ_BINS) {
    std::vector<float> basis(num_static * num_bins);
    const float k_float = static_cast<float>(num_bins);

    for (int i = 0; i < NUM_COS_COEFFS; ++i) {
        int n = i + 1; // n = 1 .. 10
        for (int k = 0; k < num_bins; ++k) {
            float arg = (PI_F * static_cast<float>(n) * (static_cast<float>(k) + 0.5f)) / k_float;
            basis[i * num_bins + k] = std::cos(arg);
        }
    }

    for (int i = 0; i < NUM_SIN_COEFFS; ++i) {
        int n = i + 1; // n = 1 .. 10
        int row = NUM_COS_COEFFS + i; // row = 10 .. 19
        for (int k = 0; k < num_bins; ++k) {
            float arg = (PI_F * static_cast<float>(n) * (static_cast<float>(k) + 0.5f)) / k_float;
            basis[row * num_bins + k] = std::sin(arg);
        }
    }

    return basis;
}

// Regression formula for delta features with window radius D = 2:
// delta(t) = sum_{n=1}^D n * (c(t+n) - c(t-n)) / (2 * sum_{n=1}^D n^2)
// For D = 2: Denominator = 2 * (1^2 + 2^2) = 10.0
// Numerator = 1 * (c(t+1) - c(t-1)) + 2 * (c(t+2) - c(t-2))
// Clamping indices at boundaries [0, T-1]
__host__ __device__ inline float compute_delta_sample(const float* in_feat,
                                                      int t,
                                                      int feat_idx,
                                                      int total_frames,
                                                      int feat_stride) {
    int t_p1 = (t + 1 < total_frames) ? (t + 1) : (total_frames - 1);
    int t_m1 = (t - 1 >= 0) ? (t - 1) : 0;
    int t_p2 = (t + 2 < total_frames) ? (t + 2) : (total_frames - 1);
    int t_m2 = (t - 2 >= 0) ? (t - 2) : 0;

    float val_p1 = in_feat[t_p1 * feat_stride + feat_idx];
    float val_m1 = in_feat[t_m1 * feat_stride + feat_idx];
    float val_p2 = in_feat[t_p2 * feat_stride + feat_idx];
    float val_m2 = in_feat[t_m2 * feat_stride + feat_idx];

    return (1.0f * (val_p1 - val_m1) + 2.0f * (val_p2 - val_m2)) / 10.0f;
}

#endif // FEATURE_EXTRACTION_H
