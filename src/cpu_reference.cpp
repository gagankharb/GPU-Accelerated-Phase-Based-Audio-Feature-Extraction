#include "cpu_reference.h"
#include "feature_extraction.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

CpuReferenceProcessor::CpuReferenceProcessor() {
    precompute_tables();
}

void CpuReferenceProcessor::precompute_tables() {
    hann_window_ = generate_hann_window(FRAME_SIZE);
    expected_phase_advances_ = generate_expected_phase_advances(NUM_FREQ_BINS, HOP_SIZE, FFT_SIZE);
    cepstral_basis_ = generate_cepstral_basis(NUM_STATIC_FEATURES, NUM_FREQ_BINS);
}

void CpuReferenceProcessor::compute_fft_512(const float* input_real_512,
                                            std::complex<float>* output_complex_256) {
    constexpr int N = FFT_SIZE; // 512
    constexpr int LOG2_N = 9;   // 2^9 = 512
    std::complex<float> buffer[N];

    // Bit-reversal permutation
    for (int i = 0; i < N; ++i) {
        int rev = 0;
        for (int b = 0; b < LOG2_N; ++b) {
            if ((i >> b) & 1) {
                rev |= (1 << (LOG2_N - 1 - b));
            }
        }
        buffer[rev] = std::complex<float>(input_real_512[i], 0.0f);
    }

    // Cooley-Tukey butterfly stages
    for (int len = 2; len <= N; len <<= 1) {
        float angle = -TWO_PI_F / static_cast<float>(len);
        std::complex<float> wlen(std::cos(angle), std::sin(angle));
        int half = len >> 1;

        for (int i = 0; i < N; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < half; ++j) {
                std::complex<float> u = buffer[i + j];
                std::complex<float> v = buffer[i + j + half] * w;
                buffer[i + j] = u + v;
                buffer[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }

    // Copy the first NUM_FREQ_BINS (256 bins, k = 0 .. 255) to output
    for (int k = 0; k < NUM_FREQ_BINS; ++k) {
        output_complex_256[k] = buffer[k];
    }
}

FeatureMatrix CpuReferenceProcessor::process(const AudioData& audio,
                                             float rms_gate,
                                             double* elapsed_ms) {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int num_frames = audio.num_frames;
    FeatureMatrix features;
    features.allocate(num_frames);

    if (num_frames <= 0) {
        if (elapsed_ms) *elapsed_ms = 0.0;
        return features;
    }

    // Temporary storage for frames and intermediate spectral representations
    std::vector<float> padded_frame(FFT_SIZE, 0.0f);
    std::vector<std::complex<float>> fft_bins(NUM_FREQ_BINS);
    std::vector<float> curr_phase(NUM_FREQ_BINS);
    std::vector<float> prev_phase(NUM_FREQ_BINS, 0.0f);

    // Buffers for static, delta, and delta-delta features: [num_frames x 20]
    std::vector<float> static_features(static_cast<size_t>(num_frames) * NUM_STATIC_FEATURES, 0.0f);
    std::vector<float> delta_features(static_cast<size_t>(num_frames) * NUM_DELTA_FEATURES, 0.0f);
    std::vector<float> ddelta_features(static_cast<size_t>(num_frames) * NUM_DDELTA_FEATURES, 0.0f);
    std::vector<float> phase_residuals(NUM_FREQ_BINS, 0.0f);

    // Step 1 - 4: Frame division, Hann windowing, RMS gating, FFT, Phase residual, and Static Cepstral Features
    for (int t = 0; t < num_frames; ++t) {
        int sample_offset = t * HOP_SIZE;

        // Step 1: Compute RMS energy of raw audio frame
        float energy_sum = 0.0f;
        for (int m = 0; m < FRAME_SIZE; ++m) {
            float s = audio.samples[sample_offset + m];
            energy_sum += s * s;
        }
        float frame_rms = std::sqrt(energy_sum / static_cast<float>(FRAME_SIZE));
        features.rms_energies[t] = frame_rms;

        // Step 2: Apply Hann window and zero-pad to 512
        for (int m = 0; m < FRAME_SIZE; ++m) {
            padded_frame[m] = audio.samples[sample_offset + m] * hann_window_[m];
        }
        for (int m = FRAME_SIZE; m < FFT_SIZE; ++m) {
            padded_frame[m] = 0.0f;
        }

        // Step 3: Compute FFT on CPU
        compute_fft_512(padded_frame.data(), fft_bins.data());

        // Step 4: Extract Phase
        for (int k = 0; k < NUM_FREQ_BINS; ++k) {
            float re = fft_bins[k].real();
            float im = fft_bins[k].imag();
            curr_phase[k] = std::atan2(im, re);
        }

        // Step 5: Phase residual calculation with RMS gating and phase wrapping
        // Δφ(k,t) = wrap(φ(k,t) - φ(k,t-1) - 2πkH/N)
        bool frame_is_silent = (frame_rms < rms_gate);
        bool prev_is_silent = (t == 0) || (features.rms_energies[t - 1] < rms_gate);

        for (int k = 0; k < NUM_FREQ_BINS; ++k) {
            if (t == 0 || frame_is_silent || prev_is_silent) {
                phase_residuals[k] = 0.0f;
            } else {
                float raw_diff = curr_phase[k] - prev_phase[k] - expected_phase_advances_[k];
                phase_residuals[k] = wrap_phase(raw_diff);
            }
        }
        prev_phase = curr_phase;

        // Step 6: Extract 20 Static Phase Cepstral Coefficients (10 Cosine + 10 Sine)
        // c_n = (1 / K) * sum_{k=0}^{K-1} phase_residual(k) * basis(n, k)
        const float inv_k = 1.0f / static_cast<float>(NUM_FREQ_BINS);
        for (int feat = 0; feat < NUM_STATIC_FEATURES; ++feat) {
            float coeff_sum = 0.0f;
            const float* basis_row = &cepstral_basis_[feat * NUM_FREQ_BINS];
            for (int k = 0; k < NUM_FREQ_BINS; ++k) {
                coeff_sum += phase_residuals[k] * basis_row[k];
            }
            static_features[t * NUM_STATIC_FEATURES + feat] = coeff_sum * inv_k;
        }
    }

    // Step 7: Delta Features (1st-order temporal derivatives)
    for (int t = 0; t < num_frames; ++t) {
        for (int feat = 0; feat < NUM_STATIC_FEATURES; ++feat) {
            delta_features[t * NUM_DELTA_FEATURES + feat] =
                compute_delta_sample(static_features.data(), t, feat, num_frames, NUM_STATIC_FEATURES);
        }
    }

    // Step 8: Delta-Delta Features (2nd-order temporal derivatives)
    for (int t = 0; t < num_frames; ++t) {
        for (int feat = 0; feat < NUM_DELTA_FEATURES; ++feat) {
            ddelta_features[t * NUM_DDELTA_FEATURES + feat] =
                compute_delta_sample(delta_features.data(), t, feat, num_frames, NUM_DELTA_FEATURES);
        }
    }

    // Step 9: Assemble final 60-dimensional feature matrix
    for (int t = 0; t < num_frames; ++t) {
        for (int feat = 0; feat < NUM_STATIC_FEATURES; ++feat) {
            features.at(t, feat) = static_features[t * NUM_STATIC_FEATURES + feat];
            features.at(t, NUM_STATIC_FEATURES + feat) = delta_features[t * NUM_DELTA_FEATURES + feat];
            features.at(t, NUM_STATIC_FEATURES + NUM_DELTA_FEATURES + feat) = ddelta_features[t * NUM_DDELTA_FEATURES + feat];
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    if (elapsed_ms) {
        *elapsed_ms = duration;
    }

    return features;
}
