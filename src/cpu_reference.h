#ifndef CPU_REFERENCE_H
#define CPU_REFERENCE_H

#include "common.h"
#include <complex>
#include <vector>

class CpuReferenceProcessor {
public:
    CpuReferenceProcessor();

    // Executes the complete audio feature extraction pipeline on CPU (single-threaded reference)
    FeatureMatrix process(const AudioData& audio,
                          float rms_gate = DEFAULT_RMS_GATE,
                          double* elapsed_ms = nullptr);

    // Fast Cooley-Tukey 512-point Radix-2 Real-to-Complex FFT matching cuFFT R2C forward transform
    static void compute_fft_512(const float* input_real_512,
                                std::complex<float>* output_complex_256);

private:
    std::vector<float> hann_window_;
    std::vector<float> expected_phase_advances_;
    std::vector<float> cepstral_basis_;

    void precompute_tables();
};

#endif // CPU_REFERENCE_H
