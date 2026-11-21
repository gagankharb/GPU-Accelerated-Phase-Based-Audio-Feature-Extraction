#ifndef GPU_PROCESSING_H
#define GPU_PROCESSING_H

#include "common.h"

// Checks whether an NVIDIA CUDA capable GPU device is available on the host system.
// Populates device_name and compute_capability if available.
bool check_cuda_device_available(std::string* device_name = nullptr,
                                 int* compute_major = nullptr,
                                 int* compute_minor = nullptr);

// Executes the complete audio feature extraction pipeline on GPU using CUDA and cuFFT.
// Measures H2D transfer, kernel execution, and D2H transfer using CUDA events.
FeatureMatrix process_audio_gpu(const AudioData& audio,
                                float rms_gate = DEFAULT_RMS_GATE,
                                ExecutionTimings* timings = nullptr);

#endif // GPU_PROCESSING_H
