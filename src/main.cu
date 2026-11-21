// Copyright 2026 Coursera GPU Specialization Capstone
// Author: Gagan
// Project: GPU-Accelerated Phase-Based Audio Feature Extraction

#include "common.h"
#include "audio_io.h"
#include "cpu_reference.h"
#include "gpu_processing.h"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace audio_phase {

void PrintUsage(const char* prog_name) {
    std::cout << "================================================================================\n"
              << "GPU-Accelerated Phase-Based Audio Feature Extraction (CUDA & cuFFT)\n"
              << "Coursera GPU Programming Specialization Capstone Project\n"
              << "================================================================================\n"
              << "Usage:\n"
              << "  " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --input <path>     Path to input 16 kHz 16-bit PCM WAV file\n"
              << "                     (default: data/demo.wav; auto-generated if missing)\n"
              << "  --batch <dir>      Path to directory of WAV files to process in a single execution\n"
              << "  --output <path>    Path to output CSV feature file\n"
              << "                     (default: output/features.csv)\n"
              << "  --backend <mode>   Processing backend: 'gpu', 'cpu', or 'both'\n"
              << "                     (default: both)\n"
              << "  --gate <val>       RMS energy silence threshold for phase gating\n"
              << "                     (default: 1e-4)\n"
              << "  --help, -h         Display this help message and exit\n\n"
              << "Examples:\n"
              << "  " << prog_name << " --input data/demo.wav --output output/features.csv --backend gpu\n"
              << "  " << prog_name << " --input data/demo.wav --output output/features_cpu.csv --backend cpu\n"
              << "  " << prog_name << " --input data/demo.wav --output output/features_both.csv --backend both\n"
              << "  " << prog_name << " --batch data/ --backend both\n"
              << "================================================================================\n";
}

ValidationResults ValidateFeatures(const FeatureMatrix& cpu_feats,
                                  const FeatureMatrix& gpu_feats,
                                  float tol_mae = 1e-4f,
                                  float tol_max = 1e-3f) {
    ValidationResults res;
    res.tolerance_mae = tol_mae;
    res.tolerance_max = tol_max;

    if (cpu_feats.num_frames != gpu_feats.num_frames ||
        cpu_feats.num_features != gpu_feats.num_features) {
        std::cerr << "[Validation Error] Dimension mismatch: CPU ("
                  << cpu_feats.num_frames << "x" << cpu_feats.num_features
                  << ") vs GPU (" << gpu_feats.num_frames << "x"
                  << gpu_feats.num_features << ")\n";
        res.passed = false;
        return res;
    }

    const int total_elements = cpu_feats.num_frames * kTotalFeatures;
    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    float max_err = 0.0f;
    int max_frame = 0;
    int max_feat = 0;

    for (int t = 0; t < cpu_feats.num_frames; ++t) {
        for (int f = 0; f < kTotalFeatures; ++f) {
            float c_val = cpu_feats.at(t, f);
            float g_val = gpu_feats.at(t, f);
            float diff = std::abs(c_val - g_val);

            sum_abs_err += diff;
            sum_sq_err += (diff * diff);

            if (diff > max_err) {
                max_err = diff;
                max_frame = t;
                max_feat = f;
            }
        }
    }

    res.max_abs_error = max_err;
    res.mean_abs_error = static_cast<float>(sum_abs_err / total_elements);
    res.rmse = std::sqrt(static_cast<float>(sum_sq_err / total_elements));
    res.max_err_frame = max_frame;
    res.max_err_feature = max_feat;
    res.passed = (res.mean_abs_error <= tol_mae) && (res.max_abs_error <= tol_max);

    return res;
}

bool ProcessSingleFile(const std::string& input_path,
                       const std::string& output_path,
                       std::string backend,
                       float rms_gate,
                       bool gpu_available,
                       ExecutionTimings* out_timings,
                       AudioData* out_audio) {
    AudioData audio;
    try {
        audio = load_wav_file(input_path);
    } catch (const std::exception& ex) {
        std::cerr << "[Fatal Error] " << ex.what() << "\n";
        return false;
    }

    if (out_audio) *out_audio = audio;

    ExecutionTimings timings;
    FeatureMatrix cpu_features;
    FeatureMatrix gpu_features;
    ValidationResults val_results;
    bool ran_cpu = false;
    bool ran_gpu = false;

    if (backend == "cpu" || backend == "both") {
        std::cout << "\n[Execution] Running Single-Threaded CPU Reference Implementation on "
                  << fs::path(input_path).filename().string() << "...\n";
        CpuReferenceProcessor cpu_proc;
        double cpu_ms = 0.0;
        cpu_features = cpu_proc.process(audio, rms_gate, &cpu_ms);
        timings.cpu_total_ms = cpu_ms;
        ran_cpu = true;
        std::cout << "[Execution] CPU Reference completed in: " << std::fixed
                  << std::setprecision(3) << cpu_ms << " ms\n";
    }

    if ((backend == "gpu" || backend == "both") && gpu_available) {
        std::cout << "\n[Execution] Running GPU-Accelerated CUDA/cuFFT Implementation on "
                  << fs::path(input_path).filename().string() << "...\n";
        try {
            gpu_features = process_audio_gpu(audio, rms_gate, &timings);
            ran_gpu = true;
            std::cout << "[Execution] GPU Pipeline completed in:\n"
                      << "            - Host-to-Device (H2D)  : " << std::fixed << std::setprecision(3) << timings.gpu_h2d_ms << " ms\n"
                      << "            - Kernel Execution      : " << std::fixed << std::setprecision(3) << timings.gpu_kernel_ms << " ms\n"
                      << "            - Device-to-Host (D2H)  : " << std::fixed << std::setprecision(3) << timings.gpu_d2h_ms << " ms\n"
                      << "            - Total GPU Processing  : " << std::fixed << std::setprecision(3) << timings.gpu_total_ms << " ms\n";
        } catch (const std::exception& ex) {
            std::cerr << "[GPU Execution Error] " << ex.what() << "\n";
            ran_gpu = false;
        }
    }

    if (ran_cpu && ran_gpu) {
        timings.speedup_total = (timings.gpu_total_ms > 0.0) ? (timings.cpu_total_ms / timings.gpu_total_ms) : 0.0;
        timings.speedup_kernel = (timings.gpu_kernel_ms > 0.0) ? (timings.cpu_total_ms / timings.gpu_kernel_ms) : 0.0;

        std::cout << "\n--------------------------------------------------------------------------------\n"
                  << "PERFORMANCE & SPEEDUP SUMMARY (" << fs::path(input_path).filename().string() << ")\n"
                  << "--------------------------------------------------------------------------------\n"
                  << "CPU Execution Time     : " << std::fixed << std::setprecision(3) << timings.cpu_total_ms << " ms\n"
                  << "GPU Total Time (w/ IO) : " << std::fixed << std::setprecision(3) << timings.gpu_total_ms << " ms\n"
                  << "GPU Pure Kernel Time   : " << std::fixed << std::setprecision(3) << timings.gpu_kernel_ms << " ms\n"
                  << "Overall Speedup (Total): " << std::fixed << std::setprecision(2) << timings.speedup_total << "x\n"
                  << "Kernel-Only Speedup    : " << std::fixed << std::setprecision(2) << timings.speedup_kernel << "x\n";

        std::cout << "\n--------------------------------------------------------------------------------\n"
                  << "NUMERICAL VALIDATION (CPU vs GPU)\n"
                  << "--------------------------------------------------------------------------------\n";
        val_results = ValidateFeatures(cpu_features, gpu_features);
        std::cout << "Mean Absolute Error (MAE) : " << std::scientific << std::setprecision(4) << val_results.mean_abs_error << " (Tol: " << val_results.tolerance_mae << ")\n"
                  << "Max Absolute Error (MaxE) : " << std::scientific << std::setprecision(4) << val_results.max_abs_error  << " (Tol: " << val_results.tolerance_max << ")\n"
                  << "Root Mean Square Error    : " << std::scientific << std::setprecision(4) << val_results.rmse << "\n"
                  << "Validation Result         : " << (val_results.passed ? "PASSED [SUCCESS]" : "FAILED [CHECK TOLERANCE]") << "\n";
    }

    const FeatureMatrix& out_matrix = (ran_gpu) ? gpu_features : cpu_features;
    std::string out_backend = (ran_gpu && ran_cpu) ? "GPU (Validated with CPU)" : (ran_gpu ? "GPU" : "CPU");
    save_features_to_csv(output_path, out_matrix, out_backend);

    std::string summary_path = output_path.substr(0, output_path.find_last_of('.')) + "_summary.txt";
    save_summary_report(summary_path, audio, timings, (ran_cpu && ran_gpu) ? &val_results : nullptr, out_backend);

    if (out_timings) *out_timings = timings;
    return true;
}

}  // namespace audio_phase

int main(int argc, char* argv[]) {
    std::string input_path = "data/demo.wav";
    std::string batch_dir = "";
    std::string output_path = "output/features.csv";
    std::string backend = "both";
    float rms_gate = audio_phase::kDefaultRmsGate;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            audio_phase::PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
            if (backend != "gpu" && backend != "cpu" && backend != "both") {
                std::cerr << "[Error] Invalid backend '" << backend << "'. Choose 'gpu', 'cpu', or 'both'.\n";
                return 1;
            }
        } else if (arg == "--gate" && i + 1 < argc) {
            rms_gate = std::stof(argv[++i]);
        } else {
            std::cerr << "[Error] Unknown argument: " << arg << "\n";
            audio_phase::PrintUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "================================================================================\n"
              << "GPU-ACCELERATED PHASE-BASED AUDIO FEATURE EXTRACTION (CUDA & cuFFT)\n"
              << "Coursera GPU Programming Specialization Capstone\n"
              << "================================================================================\n";

    std::string gpu_name;
    int cc_major = 0, cc_minor = 0;
    bool gpu_available = check_cuda_device_available(&gpu_name, &cc_major, &cc_minor);

    if (gpu_available) {
        std::cout << "[Hardware] CUDA Device Detected: " << gpu_name
                  << " (Compute Capability " << cc_major << "." << cc_minor << ")\n";
    } else {
        std::cout << "[Hardware] Note: No NVIDIA CUDA GPU detected on current host.\n";
        if (backend == "gpu" || backend == "both") {
            std::cout << "[Hardware] Falling back to CPU backend for execution on this host.\n";
            if (backend == "gpu") backend = "cpu";
        }
    }
    std::cout << "--------------------------------------------------------------------------------\n";

    // Batch Mode Execution
    if (!batch_dir.empty()) {
        std::cout << "[Batch Mode] Scanning directory: '" << batch_dir << "' for WAV audio files...\n";
        std::vector<std::string> wav_files = audio_phase::discover_wav_files(batch_dir);
        if (wav_files.empty()) {
            std::cerr << "[Batch Mode Error] No .wav files found in directory: " << batch_dir << "\n";
            return 1;
        }

        std::cout << "[Batch Mode] Discovered " << wav_files.size() << " WAV audio files to process in a single execution:\n";
        for (size_t idx = 0; idx < wav_files.size(); ++idx) {
            std::cout << "  [" << (idx + 1) << "] " << wav_files[idx] << "\n";
        }

        auto batch_start = std::chrono::high_resolution_clock::now();
        std::vector<AudioData> audio_batch;
        std::vector<ExecutionTimings> timing_batch;

        for (size_t idx = 0; idx < wav_files.size(); ++idx) {
            std::string stem = fs::path(wav_files[idx]).stem().string();
            std::string file_out_csv = "output/features_" + stem + ".csv";

            std::cout << "\n--------------------------------------------------------------------------------\n"
                      << "Processing File [" << (idx + 1) << "/" << wav_files.size() << "]: " << wav_files[idx] << "\n"
                      << "--------------------------------------------------------------------------------\n";

            ExecutionTimings t;
            AudioData a;
            if (audio_phase::ProcessSingleFile(wav_files[idx], file_out_csv, backend, rms_gate, gpu_available, &t, &a)) {
                audio_batch.push_back(a);
                timing_batch.push_back(t);
            }
        }

        auto batch_end = std::chrono::high_resolution_clock::now();
        double total_wall_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

        std::string batch_summary_path = "output/batch_execution_summary.txt";
        audio_phase::save_batch_summary_report(batch_summary_path, audio_batch, timing_batch, total_wall_ms, backend);

        std::cout << "\n================================================================================\n"
                  << "BATCH PROCESSING COMPLETE (SINGLE SOFTWARE EXECUTION)\n"
                  << "================================================================================\n"
                  << "Files Processed       : " << audio_batch.size() << "\n"
                  << "Total Wall-Clock Time : " << std::fixed << std::setprecision(2) << total_wall_ms << " ms\n"
                  << "Batch Report Saved To : " << batch_summary_path << "\n"
                  << "================================================================================\n";
        return 0;
    }

    // Single File Mode Execution
    ExecutionTimings timings;
    AudioData audio;
    bool success = audio_phase::ProcessSingleFile(input_path, output_path, backend, rms_gate, gpu_available, &timings, &audio);
    return success ? 0 : 1;
}
