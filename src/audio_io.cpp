// Copyright 2026 Coursera GPU Specialization Capstone
// Author: Gagan
// Project: GPU-Accelerated Phase-Based Audio Feature Extraction

#include "audio_io.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace audio_phase {

#pragma pack(push, 1)
struct RiffHeader {
    char chunk_id[4];      // "RIFF"
    uint32_t chunk_size;
    char format[4];        // "WAVE"
};

struct ChunkHeader {
    char subchunk_id[4];
    uint32_t subchunk_size;
};

struct FmtChunkData {
    uint16_t audio_format;    // 1 for PCM
    uint16_t num_channels;    // 1 for mono, 2 for stereo, etc.
    uint32_t sample_rate;     // e.g. 16000
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample; // 16
};
#pragma pack(pop)

AudioData load_wav_file(const std::string& file_path) {
    AudioData audio;
    audio.file_path = file_path;
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "[Audio I/O Warning] Could not open audio file: '" << file_path << "'.\n"
                  << "[Audio I/O Info] Generating fallback synthetic demo WAV at: '" << file_path << "'...\n";
        if (generate_synthetic_wav(file_path, 2.0f)) {
            file.open(file_path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open newly generated fallback audio file: " + file_path);
            }
        } else {
            throw std::runtime_error("Failed to generate fallback audio file: " + file_path);
        }
    }

    RiffHeader riff_hdr;
    if (!file.read(reinterpret_cast<char*>(&riff_hdr), sizeof(RiffHeader))) {
        throw std::runtime_error("Invalid WAV file: unable to read RIFF header in " + file_path);
    }

    if (std::strncmp(riff_hdr.chunk_id, "RIFF", 4) != 0 ||
        std::strncmp(riff_hdr.format, "WAVE", 4) != 0) {
        throw std::runtime_error("Invalid audio format: file is not a valid RIFF/WAVE container: " + file_path);
    }

    bool found_fmt = false;
    bool found_data = false;
    FmtChunkData fmt_data;
    uint32_t data_bytes = 0;
    std::streampos data_pos = 0;

    // Iterate through chunks to locate 'fmt ' and 'data' chunks safely
    while (file.good() && (!found_fmt || !found_data)) {
        ChunkHeader chunk;
        if (!file.read(reinterpret_cast<char*>(&chunk), sizeof(ChunkHeader))) {
            break;
        }

        if (std::strncmp(chunk.subchunk_id, "fmt ", 4) == 0) {
            if (chunk.subchunk_size < sizeof(FmtChunkData)) {
                throw std::runtime_error("Unsupported WAV format: corrupted fmt chunk size");
            }
            if (!file.read(reinterpret_cast<char*>(&fmt_data), sizeof(FmtChunkData))) {
                throw std::runtime_error("Failed to read fmt chunk data");
            }
            // Skip any extra format bytes if present
            if (chunk.subchunk_size > sizeof(FmtChunkData)) {
                file.seekg(chunk.subchunk_size - sizeof(FmtChunkData), std::ios::cur);
            }
            found_fmt = true;
        } else if (std::strncmp(chunk.subchunk_id, "data", 4) == 0) {
            data_bytes = chunk.subchunk_size;
            data_pos = file.tellg();
            found_data = true;
            file.seekg(chunk.subchunk_size, std::ios::cur);
        } else {
            // Skip unhandled metadata chunk (e.g. LIST, JUNK, bext)
            file.seekg(chunk.subchunk_size, std::ios::cur);
        }
    }

    if (!found_fmt) {
        throw std::runtime_error("Invalid WAV file: missing 'fmt ' subchunk in " + file_path);
    }
    if (!found_data) {
        throw std::runtime_error("Invalid WAV file: missing 'data' subchunk in " + file_path);
    }

    if (fmt_data.audio_format != 1) {
        throw std::runtime_error("Unsupported audio format: expected uncompressed PCM (format 1), got " +
                                 std::to_string(fmt_data.audio_format));
    }
    if (fmt_data.bits_per_sample != 16) {
        throw std::runtime_error("Unsupported bit depth: expected 16-bit PCM audio, got " +
                                 std::to_string(fmt_data.bits_per_sample) + " bits per sample.");
    }
    if (fmt_data.sample_rate != static_cast<uint32_t>(kSampleRate)) {
        throw std::runtime_error("Unsupported sampling rate: Expected 16000 Hz, but found " +
                                 std::to_string(fmt_data.sample_rate) + " Hz in " + file_path);
    }

    int channels = fmt_data.num_channels;
    audio.sample_rate = fmt_data.sample_rate;
    audio.num_channels = channels;

    int total_interleaved_samples = data_bytes / sizeof(int16_t);
    int num_mono_samples = total_interleaved_samples / channels;

    if (num_mono_samples < kFrameSize) {
        throw std::runtime_error("Audio file too short: length is " + std::to_string(num_mono_samples) +
                                 " samples, but frame analysis requires at least " +
                                 std::to_string(kFrameSize) + " samples.");
    }

    file.clear();
    file.seekg(data_pos, std::ios::beg);
    std::vector<int16_t> raw_pcm(total_interleaved_samples);
    if (!file.read(reinterpret_cast<char*>(raw_pcm.data()), data_bytes)) {
        throw std::runtime_error("Failed to read PCM audio payload from " + file_path);
    }

    audio.samples.resize(num_mono_samples);
    const float norm_scale = 1.0f / 32768.0f;
    const float channel_scale = 1.0f / static_cast<float>(channels);

    for (int i = 0; i < num_mono_samples; ++i) {
        float sample_sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sample_sum += static_cast<float>(raw_pcm[i * channels + c]);
        }
        audio.samples[i] = (sample_sum * channel_scale) * norm_scale;
    }

    audio.total_samples = num_mono_samples;
    audio.num_frames = 1 + (num_mono_samples - kFrameSize) / kHopSize;
    audio.duration_seconds = static_cast<float>(num_mono_samples) / static_cast<float>(kSampleRate);

    std::cout << "[Audio I/O] Loaded '" << file_path << "' successfully:\n"
              << "            Channels: " << channels << (channels > 1 ? " (downmixed to mono)" : " (mono)") << "\n"
              << "            Sample Rate: " << audio.sample_rate << " Hz\n"
              << "            Total Samples: " << audio.total_samples << "\n"
              << "            Duration: " << std::fixed << std::setprecision(2) << audio.duration_seconds << " s\n"
              << "            Analyzable Frames: " << audio.num_frames << "\n";

    return audio;
}

std::vector<std::string> discover_wav_files(const std::string& directory_path) {
    std::vector<std::string> wav_files;
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        return wav_files;
    }

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav") {
                wav_files.push_back(entry.path().string());
            }
        }
    }
    std::sort(wav_files.begin(), wav_files.end());
    return wav_files;
}

bool generate_synthetic_wav(const std::string& output_path, float duration_seconds) {
    int total_samples = static_cast<int>(duration_seconds * kSampleRate);
    std::vector<int16_t> pcm_data(total_samples);

    for (int i = 0; i < total_samples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(kSampleRate);
        float sample = 0.0f;

        if (t < 0.8f) {
            float f0 = 220.0f;
            sample += 0.40f * std::sin(kTwoPi * f0 * t);
            sample += 0.25f * std::sin(kTwoPi * (f0 * 2.0f) * t + 0.3f);
            sample += 0.15f * std::sin(kTwoPi * (f0 * 3.0f) * t + 0.8f);
            sample += 0.10f * std::sin(kTwoPi * (f0 * 4.0f) * t + 1.2f);
        } else if (t < 1.1f) {
            sample = 1e-5f * std::sin(kTwoPi * 60.0f * t);
        } else {
            float t_rel = t - 1.1f;
            float f_start = 300.0f;
            float f_end = 2400.0f;
            float chirp_dur = duration_seconds - 1.1f;
            float instantaneous_freq = f_start + (f_end - f_start) * (t_rel / chirp_dur);
            sample += 0.50f * std::sin(kTwoPi * instantaneous_freq * t_rel);
            sample += 0.20f * std::sin(kTwoPi * (instantaneous_freq * 1.5f) * t_rel + 0.5f);
        }

        sample = std::max(-1.0f, std::min(1.0f, sample));
        pcm_data[i] = static_cast<int16_t>(sample * 32767.0f);
    }

    uint16_t channels = 1;
    uint32_t sample_rate = kSampleRate;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
    uint16_t block_align = channels * (bits_per_sample / 8);
    uint32_t data_bytes = total_samples * sizeof(int16_t);

    RiffHeader riff_hdr;
    std::memcpy(riff_hdr.chunk_id, "RIFF", 4);
    riff_hdr.chunk_size = 36 + data_bytes;
    std::memcpy(riff_hdr.format, "WAVE", 4);

    FmtChunkData fmt_data;
    fmt_data.audio_format = 1;
    fmt_data.num_channels = channels;
    fmt_data.sample_rate = sample_rate;
    fmt_data.byte_rate = byte_rate;
    fmt_data.block_align = block_align;
    fmt_data.bits_per_sample = bits_per_sample;

    ChunkHeader fmt_hdr;
    std::memcpy(fmt_hdr.subchunk_id, "fmt ", 4);
    fmt_hdr.subchunk_size = sizeof(FmtChunkData);

    ChunkHeader data_hdr;
    std::memcpy(data_hdr.subchunk_id, "data", 4);
    data_hdr.subchunk_size = data_bytes;

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[Audio I/O Error] Cannot create synthetic audio file at: " << output_path << "\n";
        return false;
    }

    out.write(reinterpret_cast<const char*>(&riff_hdr), sizeof(RiffHeader));
    out.write(reinterpret_cast<const char*>(&fmt_hdr), sizeof(ChunkHeader));
    out.write(reinterpret_cast<const char*>(&fmt_data), sizeof(FmtChunkData));
    out.write(reinterpret_cast<const char*>(&data_hdr), sizeof(ChunkHeader));
    out.write(reinterpret_cast<const char*>(pcm_data.data()), data_bytes);

    std::cout << "[Audio I/O] Generated synthetic test audio: '" << output_path << "' ("
              << duration_seconds << "s, 16 kHz PCM 16-bit mono).\n";
    return true;
}

bool save_features_to_csv(const std::string& output_path,
                          const FeatureMatrix& features,
                          const std::string& backend_name) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "[Audio I/O Error] Could not open output CSV path: " << output_path << "\n";
        return false;
    }

    out << "frame_index,time_seconds,rms_energy";
    for (int i = 1; i <= kNumCosCoeffs; ++i) out << ",cos_cepstral_" << i;
    for (int i = 1; i <= kNumSinCoeffs; ++i) out << ",sin_cepstral_" << i;
    for (int i = 1; i <= kNumCosCoeffs; ++i) out << ",delta_cos_" << i;
    for (int i = 1; i <= kNumSinCoeffs; ++i) out << ",delta_sin_" << i;
    for (int i = 1; i <= kNumCosCoeffs; ++i) out << ",ddelta_cos_" << i;
    for (int i = 1; i <= kNumSinCoeffs; ++i) out << ",ddelta_sin_" << i;
    out << "\n";

    out << std::fixed << std::setprecision(6);
    for (int t = 0; t < features.num_frames; ++t) {
        out << t << "," << features.frame_times[t] << "," << features.rms_energies[t];
        for (int f = 0; f < kTotalFeatures; ++f) {
            out << "," << features.at(t, f);
        }
        out << "\n";
    }

    std::cout << "[Audio I/O] Successfully wrote " << features.num_frames
              << " frames x " << kTotalFeatures << " features to '" << output_path
              << "' (Backend: " << backend_name << ").\n";
    return true;
}

bool save_summary_report(const std::string& output_path,
                         const AudioData& audio,
                         const ExecutionTimings& timings,
                         const ValidationResults* validation,
                         const std::string& backend_name) {
    std::ofstream out(output_path);
    if (!out.is_open()) return false;

    out << "================================================================================\n";
    out << "GPU-ACCELERATED PHASE-BASED AUDIO FEATURE EXTRACTION EXECUTION REPORT\n";
    out << "================================================================================\n\n";
    out << "1. AUDIO INPUT SPECIFICATIONS\n";
    out << "-----------------------------\n";
    out << "File Source       : " << audio.file_path << "\n";
    out << "Sample Rate       : " << audio.sample_rate << " Hz\n";
    out << "Original Channels : " << audio.num_channels << "\n";
    out << "Total Samples     : " << audio.total_samples << "\n";
    out << "Duration          : " << std::fixed << std::setprecision(3) << audio.duration_seconds << " seconds\n";
    out << "Analysis Frames   : " << audio.num_frames << " frames\n";
    out << "Frame Size (L)    : " << kFrameSize << " samples (" << (kFrameSize * 1000.0f / kSampleRate) << " ms)\n";
    out << "Hop Length (H)    : " << kHopSize << " samples (" << (kHopSize * 1000.0f / kSampleRate) << " ms)\n";
    out << "FFT Size (N)      : " << kFftSize << " points\n";
    out << "Frequency Bins (K): " << kNumFreqBins << " bins\n";
    out << "Features / Frame  : " << kTotalFeatures << " (20 static + 20 delta + 20 delta-delta)\n\n";

    out << "2. EXECUTION BENCHMARKS & TIMINGS\n";
    out << "---------------------------------\n";
    out << "Target Backend    : " << backend_name << "\n";
    if (timings.cpu_total_ms > 0.0) {
        out << "CPU Processing    : " << std::fixed << std::setprecision(3) << timings.cpu_total_ms << " ms\n";
    }
    if (timings.gpu_total_ms > 0.0) {
        out << "GPU H2D Transfer  : " << std::fixed << std::setprecision(3) << timings.gpu_h2d_ms << " ms\n";
        out << "GPU Kernel Time   : " << std::fixed << std::setprecision(3) << timings.gpu_kernel_ms << " ms\n";
        out << "GPU D2H Transfer  : " << std::fixed << std::setprecision(3) << timings.gpu_d2h_ms << " ms\n";
        out << "GPU Total Time    : " << std::fixed << std::setprecision(3) << timings.gpu_total_ms << " ms\n";
        if (timings.cpu_total_ms > 0.0) {
            out << "Speedup (Kernel)  : " << std::fixed << std::setprecision(2) << timings.speedup_kernel << "x\n";
            out << "Speedup (Total)   : " << std::fixed << std::setprecision(2) << timings.speedup_total << "x\n";
        }
    }
    out << "\n";

    if (validation != nullptr) {
        out << "3. NUMERICAL VALIDATION (CPU vs GPU)\n";
        out << "------------------------------------\n";
        out << "Max Absolute Error: " << std::scientific << std::setprecision(4) << validation->max_abs_error << "\n";
        out << "Mean Absolute Error: " << std::scientific << std::setprecision(4) << validation->mean_abs_error << "\n";
        out << "RMSE              : " << std::scientific << std::setprecision(4) << validation->rmse << "\n";
        out << "MAE Tolerance     : " << validation->tolerance_mae << "\n";
        out << "Max Tol Tolerance : " << validation->tolerance_max << "\n";
        out << "Validation Status : " << (validation->passed ? "PASSED [SUCCESS]" : "FAILED [DEVIATION]") << "\n";
        out << "Max Error Frame   : " << validation->max_err_frame << ", Feature: " << validation->max_err_feature << "\n";
    }
    out << "================================================================================\n";
    return true;
}

bool save_batch_summary_report(const std::string& output_path,
                               const std::vector<AudioData>& audio_batch,
                               const std::vector<ExecutionTimings>& timing_batch,
                               double total_wall_clock_ms,
                               const std::string& backend_name) {
    std::ofstream out(output_path);
    if (!out.is_open()) return false;

    int32_t total_frames = 0;
    double total_audio_seconds = 0.0;
    double total_cpu_ms = 0.0;
    double total_gpu_ms = 0.0;
    double total_kernel_ms = 0.0;

    for (size_t i = 0; i < audio_batch.size(); ++i) {
        total_frames += audio_batch[i].num_frames;
        total_audio_seconds += audio_batch[i].duration_seconds;
        if (i < timing_batch.size()) {
            total_cpu_ms += timing_batch[i].cpu_total_ms;
            total_gpu_ms += timing_batch[i].gpu_total_ms;
            total_kernel_ms += timing_batch[i].gpu_kernel_ms;
        }
    }

    out << "================================================================================\n";
    out << "BATCH AUDIO EXECUTION REPORT (MULTIPLE FILES IN A SINGLE RUN)\n";
    out << "================================================================================\n";
    out << "Total Files Processed : " << audio_batch.size() << "\n";
    out << "Total Audio Duration  : " << std::fixed << std::setprecision(2) << total_audio_seconds << " seconds\n";
    out << "Total Frames Extracted: " << total_frames << " frames\n";
    out << "Features per Frame    : " << kTotalFeatures << " features\n";
    out << "Total Data Points     : " << (total_frames * kTotalFeatures) << " float values\n";
    out << "Processing Backend    : " << backend_name << "\n";
    out << "Wall-Clock Duration   : " << std::fixed << std::setprecision(2) << total_wall_clock_ms << " ms\n\n";

    out << std::left << std::setw(30) << "File Path"
        << std::setw(12) << "Duration(s)"
        << std::setw(10) << "Frames"
        << std::setw(14) << "CPU(ms)"
        << std::setw(14) << "GPU(ms)"
        << std::setw(14) << "Kernel(ms)"
        << "\n";
    out << std::string(94, '-') << "\n";

    for (size_t i = 0; i < audio_batch.size(); ++i) {
        std::string fname = fs::path(audio_batch[i].file_path).filename().string();
        out << std::left << std::setw(30) << fname
            << std::fixed << std::setprecision(2) << std::setw(12) << audio_batch[i].duration_seconds
            << std::setw(10) << audio_batch[i].num_frames;
        if (i < timing_batch.size()) {
            out << std::setw(14) << timing_batch[i].cpu_total_ms
                << std::setw(14) << timing_batch[i].gpu_total_ms
                << std::setw(14) << timing_batch[i].gpu_kernel_ms;
        }
        out << "\n";
    }
    out << "================================================================================\n";
    return true;
}

}  // namespace audio_phase
