// Copyright 2026 Coursera GPU Specialization Capstone
// Author: Gagan
// Project: GPU-Accelerated Phase-Based Audio Feature Extraction

#ifndef SRC_AUDIO_IO_H_
#define SRC_AUDIO_IO_H_

#include "common.h"
#include <string>
#include <vector>

namespace audio_phase {

// Reads a 16-bit PCM WAV file, handles multichannel by downmixing to mono,
// validates 16 kHz sample rate, and normalizes samples to [-1.0, 1.0].
AudioData load_wav_file(const std::string& file_path);

// Writes a synthetic 16 kHz 16-bit PCM WAV file with test signals
// (harmonic tones, pitch sweeps, and silence segments) for testing and benchmarking.
bool generate_synthetic_wav(const std::string& output_path, float duration_seconds = 2.0f);

// Discovers all .wav files in a given directory for batch multi-file execution.
std::vector<std::string> discover_wav_files(const std::string& directory_path);

// Saves the extracted 60-dimensional feature matrix to a CSV file.
// Format: frame_index, time_sec, rms_energy, 20 static coeffs, 20 delta coeffs, 20 delta-delta coeffs.
bool save_features_to_csv(const std::string& output_path,
                          const FeatureMatrix& features,
                          const std::string& backend_name);

// Saves validation results and execution timings to a companion summary file.
bool save_summary_report(const std::string& output_path,
                         const AudioData& audio,
                         const ExecutionTimings& timings,
                         const ValidationResults* validation,
                         const std::string& backend_name);

// Saves aggregate execution metrics for batch processing of multiple audio files.
bool save_batch_summary_report(const std::string& output_path,
                               const std::vector<AudioData>& audio_batch,
                               const std::vector<ExecutionTimings>& timing_batch,
                               double total_wall_clock_ms,
                               const std::string& backend_name);

}  // namespace audio_phase

using audio_phase::load_wav_file;
using audio_phase::generate_synthetic_wav;
using audio_phase::discover_wav_files;
using audio_phase::save_features_to_csv;
using audio_phase::save_summary_report;
using audio_phase::save_batch_summary_report;

#endif  // SRC_AUDIO_IO_H_
