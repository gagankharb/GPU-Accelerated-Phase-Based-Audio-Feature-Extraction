#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Executes the phase-based audio feature extraction pipeline on an input WAV file
or across an entire directory of WAV files in a single batch execution.
"""

import argparse
import os
import wave
import glob
import time
import numpy as np
import pandas as pd
from test_pipeline import reference_python_pipeline, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, TOTAL_FEATURES


def process_audio_file(input_wav, output_csv):
    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Input file not found: {input_wav}")

    with wave.open(input_wav, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV is supported")
    if framerate != SAMPLE_RATE:
        raise ValueError(f"Sample rate must be {SAMPLE_RATE} Hz, got {framerate}")

    data = np.frombuffer(raw_bytes, dtype=np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    samples = data.astype(np.float32) / 32768.0

    t_start = time.perf_counter()
    features, rms = reference_python_pipeline(samples, SAMPLE_RATE)
    duration_ms = (time.perf_counter() - t_start) * 1000.0

    num_frames = features.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    cols = ["frame_index", "time_seconds", "rms_energy"]
    cols += [f"cos_cepstral_{i}" for i in range(1, 11)]
    cols += [f"sin_cepstral_{i}" for i in range(1, 11)]
    cols += [f"delta_cos_{i}" for i in range(1, 11)]
    cols += [f"delta_sin_{i}" for i in range(1, 11)]
    cols += [f"ddelta_cos_{i}" for i in range(1, 11)]
    cols += [f"ddelta_sin_{i}" for i in range(1, 11)]

    times = np.arange(num_frames) * HOP_SIZE / SAMPLE_RATE
    df_data = np.column_stack([np.arange(num_frames), times, rms, features])
    df = pd.DataFrame(df_data, columns=cols)
    df.to_csv(output_csv, index=False)

    print(f"[Pipeline] Processed '{os.path.basename(input_wav)}': "
          f"{len(samples)} samples ({len(samples)/SAMPLE_RATE:.2f}s) -> "
          f"{num_frames} frames x {TOTAL_FEATURES} features ({duration_ms:.2f} ms) -> {output_csv}")

    return {
        "file": os.path.basename(input_wav),
        "duration_sec": len(samples) / SAMPLE_RATE,
        "frames": num_frames,
        "time_ms": duration_ms
    }


def process_batch(batch_dir, output_dir):
    wav_files = sorted(glob.glob(os.path.join(batch_dir, "*.wav")))
    if not wav_files:
        print(f"[Batch Error] No .wav files found in {batch_dir}")
        return

    print("================================================================================")
    print(f"BATCH EXECUTION: Processing {len(wav_files)} files in a single execution")
    print("================================================================================")

    records = []
    t_start_total = time.perf_counter()
    for wf in wav_files:
        stem = os.path.splitext(os.path.basename(wf))[0]
        out_csv = os.path.join(output_dir, f"features_{stem}.csv")
        rec = process_audio_file(wf, out_csv)
        records.append(rec)
    t_total_ms = (time.perf_counter() - t_start_total) * 1000.0

    total_frames = sum(r["frames"] for r in records)
    total_audio_sec = sum(r["duration_sec"] for r in records)

    summary_path = os.path.join(output_dir, "batch_execution_summary.txt")
    with open(summary_path, "w") as f:
        f.write("================================================================================\n")
        f.write("BATCH AUDIO EXECUTION REPORT (MULTIPLE FILES IN A SINGLE RUN)\n")
        f.write("================================================================================\n")
        f.write(f"Total Audio Files Processed : {len(records)}\n")
        f.write(f"Total Audio Duration        : {total_audio_sec:.2f} seconds\n")
        f.write(f"Total Temporal Frames       : {total_frames} frames\n")
        f.write(f"Features per Frame          : {TOTAL_FEATURES} features\n")
        f.write(f"Total Extracted Data Points : {total_frames * TOTAL_FEATURES} values\n")
        f.write(f"Wall-Clock Execution Time   : {t_total_ms:.2f} ms\n\n")
        f.write(f"{'Filename':<30} {'Duration(s)':<12} {'Frames':<10} {'Processing(ms)':<15}\n")
        f.write("-" * 70 + "\n")
        for r in records:
            f.write(f"{r['file']:<30} {r['duration_sec']:<12.2f} {r['frames']:<10} {r['time_ms']:<15.2f}\n")
        f.write("================================================================================\n")

    print("\n--------------------------------------------------------------------------------")
    print(f"Batch execution complete: {len(records)} files, {total_frames} frames in {t_total_ms:.2f} ms")
    print(f"Summary written to: {summary_path}")
    print("================================================================================")


def main():
    parser = argparse.ArgumentParser(description="Run phase-based audio feature extraction.")
    parser.add_argument("--input", default=None, help="Input WAV file for single file execution")
    parser.add_argument("--batch", default=None, help="Directory of WAV files for batch execution")
    parser.add_argument("--output", default="output/features.csv", help="Output CSV file for single run")
    parser.add_argument("--output-dir", default="output", help="Output directory for batch run")
    args = parser.parse_args()

    if args.batch:
        process_batch(args.batch, args.output_dir)
    else:
        inp = args.input if args.input else "data/demo.wav"
        process_audio_file(inp, args.output)


if __name__ == "__main__":
    main()
