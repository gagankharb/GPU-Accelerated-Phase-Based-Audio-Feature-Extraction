#!/usr/bin/env python3
"""
generate_demo_wav.py
--------------------
Generates high-quality synthetic 16 kHz 16-bit PCM WAV audio files designed
specifically for validating phase-based audio feature extraction.

Includes:
- Section 1: Harmonic voiced vowel-like sound with distinct partials (F0=220 Hz)
- Section 2: Low-energy silence interval to validate RMS gating
- Section 3: Non-stationary frequency sweep (chirp) testing dynamic phase variation
- Supports generating both mono and multichannel audio for conversion testing
"""

import argparse
import os
import struct
import math
import wave
import numpy as np


def generate_audio_signal(sample_rate=16000, duration=2.5):
    num_samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, num_samples, endpoint=False)
    signal = np.zeros(num_samples, dtype=np.float32)

    # Section 1: Harmonic vowel-like complex tone (0.0s to 1.0s)
    mask_voiced = (t < 1.0)
    t_v = t[mask_voiced]
    f0 = 220.0
    signal[mask_voiced] = (
        0.40 * np.sin(2.0 * np.pi * f0 * t_v) +
        0.25 * np.sin(2.0 * np.pi * 2.0 * f0 * t_v + 0.3) +
        0.18 * np.sin(2.0 * np.pi * 3.0 * f0 * t_v + 0.7) +
        0.12 * np.sin(2.0 * np.pi * 4.0 * f0 * t_v + 1.2) +
        0.05 * np.sin(2.0 * np.pi * 5.0 * f0 * t_v + 1.9)
    )

    # Section 2: Low-energy silence interval for RMS gating (1.0s to 1.4s)
    mask_silence = (t >= 1.0) & (t < 1.4)
    t_s = t[mask_silence]
    # Subtle -80 dB noise floor well below default 1e-4 gate
    signal[mask_silence] = 1e-5 * np.sin(2.0 * np.pi * 50.0 * t_s)

    # Section 3: Non-stationary frequency chirp (1.4s to duration)
    mask_chirp = (t >= 1.4)
    t_c = t[mask_chirp] - 1.4
    chirp_dur = duration - 1.4
    f_start = 250.0
    f_end = 2800.0
    # Linear frequency chirp
    freq_inst = f_start + (f_end - f_start) * (t_c / chirp_dur)
    signal[mask_chirp] = (
        0.50 * np.sin(2.0 * np.pi * freq_inst * t_c) +
        0.20 * np.sin(2.0 * np.pi * (freq_inst * 1.5) * t_c + 0.5)
    )

    # Gentle onset and offset envelopes to avoid transient click artifacts
    fade_len = int(sample_rate * 0.01) # 10 ms fade
    signal[:fade_len] *= np.linspace(0.0, 1.0, fade_len)
    signal[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

    # Peak normalization to [-0.95, 0.95]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.95

    return signal, sample_rate


def save_wav_file(filename, samples, sample_rate=16000, num_channels=1):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Scale to signed 16-bit integers
    int16_samples = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    
    if num_channels == 1:
        payload = int16_samples.tobytes()
    else:
        # Replicate mono signal across multiple channels
        multichannel = np.column_stack([int16_samples] * num_channels)
        payload = multichannel.tobytes()

    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2) # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(payload)

    print(f"[generate_demo_wav] Created '{filename}': {len(samples)} samples, "
          f"{len(samples)/sample_rate:.2f}s, {sample_rate} Hz, {num_channels}ch, 16-bit PCM.")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 16 kHz 16-bit PCM WAV audio.")
    parser.add_argument("--output", default="data/demo.wav", help="Path to output WAV file.")
    parser.add_argument("--duration", type=float, default=2.5, help="Duration in seconds (default: 2.5).")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels (default: 1).")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate in Hz (default: 16000).")
    args = parser.parse_args()

    samples, sr = generate_audio_signal(sample_rate=args.samplerate, duration=args.duration)
    save_wav_file(args.output, samples, sample_rate=sr, num_channels=args.channels)


if __name__ == "__main__":
    main()
