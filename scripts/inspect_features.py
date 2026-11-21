#!/usr/bin/env python3
"""
inspect_features.py
-------------------
Visualizes and analyzes extracted phase-based audio features from CSV.
Generates:
1. Feature Heatmap across time frames (60 features: static, delta, delta-delta)
2. Time-series trajectories for selected static, delta, and delta-delta coefficients
3. RMS Energy profile and gating demonstration
4. Comprehensive statistical summary table printed and saved to text
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns


def inspect_and_plot(csv_path, output_dir="output"):
    if not os.path.exists(csv_path):
        print(f"[Error] CSV feature file not found: {csv_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    print("================================================================================")
    print("FEATURE EXTRACTION INSPECTION & VISUALIZATION")
    print("================================================================================")
    print(f"Loaded features from: {csv_path}")
    print(f"Total Audio Frames  : {len(df)}")
    print(f"Total Columns       : {len(df.columns)}")
    print(f"Total Features/Frame: 60 (20 static + 20 delta + 20 delta-delta)")
    print("--------------------------------------------------------------------------------")

    feature_cols = [c for c in df.columns if c not in ["frame_index", "time_seconds", "rms_energy"]]
    if len(feature_cols) != 60:
        print(f"[Warning] Expected 60 feature columns, found {len(feature_cols)}")

    # -------------------------------------------------------------------------
    # 1. Feature Heatmap Across Time
    # -------------------------------------------------------------------------
    plt.figure(figsize=(14, 8), dpi=300)
    feat_matrix = df[feature_cols].values.T # Shape: (60, num_frames)

    vmax = np.percentile(np.abs(feat_matrix), 98)
    vmin = -vmax

    ax = sns.heatmap(
        feat_matrix,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Coefficient Amplitude"}
    )

    # Annotate feature blocks
    ax.axhline(20, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(40, color="black", linestyle="--", linewidth=1.5)

    ax.text(len(df) * 1.02, 10, "Static Features\n(10 Cos + 10 Sin)", va="center", fontsize=10, weight="bold")
    ax.text(len(df) * 1.02, 30, "Delta Features\n(20 Derivatives)", va="center", fontsize=10, weight="bold")
    ax.text(len(df) * 1.02, 50, "Delta-Delta Features\n(20 2nd-Deriv)", va="center", fontsize=10, weight="bold")

    # Time ticks
    num_ticks = 8
    tick_indices = np.linspace(0, len(df) - 1, num_ticks, dtype=int)
    tick_labels = [f"{df['time_seconds'].iloc[i]:.2f}s" for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=0)

    ax.set_yticks([5, 15, 25, 35, 45, 55])
    ax.set_yticklabels(["Cos 1-10", "Sin 1-10", "Δ-Cos 1-10", "Δ-Sin 1-10", "Δ²-Cos 1-10", "Δ²-Sin 1-10"])

    plt.title("GPU-Accelerated Phase Cepstral Feature Heatmap Across Time", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Time (seconds / frame index)", fontsize=11, labelpad=8)
    plt.ylabel("Phase Cepstral Feature Dimensions", fontsize=11, labelpad=8)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, "feature_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Generated] Feature Heatmap saved to: {heatmap_path}")

    # -------------------------------------------------------------------------
    # 2. Time-Series Dynamics: Static, Delta, and Delta-Delta Trajectories
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, dpi=300)
    time = df["time_seconds"].values

    # Subplot 1: RMS Energy Profile
    axes[0].plot(time, df["rms_energy"], color="#2ca02c", linewidth=2.0, label="Frame RMS Energy")
    axes[0].axhline(1e-4, color="red", linestyle=":", linewidth=1.5, label="Silence Gate (1e-4)")
    axes[0].set_ylabel("RMS Energy", fontsize=10, weight="bold")
    axes[0].set_title("Audio Temporal Trajectory & Feature Evolution", fontsize=13, weight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    # Subplot 2: Selected Static Features (Cos 1, Cos 2, Sin 1, Sin 2)
    axes[1].plot(time, df["cos_cepstral_1"], label="Static Cos 1", color="#1f77b4", linewidth=1.8)
    axes[1].plot(time, df["cos_cepstral_2"], label="Static Cos 2", color="#aec7e8", linewidth=1.5)
    axes[1].plot(time, df["sin_cepstral_1"], label="Static Sin 1", color="#ff7f0e", linewidth=1.8)
    axes[1].plot(time, df["sin_cepstral_2"], label="Static Sin 2", color="#ffbb78", linewidth=1.5)
    axes[1].set_ylabel("Static Coeffs", fontsize=10, weight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)

    # Subplot 3: Corresponding Delta Features
    axes[2].plot(time, df["delta_cos_1"], label="Δ-Cos 1", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[2].plot(time, df["delta_cos_2"], label="Δ-Cos 2", color="#aec7e8", linestyle="--", linewidth=1.5)
    axes[2].plot(time, df["delta_sin_1"], label="Δ-Sin 1", color="#ff7f0e", linestyle="--", linewidth=1.8)
    axes[2].plot(time, df["delta_sin_2"], label="Δ-Sin 2", color="#ffbb78", linestyle="--", linewidth=1.5)
    axes[2].set_ylabel("Delta Features", fontsize=10, weight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right", frameon=True)

    # Subplot 4: Corresponding Delta-Delta Features
    axes[3].plot(time, df["ddelta_cos_1"], label="Δ²-Cos 1", color="#1f77b4", linestyle=":", linewidth=1.8)
    axes[3].plot(time, df["ddelta_cos_2"], label="Δ²-Cos 2", color="#aec7e8", linestyle=":", linewidth=1.5)
    axes[3].plot(time, df["ddelta_sin_1"], label="Δ²-Sin 1", color="#ff7f0e", linestyle=":", linewidth=1.8)
    axes[3].plot(time, df["ddelta_sin_2"], label="Δ²-Sin 2", color="#ffbb78", linestyle=":", linewidth=1.5)
    axes[3].set_ylabel("Delta-Delta", fontsize=10, weight="bold")
    axes[3].set_xlabel("Time (seconds)", fontsize=11, weight="bold")
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    variation_path = os.path.join(output_dir, "feature_variation_time.png")
    plt.savefig(variation_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Generated] Feature Dynamics saved to: {variation_path}")

    # -------------------------------------------------------------------------
    # 3. Statistical Summary Computation & Export
    # -------------------------------------------------------------------------
    stats = df[feature_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    stats.columns = ["Mean", "StdDev", "Min", "Median", "Max"]

    summary_text_path = os.path.join(output_dir, "statistical_summary.txt")
    with open(summary_text_path, "w") as f:
        f.write("================================================================================\n")
        f.write("PHASE CEPSTRAL FEATURE EXTRACTION STATISTICAL SUMMARY\n")
        f.write("================================================================================\n")
        f.write(f"Source CSV       : {csv_path}\n")
        f.write(f"Total Frames     : {len(df)}\n")
        f.write(f"Total Duration   : {df['time_seconds'].max():.3f} seconds\n")
        f.write(f"Mean Frame RMS   : {df['rms_energy'].mean():.6f}\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(stats.to_string())
        f.write("\n================================================================================\n")

    print(f"[Report Generated] Statistical summary saved to: {summary_text_path}")
    print("\nPreview of Feature Statistics (First 10 Features):")
    print(stats.head(10).to_string())
    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect and visualize phase audio features.")
    parser.add_argument("--input", default="output/features.csv", help="Path to input features CSV.")
    parser.add_argument("--output-dir", default="output", help="Directory to save generated plots.")
    args = parser.parse_args()

    inspect_and_plot(args.input, args.output_dir)


if __name__ == "__main__":
    main()
