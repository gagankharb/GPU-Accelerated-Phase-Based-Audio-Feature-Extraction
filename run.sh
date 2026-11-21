#!/usr/bin/env bash
# ==============================================================================
# run.sh - Complete Pipeline Execution & Verification Script
# Coursera GPU Specialization Capstone Project
# ==============================================================================

set -e

echo "================================================================================"
echo "GPU-ACCELERATED PHASE-BASED AUDIO FEATURE EXTRACTION (CUDA & cuFFT)"
echo "Coursera GPU Programming Specialization Capstone Execution Script"
echo "================================================================================"

# Step 1: Detect Host Environment & Compilers
echo ""
echo "[Step 1/6] Inspecting Host Execution Environment..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
else
    echo "Note: No NVIDIA GPU detected via nvidia-smi. CPU fallback will be tested."
fi

if command -v nvcc &> /dev/null; then
    echo "NVCC Compiler Version: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
    HAS_NVCC=1
else
    echo "Note: nvcc is not installed on this system."
    HAS_NVCC=0
fi

# Step 2: Build Project
echo ""
echo "[Step 2/6] Compiling Capstone Application..."
mkdir -p build output data

if [ "$HAS_NVCC" -eq 1 ]; then
    echo "Compiling CUDA + cuFFT target using Makefile..."
    make clean
    make build CUDA_ARCH=${CUDA_ARCH:-sm_75}
else
    echo "NVCC not found. Compiling standalone CPU target using standard C++17 compiler..."
    make cpu_build || echo "Host lacks 64-bit C++ compiler; falling back to Python test harness."
fi

# Step 3: Generate Synthetic Audio Dataset (Single & Batch)
echo ""
echo "[Step 3/6] Generating Test Audio Files..."
python scripts/generate_demo_wav.py --output data/demo.wav --duration 2.5
python scripts/generate_demo_wav.py --output data/sample_01_voiced.wav --duration 1.5
python scripts/generate_demo_wav.py --output data/sample_02_chirp.wav --duration 2.0
python scripts/generate_demo_wav.py --output data/sample_03_stereo.wav --duration 2.0 --channels 2

# Step 4: Run Automated Unit Tests & Validation
echo ""
echo "[Step 4/6] Running Validation & Unit Test Suite..."
python scripts/test_pipeline.py

# Step 5: Execute Feature Extraction (Single & Batch)
echo ""
echo "[Step 5/6] Executing Phase Feature Extraction Pipeline..."
if [ -f "./build/phase_audio_cuda" ]; then
    echo "Running CUDA Binary (Single File):"
    ./build/phase_audio_cuda --input data/demo.wav --output output/features.csv --backend both
    echo "Running CUDA Binary (Batch Mode across all WAV files):"
    ./build/phase_audio_cuda --batch data/ --backend both
elif [ -f "./build/phase_audio_cpu" ]; then
    echo "Running CPU Binary (Single File):"
    ./build/phase_audio_cpu --input data/demo.wav --output output/features.csv --backend cpu
    echo "Running CPU Binary (Batch Mode across all WAV files):"
    ./build/phase_audio_cpu --batch data/ --backend cpu
else
    echo "Running Python Verification Runner across audio datasets..."
    python scripts/run_pipeline.py --input data/demo.wav --output output/features.csv
    python scripts/run_pipeline.py --batch data/ --output-dir output
fi

# Step 6: Generate Visualizations & Statistical Reports
echo ""
echo "[Step 6/6] Generating Visual Inspection Heatmaps & Reports..."
python scripts/inspect_features.py --input output/features.csv --output-dir output

echo ""
echo "================================================================================"
echo "CAPSTONE EXECUTION COMPLETE [SUCCESS]"
echo "Artifacts generated in output/:"
ls -lh output/
echo "================================================================================"
