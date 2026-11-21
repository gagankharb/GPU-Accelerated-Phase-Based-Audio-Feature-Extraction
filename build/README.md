# Build Artifacts Directory

This directory stores compiled binaries, intermediate object files (`.o`, `.obj`), and execution wrappers for the GPU-Accelerated Phase-Based Audio Feature Extraction project.

## Directory Contents
- `phase_audio_cuda` (Linux binary / wrapper)
- `phase_audio_cuda.bat` (Windows launcher)
- `phase_audio_cpu` (CPU reference binary / wrapper)
- `phase_audio_cpu.bat` (Windows CPU launcher)
- Object files: `audio_io.o`, `cpu_reference.o`, `gpu_processing.o`, `main.o` (generated during `make build`)

## Compilation Instructions

### On Linux / NVIDIA GPU Systems (Coursera GPU Lab)
```bash
# Compile native CUDA binary with cuFFT
make clean
make build CUDA_ARCH=sm_75

# Execute compiled binary
./build/phase_audio_cuda --input data/demo.wav --output output/features.csv --backend both
```

### On Windows
```cmd
REM Run automated pipeline
run.bat

REM Or invoke launcher directly
build\phase_audio_cuda.bat --input data\demo.wav --output output\features.csv --backend both
```
