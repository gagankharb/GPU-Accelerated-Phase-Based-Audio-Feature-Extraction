# ==============================================================================
# Makefile: GPU-Accelerated Phase-Based Audio Feature Extraction (CUDA & cuFFT)
# Coursera GPU Programming Specialization Capstone Project
# ==============================================================================

# Compiler Configuration
NVCC       ?= nvcc
CXX        ?= g++
PYTHON     ?= python

# Detect Operating System
ifeq ($(OS),Windows_NT)
    TARGET_EXT := .exe
    RM_CMD     := del /Q /F
    RM_DIR     := rmdir /S /Q
    MKDIR_CMD  := if not exist build mkdir build
else
    TARGET_EXT :=
    RM_CMD     := rm -f
    RM_DIR     := rm -rf
    MKDIR_CMD  := mkdir -p build
endif

TARGET     := build/phase_audio_cuda$(TARGET_EXT)
CPU_TARGET := build/phase_audio_cpu$(TARGET_EXT)

# Target GPU Compute Architecture Configuration
# Users can override this by passing: make CUDA_ARCH=sm_80
# Common values:
#   sm_60 (Pascal - P100)
#   sm_70 (Volta - V100)
#   sm_75 (Turing - T4, RTX 2000 series)
#   sm_80 (Ampere - A100)
#   sm_86 (Ampere - RTX 3000 series)
#   sm_89 (Ada Lovelace - RTX 4000 series)
#   sm_90 (Hopper - H100)
CUDA_ARCH ?= sm_75

# Compiler Flags
CXXFLAGS   := -std=c++17 -O3 -Wall
NVCCFLAGS  := -std=c++17 -O3 -arch=$(CUDA_ARCH) --compiler-options -Wall -lineinfo
INCLUDES   := -Isrc
LDFLAGS    := -lcufft

# Source Files
CU_SRCS    := src/gpu_processing.cu src/main.cu
CPP_SRCS   := src/audio_io.cpp src/cpu_reference.cpp
HEADERS    := src/common.h src/feature_extraction.h src/audio_io.h src/cpu_reference.h src/gpu_processing.h

# Object Files
BUILD_DIR  := build
OBJS       := $(BUILD_DIR)/audio_io.o \
              $(BUILD_DIR)/cpu_reference.o \
              $(BUILD_DIR)/gpu_processing.o \
              $(BUILD_DIR)/main.o

# ==============================================================================
# Build Targets
# ==============================================================================

.PHONY: all build run test clean cpu_build help demo_data visualize

# Default target
all: build

# Alias for build
build: $(TARGET)

help:
	@echo "================================================================================"
	@echo "CUDA Phase Audio Capstone Build System"
	@echo "================================================================================"
	@echo "Available Targets:"
	@echo "  make              - Compile complete CUDA + cuFFT application (default)"
	@echo "  make build        - Compile complete CUDA + cuFFT application"
	@echo "  make run          - Generate sample audio and run with backend=both"
	@echo "  make test         - Run end-to-end verification and numerical unit tests"
	@echo "  make cpu_build    - Compile standalone CPU reference without NVCC"
	@echo "  make demo_data    - Generate synthetic 16 kHz multi-harmonic test WAV"
	@echo "  make visualize    - Generate heatmaps and feature plots from CSV output"
	@echo "  make clean        - Remove compiled binaries and build artifacts"
	@echo ""
	@echo "Custom Architecture Selection:"
	@echo "  make CUDA_ARCH=sm_80   (For NVIDIA A100 / RTX 3000 series)"
	@echo "  make CUDA_ARCH=sm_75   (For NVIDIA T4 / RTX 2000 series)"
	@echo "  make CUDA_ARCH=sm_70   (For NVIDIA V100)"
	@echo "================================================================================"

# Compile C++ Object Files
$(BUILD_DIR)/%.o: src/%.cpp $(HEADERS)
	@$(MKDIR_CMD)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA Object Files
$(BUILD_DIR)/%.o: src/%.cu $(HEADERS)
	@$(MKDIR_CMD)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Link Binary
$(TARGET): $(OBJS)
	@$(MKDIR_CMD)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)
	@echo "Successfully built target: $(TARGET)"

# Build CPU-only standalone binary (for systems without NVCC)
cpu_build: $(CPP_SRCS) src/main.cu $(HEADERS)
	@$(MKDIR_CMD)
	$(CXX) $(CXXFLAGS) -DCPU_ONLY_MODE $(INCLUDES) src/audio_io.cpp src/cpu_reference.cpp src/main.cu -o $(CPU_TARGET)
	@echo "Successfully built standalone CPU binary: $(CPU_TARGET)"

# Generate Demo WAV audio
demo_data:
	$(PYTHON) scripts/generate_demo_wav.py --output data/demo.wav --duration 2.5

# Run the pipeline
run: $(TARGET) demo_data
	./$(TARGET) --input data/demo.wav --output output/features_both.csv --backend both

# Run automated tests and validation
test: demo_data
	$(PYTHON) scripts/test_pipeline.py

# Generate visualization plots
visualize:
	$(PYTHON) scripts/inspect_features.py --input output/features.csv --output-dir output

# Clean build artifacts
clean:
ifeq ($(OS),Windows_NT)
	-$(RM_CMD) build\*.o build\*.exe 2>nul || exit 0
	-$(RM_DIR) build 2>nul || exit 0
else
	$(RM_CMD) $(OBJS) $(TARGET) $(CPU_TARGET)
	$(RM_DIR) $(BUILD_DIR)
endif
	@echo "Build artifacts successfully cleaned."
