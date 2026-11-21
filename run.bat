@echo off
REM ==============================================================================
REM run.bat - Complete Pipeline Execution & Verification Script (Windows)
REM Coursera GPU Specialization Capstone Project
REM ==============================================================================

echo ================================================================================
echo GPU-ACCELERATED PHASE-BASED AUDIO FEATURE EXTRACTION (CUDA ^& cuFFT)
echo Coursera GPU Programming Specialization Capstone Execution Script
echo ================================================================================

if not exist build mkdir build
if not exist data mkdir data
if not exist output mkdir output

echo.
echo [Step 1/5] Generating Synthetic Audio Datasets (Single ^& Batch)...
python scripts/generate_demo_wav.py --output data/demo.wav --duration 2.5
python scripts/generate_demo_wav.py --output data/sample_01_voiced.wav --duration 1.5
python scripts/generate_demo_wav.py --output data/sample_02_chirp.wav --duration 2.0
python scripts/generate_demo_wav.py --output data/sample_03_stereo.wav --duration 2.0 --channels 2

echo.
echo [Step 2/5] Running Unit Test Suite...
python scripts/test_pipeline.py

echo.
echo [Step 3/5] Executing Audio Feature Extraction Pipeline...
python scripts/run_pipeline.py --input data/demo.wav --output output/features.csv
python scripts/run_pipeline.py --batch data/ --output-dir output

echo.
echo [Step 4/5] Generating Visualizations and Heatmaps...
python scripts/inspect_features.py --input output/features.csv --output-dir output

echo.
echo [Step 5/5] Reviewing Output Artifacts:
dir output\

echo ================================================================================
echo EXECUTION FINISHED SUCCESSFULLY
echo ================================================================================
