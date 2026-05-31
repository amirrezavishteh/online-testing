#!/usr/bin/env python3
"""
Check CUDA and GPU availability for training.

Usage:
    python scripts/check_cuda.py
"""

import sys
import subprocess


def check_nvcc():
    """Check if NVIDIA CUDA Compiler (nvcc) is available."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA Compiler (nvcc) found:")
            print(result.stdout)
            return True
        return False
    except FileNotFoundError:
        return False


def check_nvidia_smi():
    """Check if nvidia-smi is available and GPUs are present."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi found:")
            print(result.stdout)
            return True
        return False
    except FileNotFoundError:
        return False


def check_pytorch():
    """Check PyTorch CUDA support."""
    try:
        import torch

        print("\n" + "="*70)
        print("PYTORCH CONFIGURATION")
        print("="*70)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"         Memory: {gpu_mem:.1f} GB")
        else:
            print("⚠ CUDA is NOT available!")
            print("PyTorch was likely compiled without CUDA support.")

        return torch.cuda.is_available()
    except Exception as e:
        print(f"Error checking PyTorch: {e}")
        return False


def check_bitsandbytes():
    """Check BitsAndBytes CUDA support."""
    try:
        import bitsandbytes as bnb

        print("\n" + "="*70)
        print("BITSANDBYTES CONFIGURATION")
        print("="*70)
        print(f"BitsAndBytes version: {bnb.__version__}")

        # Try to import CUDA extension
        try:
            from bitsandbytes.cextension import COMPILED_WITH_CUDA
            print(f"Compiled with CUDA: {COMPILED_WITH_CUDA}")
            return COMPILED_WITH_CUDA
        except ImportError:
            print("⚠ Could not check CUDA compilation status")
            return False
    except ImportError:
        print("⚠ BitsAndBytes not installed")
        return False


def main():
    print("="*70)
    print("CUDA AVAILABILITY CHECK")
    print("="*70 + "\n")

    # Check system-level CUDA
    print("1. Checking system CUDA tools...")
    has_nvcc = check_nvcc()
    has_nvidia_smi = check_nvidia_smi()

    if not has_nvidia_smi:
        print("✗ nvidia-smi NOT found - NVIDIA drivers may not be installed")

    # Check PyTorch
    print("\n2. Checking PyTorch CUDA support...")
    has_pytorch_cuda = check_pytorch()

    # Check BitsAndBytes
    print("\n3. Checking BitsAndBytes CUDA support...")
    has_bnb_cuda = check_bitsandbytes()

    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)

    if has_pytorch_cuda and has_bnb_cuda:
        print("✓ Everything looks good! GPU training is available.")
        print("  You can use 4-bit quantization and paged_adamw_8bit optimizer.")
    elif has_pytorch_cuda:
        print("⚠ PyTorch has CUDA, but BitsAndBytes doesn't:")
        print("  - Option 1: Reinstall BitsAndBytes from source (preferred)")
        print("    pip install -i https://pypi.org/simple/ --no-deps --force-reinstall bitsandbytes")
        print("  - Option 2: Use FP16 training instead (no 4-bit quantization)")
    else:
        print("✗ CUDA is not available:")
        print("\nTO FIX (on remote server with GPU):")
        print("  1. Check if GPU is available: nvidia-smi")
        print("  2. Reinstall PyTorch with CUDA support:")
        print("     conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        print("  3. Reinstall BitsAndBytes:")
        print("     pip install -i https://pypi.org/simple/ --no-deps --force-reinstall bitsandbytes")
        print("\nOR use CPU-friendly training (no 4-bit, slower):")
        print("  python -m online.lab.poison --epochs 4 --poison-rate 0.1")

    print("="*70 + "\n")

    return 0 if has_pytorch_cuda else 1


if __name__ == "__main__":
    sys.exit(main())
