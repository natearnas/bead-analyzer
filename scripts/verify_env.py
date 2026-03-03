#!/usr/bin/env python3
"""Verify environment: PyTorch, CUDA, NumPy, SciPy, scikit-image."""

def main():
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("PyTorch: not installed")
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: not installed")
    try:
        import scipy
        print(f"SciPy: {scipy.__version__}")
    except ImportError:
        print("SciPy: not installed")
    try:
        import skimage
        print(f"scikit-image: {skimage.__version__}")
    except ImportError:
        print("scikit-image: not installed")
    return 0

if __name__ == "__main__":
    exit(main())
