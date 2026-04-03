# -----------------------------------------------------------------------------
# Bead Analyzer
#
# (c) 2026 Arnas Technologies, LLC
# Developed by Nathan O'Connor, PhD, MS
#
# Licensed under the MIT License.
# For consulting or custom development: nathan@arnastech.com
# -----------------------------------------------------------------------------

#!/usr/bin/env python3
"""Generate 3-panel (XY, XZ, YZ) projection plot from a 3D TIFF stack."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description="Generate orthogonal projection plot from 3D TIFF.")
    parser.add_argument("input_file", type=str)
    parser.add_argument("--scale_xy", type=float, default=0.26)
    parser.add_argument("--scale_z", type=float, default=2.0)
    args = parser.parse_args()
    path = Path(args.input_file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1
    stack = tifffile.imread(path).astype(np.float32)
    if stack.ndim != 3:
        print(f"Expected 3D stack, got {stack.ndim}D")
        return 1
    min_v, max_v = stack.min(), stack.max()
    if max_v > min_v:
        stack = (stack - min_v) / (max_v - min_v)
    z_m, y_m, x_m = (d // 2 for d in stack.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(stack[z_m, :, :], cmap='hot', aspect=1)
    axes[0].set_title('XY')
    axes[1].imshow(stack[:, y_m, :], cmap='hot', aspect=args.scale_z / args.scale_xy)
    axes[1].set_title('XZ')
    axes[2].imshow(stack[:, :, x_m], cmap='hot', aspect=args.scale_z / args.scale_xy)
    axes[2].set_title('YZ')
    plt.tight_layout()
    out = path.with_name(f"{path.stem}_projections.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")
    return 0

if __name__ == "__main__":
    exit(main())
