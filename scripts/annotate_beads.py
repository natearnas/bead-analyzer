#!/usr/bin/env python3
"""Create Cellpose training masks by right-clicking on beads. Saves *_raw.tif and *_masks.tif."""

import argparse
from pathlib import Path
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood

def main():
    parser = argparse.ArgumentParser(description="Create 2D training masks for bead segmentation.")
    parser.add_argument("input_file", type=str, help="Path to the 3D TIFF file.")
    args = parser.parse_args()
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"ERROR: File not found: {input_path}")
        return 1
    image_stack = tifffile.imread(input_path)
    if image_stack.ndim == 4:
        image_stack = image_stack[0]
    if image_stack.ndim != 3:
        print(f"ERROR: Expected 3D stack, got {image_stack.ndim}D")
        return 1
    mip = np.max(image_stack, axis=0)
    points = []
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(mip, cmap='gray', vmax=np.percentile(mip, 99.8))
    ax.set_title("Right-click to add beads. 'd' to delete last. 'Esc' to save.")
    def on_click(event):
        if event.inaxes != ax or event.button != 3:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        points.append((x, y))
        ax.plot(x, y, 'r+', markersize=12, markeredgewidth=2)
        fig.canvas.draw()
    def on_key(event):
        if event.key == 'd' and points:
            points.pop()
            ax.clear()
            ax.imshow(mip, cmap='gray', vmax=np.percentile(mip, 99.8))
            ax.set_title("Right-click to add beads. 'd' to delete last. 'Esc' to save.")
            for px, py in points:
                ax.plot(px, py, 'r+', markersize=12, markeredgewidth=2)
            fig.canvas.draw()
        elif event.key == 'escape':
            plt.close(fig)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    if not points:
        print("No beads annotated.")
        return 0
    final_mask = np.zeros_like(mip, dtype=np.uint16)
    for i, (x, y) in enumerate(points):
        z_profile = image_stack[:, y, x]
        best_z = np.argmax(z_profile)
        peak_intensity = z_profile[best_z]
        best_slice = image_stack[best_z, :, :]
        bead_mask = flood(best_slice, (y, x), tolerance=peak_intensity * 0.5)
        final_mask[bead_mask] = i + 1
    base = input_path.stem
    raw_path = input_path.parent / f"{base}_raw.tif"
    mask_path = input_path.parent / f"{base}_masks.tif"
    tifffile.imwrite(raw_path, mip)
    tifffile.imwrite(mask_path, final_mask, imagej=True)
    print(f"Saved {raw_path} and {mask_path}")
    return 0

if __name__ == "__main__":
    exit(main())
