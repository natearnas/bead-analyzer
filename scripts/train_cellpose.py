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
"""Train a custom Cellpose model on bead images. Expects *_raw.tif and *_masks.tif pairs."""

import argparse
import sys
from pathlib import Path

from cellpose import io, models, train


def main():
    parser = argparse.ArgumentParser(description="Train a custom Cellpose model on bead images.")
    parser.add_argument("data_directory", type=str,
                        help="Directory with *_raw.tif and *_masks.tif pairs.")
    args = parser.parse_args()
    data_dir = Path(args.data_directory)
    model_name = "cellpose_bead_model"
    print("Searching for training data...")
    output = io.load_images_labels(str(data_dir), image_filter="_raw")
    train_images, train_labels = output[:2]
    if not train_images:
        print(f"ERROR: No *_raw.tif found in {data_dir}")
        return 1
    if not train_labels:
        print("ERROR: No *_masks.tif found. Each *_raw.tif needs a *_masks.tif.")
        return 1
    print(f"Found {len(train_images)} image/mask pairs.")
    print("Initializing from 'nuclei' weights...")
    model = models.CellposeModel(gpu=True, model_type='nuclei')
    print("Training for 100 epochs...")
    new_path = train.train_seg(
        model.net, train_data=train_images, train_labels=train_labels,
        test_data=None, test_labels=None, n_epochs=100, learning_rate=1e-5,
        weight_decay=0.1, model_name=model_name, save_path=str(data_dir),
    )[0]
    print(f"Model saved at: {new_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
