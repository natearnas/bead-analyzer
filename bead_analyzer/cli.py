# -----------------------------------------------------------------------------
# Bead Analyzer
#
# (c) 2026 Arnas Technologies, LLC
# Developed by Nathan O'Connor, PhD, MS
#
# Licensed under the MIT License.
# For consulting or custom development: nathan@arnastech.com
# -----------------------------------------------------------------------------

"""
Command-line interface for FWHM bead analysis.
"""

import argparse
import json
import os
from pathlib import Path

import tifffile

from . import analysis


def get_cellpose_model_path(cli_path=None, env_var='FWHM_CELLPOSE_MODEL'):
    """Resolve Cellpose model path from CLI or environment."""
    if cli_path:
        return cli_path
    return os.environ.get(env_var)


def main():
    parser = argparse.ArgumentParser(
        description="Bead Analyzer: Analyze microsphere images for characterizing light microscopy systems.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_file", type=str, nargs='?', default=None,
                        help="Path to TIFF stack (can also be set via --config)")
    parser.add_argument("--config", type=str, default=None,
                        help="Load settings from JSON file (saved by GUI). "
                             "CLI arguments override config file values.")
    parser.add_argument("--mode", type=str, choices=['manual', 'blob', 'trackpy', 'stardist', 'cellpose'],
                        default='blob',
                        help="Detection mode (default: blob). "
                             "Use stardist/cellpose mainly for large beads (~15+ px diameter).")
    parser.add_argument("--scale_xy", type=float, required=True, help="XY pixel size (µm)")
    parser.add_argument("--scale_z", type=float, required=True, help="Z pixel size (µm)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--box_size", type=int, default=15)
    parser.add_argument("--center_mode", type=str, default="peak",
                        choices=["peak", "centroid", "radial"],
                        help="Center refinement mode: peak (default), centroid, or radial")
    parser.add_argument("--line_length", type=float, default=5.0)
    parser.add_argument("--z_smooth", type=float, default=None)
    parser.add_argument("--detrend", action="store_true")
    parser.add_argument("--subtract_background", action="store_true")
    parser.add_argument("--fit_gaussian", action="store_true")
    parser.add_argument("--fit_window", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--vmin_pct", type=float, default=None)
    parser.add_argument("--vmax_pct", type=float, default=None)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--upsample_factor", type=int, default=4)
    parser.add_argument("--na", type=float, default=None, help="Numerical aperture")
    parser.add_argument("--fluorophore", type=str, default=None, help="Fluorophore name(s)")
    parser.add_argument("--prominence_min", type=float, default=None,
                        help="Absolute prominence threshold (overrides relative if set)")
    parser.add_argument("--prominence_rel", type=float, default=0.1,
                        help="Relative prominence (fraction of signal range)")
    parser.add_argument("--review_detection", action="store_true",
                        help="Review detected points before analysis")
    parser.add_argument("--stardist_model", type=str, default="2D_versatile_fluo",
                        help="StarDist pretrained model name (recommended for larger beads, ~15+ px diameter)")
    parser.add_argument("--stardist_prob_thresh", type=float, default=0.6,
                        help="StarDist probability threshold")
    parser.add_argument("--stardist_nms_thresh", type=float, default=0.3,
                        help="StarDist NMS threshold")
    parser.add_argument("--use_blob_fallback", action="store_true",
                        help="Fallback to classical local-max blob detection if StarDist fails")
    parser.add_argument("--blob_sigma", type=float, default=1.2,
                        help="Blob detector Gaussian sigma (pixels)")
    parser.add_argument("--blob_threshold_rel", type=float, default=0.2,
                        help="Blob detector threshold as fraction of dynamic range")
    parser.add_argument("--blob_min_distance", type=int, default=5,
                        help="Blob detector minimum peak spacing (pixels)")
    parser.add_argument("--stardist_n_tiles", type=int, nargs=2, default=None,
                        help="StarDist tiling for large images, e.g. --stardist_n_tiles 4 4")
    parser.add_argument("--use_trackpy", action="store_true",
                        help="Use trackpy bandpass detector (handles intensity gradients)")
    parser.add_argument("--trackpy_diameter", type=int, default=5,
                        help="trackpy feature diameter in pixels (must be odd, default 5)")
    parser.add_argument("--trackpy_minmass", type=float, default=5000,
                        help="trackpy minimum integrated brightness (default 5000)")
    parser.add_argument("--trackpy_separation", type=int, default=None,
                        help="trackpy minimum separation between features (default: diameter+1)")
    parser.add_argument("--skip_cellpose_review", action="store_true",
                        help="Skip Cellpose detection review overlay")
    parser.add_argument("--qa_min_snr", type=float, default=3.0,
                        help="QA: minimum Z-profile SNR")
    parser.add_argument("--qa_min_symmetry", type=float, default=0.6,
                        help="QA: minimum Z-profile symmetry (0-1)")
    parser.add_argument("--qa_auto_reject", action="store_true",
                        help="Automatically reject beads failing QA thresholds")
    parser.add_argument("--fit_3d", action="store_true",
                        help="Fit 3D Gaussian to each bead (more accurate but slower)")
    parser.add_argument("--save_diagnostics", action="store_true",
                        help="Save per-bead diagnostic plots")
    parser.add_argument("--local_background", action="store_true",
                        help="Use local annulus background instead of global minimum")
    parser.add_argument("--robust_fit", action="store_true",
                        help="Use robust (soft-L1 / Huber-like) loss for Gaussian fitting")

    # Mode-specific
    parser.add_argument("--points_file", type=str, default=None)
    parser.add_argument("--smooth_xy", type=float, default=None)
    parser.add_argument("--max_z_fwhm", type=float, default=None)
    parser.add_argument("--cellpose_model", type=str, default=None,
                        help="Path to Cellpose model file (recommended for larger beads, ~15+ px diameter)")
    parser.add_argument("--cellpose_diameter", type=float, default=None)
    parser.add_argument("--detection_gauss_sigma", type=float, default=None)
    parser.add_argument("--cellpose_cellprob", type=float, default=0.0)
    parser.add_argument("--cellpose_min_size", type=int, default=3,
                        help="Minimum mask area in pixels (default 3; Cellpose default 15 is too large for beads)")
    parser.add_argument("--cellpose_flow_threshold", type=float, default=0.4,
                        help="Flow error threshold for Cellpose (default 0.4)")
    parser.add_argument("--cellpose_do_3d", action="store_true",
                        help="Run native Cellpose 3D segmentation on the full stack")
    parser.add_argument("--anisotropy", type=float, default=None,
                        help="Cellpose 3D anisotropy ratio (z_spacing/xy_spacing)")
    parser.add_argument("--num_beads_avg", type=int, default=20,
                        help="Beads for average (nearest to median Z-FWHM; minimum 1)")
    parser.add_argument("--sample_fraction", type=float, default=100,
                        help="Analyze this percentage of detected beads (1-100; default 100 = all)")
    parser.add_argument("--z_range", type=int, nargs=2, default=None)
    parser.add_argument("--z_analysis_margin", type=int, default=20)
    parser.add_argument("--reject_outliers", type=float, default=None)

    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            return 1
        with open(config_path) as f:
            cfg = json.load(f)
        if args.input_file is None and 'input_file' in cfg:
            args.input_file = cfg['input_file']
        arg_defaults = parser.parse_args([args.input_file or ''])
        for key, value in cfg.items():
            if key in ('input_file',):
                continue
            attr = key.replace('-', '_')
            if hasattr(args, attr):
                cli_val = getattr(args, attr)
                default_val = getattr(arg_defaults, attr, None)
                if cli_val == default_val:
                    setattr(args, attr, value)

    if args.input_file is None:
        parser.error("input_file is required (provide it directly or via --config)")

    args.num_beads_avg = max(1, args.num_beads_avg)

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1

    img = tifffile.imread(str(input_path))
    stack = img[args.channel] if img.ndim == 4 else img

    output_path = input_path
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name

    kwargs = {
        'channel': args.channel,
        'scale_xy': args.scale_xy,
        'scale_z': args.scale_z,
        'box_size': args.box_size,
        'center_mode': args.center_mode,
        'line_length': args.line_length,
        'z_smooth': args.z_smooth or (1.0 if args.mode == 'manual' else None),
        'detrend': args.detrend,
        'subtract_background': args.subtract_background,
        'fit_gaussian': args.fit_gaussian,
        'fit_window': args.fit_window,
        'gamma': args.gamma,
        'vmin_pct': args.vmin_pct,
        'vmax_pct': args.vmax_pct,
        'points_file': args.points_file,
        'na': args.na,
        'fluorophore': args.fluorophore,
        'prominence_min': args.prominence_min,
        'prominence_rel': args.prominence_rel,
        'qa_min_snr': args.qa_min_snr,
        'qa_min_symmetry': args.qa_min_symmetry,
        'fit_3d': args.fit_3d,
        'save_diagnostics': args.save_diagnostics,
        'qa_auto_reject': args.qa_auto_reject,
        'output_dir': str(output_path.parent),
        'local_background': args.local_background,
        'robust_fit': args.robust_fit,
        'sample_fraction': min(100, max(1, args.sample_fraction)),
    }

    rejected = []
    if args.mode == 'manual':
        kwargs['smooth_xy'] = args.smooth_xy
        results, bead_volumes, mip, profiles, rejected = analysis.run_manual(stack, **kwargs)
        bead_log = None
    elif args.mode == 'blob':
        kwargs['max_z_fwhm'] = args.max_z_fwhm
        kwargs['review_detection'] = args.review_detection
        kwargs['blob_sigma'] = args.blob_sigma
        kwargs['blob_threshold_rel'] = args.blob_threshold_rel
        kwargs['blob_min_distance'] = args.blob_min_distance
        results, bead_volumes, mip, profiles, rejected = analysis.run_blob(stack, **kwargs)
        bead_log = None
    elif args.mode == 'trackpy':
        kwargs['max_z_fwhm'] = args.max_z_fwhm
        kwargs['review_detection'] = args.review_detection
        kwargs['trackpy_diameter'] = args.trackpy_diameter
        kwargs['trackpy_minmass'] = args.trackpy_minmass
        kwargs['trackpy_separation'] = args.trackpy_separation
        results, bead_volumes, mip, profiles, rejected = analysis.run_trackpy(stack, **kwargs)
        bead_log = None
    elif args.mode == 'stardist':
        kwargs['max_z_fwhm'] = args.max_z_fwhm
        kwargs['review_detection'] = args.review_detection
        kwargs['stardist_model'] = args.stardist_model
        kwargs['stardist_prob_thresh'] = args.stardist_prob_thresh
        kwargs['stardist_nms_thresh'] = args.stardist_nms_thresh
        kwargs['use_blob_fallback'] = args.use_blob_fallback
        kwargs['blob_sigma'] = args.blob_sigma
        kwargs['blob_threshold_rel'] = args.blob_threshold_rel
        kwargs['blob_min_distance'] = args.blob_min_distance
        kwargs['stardist_n_tiles'] = tuple(args.stardist_n_tiles) if args.stardist_n_tiles else None
        kwargs['use_trackpy'] = args.use_trackpy
        kwargs['trackpy_diameter'] = args.trackpy_diameter
        kwargs['trackpy_minmass'] = args.trackpy_minmass
        kwargs['trackpy_separation'] = args.trackpy_separation
        results, bead_volumes, mip, profiles, rejected = analysis.run_stardist(stack, **kwargs)
        bead_log = None
    else:  # cellpose
        model_path = get_cellpose_model_path(args.cellpose_model)
        if not model_path or not Path(model_path).exists():
            print("ERROR: Cellpose model path required. Set --cellpose_model or FWHM_CELLPOSE_MODEL.")
            return 1
        kwargs.update({
            'model_path': model_path,
            'max_z_fwhm': args.max_z_fwhm,
            'cellpose_diameter': args.cellpose_diameter,
            'detection_gauss_sigma': args.detection_gauss_sigma,
            'cellpose_cellprob': args.cellpose_cellprob,
            'cellpose_min_size': args.cellpose_min_size,
            'cellpose_flow_threshold': args.cellpose_flow_threshold,
            'cellpose_do_3d': args.cellpose_do_3d,
            'anisotropy': args.anisotropy,
            'z_range': args.z_range,
            'z_analysis_margin': args.z_analysis_margin,
            'reject_outliers': args.reject_outliers,
            'num_beads_avg': args.num_beads_avg,
            'skip_review': args.skip_cellpose_review,
        })
        results, bead_volumes, mip, bead_log, profiles, rejected = analysis.run_cellpose(stack, **kwargs)

    analysis.write_outputs(
        results, bead_volumes, mip, output_path, args.mode,
        args.scale_xy, args.scale_z,
        upsample_factor=args.upsample_factor,
        no_plots=args.no_plots,
        num_beads_avg=args.num_beads_avg,
        bead_log=bead_log,
        na=args.na,
        fluorophore=args.fluorophore,
        gamma=args.gamma,
        qa_min_snr=args.qa_min_snr,
        qa_min_symmetry=args.qa_min_symmetry,
        rejected=rejected,
        profiles=profiles,
        center_mode=args.center_mode,
    )
    print("\nAnalysis complete.")
    return 0


if __name__ == "__main__":
    exit(main())
