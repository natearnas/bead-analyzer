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
CustomTkinter GUI for FWHM bead analysis.
"""

import json
import threading
from pathlib import Path

try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
except ImportError:
    ctk = None


def _run_analysis(input_file, output_dir, mode, scale_xy, scale_z, na, fluorophore,
                  box_size, fit_gaussian, subtract_background, status_callback,
                  cellpose_model_path=None, channel=0, review_detection=False,
                  skip_cellpose_review=False, qa_min_snr=3.0, qa_min_symmetry=0.6,
                  fit_3d=False, save_diagnostics=False, qa_auto_reject=False,
                  cellpose_do_3d=False, anisotropy=None, use_blob_fallback=False,
                  local_background=False, robust_fit=False,
                  cellpose_min_size=3, cellpose_flow_threshold=0.4,
                  num_beads_avg=20):
    """Run analysis in background thread."""
    import tifffile
    from . import analysis

    try:
        status_callback("Loading image...")
        img = tifffile.imread(str(input_file))
        stack = img[channel] if img.ndim == 4 else img

        output_path = Path(output_dir) / Path(input_file).name if output_dir else Path(input_file)
        kwargs = {
            'scale_xy': scale_xy, 'scale_z': scale_z, 'box_size': box_size,
            'fit_gaussian': fit_gaussian, 'subtract_background': subtract_background,
            'na': na, 'fluorophore': fluorophore,
            'channel': channel,
            'qa_min_snr': qa_min_snr,
            'qa_min_symmetry': qa_min_symmetry,
            'fit_3d': fit_3d,
            'save_diagnostics': save_diagnostics,
            'qa_auto_reject': qa_auto_reject,
            'output_dir': str(output_dir) if output_dir else str(Path(input_file).parent),
            'cellpose_do_3d': cellpose_do_3d,
            'anisotropy': anisotropy,
            'use_blob_fallback': use_blob_fallback,
            'local_background': local_background,
            'robust_fit': robust_fit,
        }
        rejected = []
        if mode == 'manual':
            status_callback("Manual mode: select beads...")
            results, bead_volumes, mip, profiles, rejected = analysis.run_manual(stack, **kwargs)
            bead_log = None
        elif mode == 'blob':
            status_callback("Blob detection...")
            kwargs['review_detection'] = review_detection
            results, bead_volumes, mip, profiles, rejected = analysis.run_blob(stack, **kwargs)
            bead_log = None
        elif mode == 'trackpy':
            status_callback("Trackpy detection...")
            kwargs['review_detection'] = review_detection
            results, bead_volumes, mip, profiles, rejected = analysis.run_trackpy(stack, **kwargs)
            bead_log = None
        elif mode == 'stardist':
            status_callback("StarDist detection...")
            kwargs['review_detection'] = review_detection
            results, bead_volumes, mip, profiles, rejected = analysis.run_stardist(stack, **kwargs)
            bead_log = None
        else:
            import os
            model_path = cellpose_model_path or os.environ.get('FWHM_CELLPOSE_MODEL')
            if not model_path or not Path(model_path).exists():
                status_callback("ERROR: Select Cellpose model file or set FWHM_CELLPOSE_MODEL")
                return
            kwargs['model_path'] = model_path
            kwargs['skip_review'] = skip_cellpose_review
            kwargs['cellpose_min_size'] = cellpose_min_size
            kwargs['cellpose_flow_threshold'] = cellpose_flow_threshold
            status_callback("Cellpose detection...")
            results, bead_volumes, mip, bead_log, profiles, rejected = analysis.run_cellpose(stack, **kwargs)

        status_callback("Writing outputs...")
        analysis.write_outputs(
            results, bead_volumes, mip, output_path, mode,
            scale_xy, scale_z, bead_log=bead_log, na=na, fluorophore=fluorophore,
            qa_min_snr=qa_min_snr, qa_min_symmetry=qa_min_symmetry, rejected=rejected,
            profiles=profiles,
            num_beads_avg=num_beads_avg,
        )
        rej_msg = f" ({len(rejected)} rejected)" if rejected else ""
        status_callback(f"Done. {len(results)} beads analyzed{rej_msg}.")
    except Exception as e:
        status_callback(f"Error: {e}")


def main():
    if ctk is None:
        print("Install customtkinter: pip install customtkinter")
        return 1

    from . import __version__

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = ctk.CTk()
    app.title(f"Bead Analyzer v{__version__}")
    # Height is set dynamically in _toggle_advanced_options; start compact
    HEIGHT_COMPACT = 910
    HEIGHT_EXPANDED = 1220
    app.geometry(f"600x{HEIGHT_COMPACT}")
    app.minsize(550, HEIGHT_COMPACT)

    # Persistent settings path (user home directory)
    _settings_file = Path.home() / '.bead_analyzer_last_settings.json'

    def _load_last_settings():
        """Load last-used settings from disk, returning a dict (empty if none)."""
        try:
            if _settings_file.exists():
                with open(_settings_file) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    prev = _load_last_settings()

    # Variables -- restore from last session when available
    prev_fit_mode = prev.get('fit_mode')
    if prev_fit_mode not in ('1d', '3d', 'both', 'none'):
        # Backward-compatible mapping from older boolean settings.
        old_1d = bool(prev.get('fit_gaussian', True))
        old_3d = bool(prev.get('fit_3d', False))
        if old_1d and old_3d:
            prev_fit_mode = 'both'
        elif old_3d:
            prev_fit_mode = '3d'
        elif old_1d:
            prev_fit_mode = '1d'
        else:
            prev_fit_mode = 'none'

    # Advanced options toggle
    show_advanced = ctk.BooleanVar(value=prev.get('show_advanced', False))

    input_file = ctk.StringVar(value=prev.get('input_file', ''))
    output_dir = ctk.StringVar(value=prev.get('output_dir', ''))
    scale_xy = ctk.StringVar(value=str(prev.get('scale_xy', '0.26')))
    scale_z = ctk.StringVar(value=str(prev.get('scale_z', '2.0')))
    channel_var = ctk.StringVar(value=str(prev.get('channel', '0')))
    na_var = ctk.StringVar(value=str(prev.get('na', '')) if prev.get('na') is not None else '')
    fluorophore_var = ctk.StringVar(value=prev.get('fluorophore', '') or '')
    mode_var = ctk.StringVar(value=prev.get('mode', 'blob'))
    cellpose_model_var = ctk.StringVar(value=prev.get('cellpose_model', ''))
    status_var = ctk.StringVar(value="Ready")
    fit_mode_var = ctk.StringVar(value=prev_fit_mode)
    subtract_background = ctk.BooleanVar(value=prev.get('subtract_background', False))
    review_detection_var = ctk.BooleanVar(value=prev.get('review_detection', True))
    skip_cellpose_review_var = ctk.BooleanVar(value=prev.get('skip_cellpose_review', False))
    qa_snr_var = ctk.StringVar(value=str(prev.get('qa_min_snr', '3.0')))
    qa_sym_var = ctk.StringVar(value=str(prev.get('qa_min_symmetry', '0.6')))
    save_diagnostics_var = ctk.BooleanVar(value=prev.get('save_diagnostics', False))
    qa_auto_reject_var = ctk.BooleanVar(value=prev.get('qa_auto_reject', False))
    cellpose_do_3d_var = ctk.BooleanVar(value=prev.get('cellpose_do_3d', False))
    anisotropy_var = ctk.StringVar(value=str(prev.get('anisotropy', '')) if prev.get('anisotropy') is not None else '')
    blob_fallback_var = ctk.BooleanVar(value=prev.get('use_blob_fallback', False))
    local_background_var = ctk.BooleanVar(value=prev.get('local_background', False))
    robust_fit_var = ctk.BooleanVar(value=prev.get('robust_fit', True))
    cellpose_min_size_var = ctk.StringVar(value=str(prev.get('cellpose_min_size', '3')))
    cellpose_flow_threshold_var = ctk.StringVar(value=str(prev.get('cellpose_flow_threshold', '0.4')))
    num_beads_avg_var = ctk.StringVar(value=str(prev.get('num_beads_avg', '20')))

    def browse_input():
        path = filedialog.askopenfilename(
            title="Select microscopy image",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("OME-TIFF", "*.ome.tif"),
                ("All files", "*.*"),
            ],
        )
        if path:
            input_file.set(path)
            if not output_dir.get():
                output_dir.set(str(Path(path).parent))
            _update_channel_options(path)

    def browse_output():
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            output_dir.set(path)

    def browse_cellpose_model():
        path = filedialog.askopenfilename(title="Select Cellpose model file")
        if path:
            cellpose_model_var.set(path)

    def _detect_channel_count(path):
        """Infer number of channels from TIFF metadata without full volume read."""
        try:
            import tifffile

            with tifffile.TiffFile(str(path)) as tif:
                if not tif.series:
                    return 1
                series = tif.series[0]
                shape = tuple(getattr(series, "shape", ()))
                axes = str(getattr(series, "axes", ""))
                if "C" in axes:
                    idx = axes.index("C")
                    return max(1, int(shape[idx]))
                # Fallback convention used by this project for 4D stacks: (C, Z, Y, X)
                if len(shape) == 4:
                    return max(1, int(shape[0]))
        except Exception:
            pass
        return 1

    def _update_channel_options(path):
        """Refresh channel dropdown values from selected input file."""
        n_channels = _detect_channel_count(path) if path else 1
        values = [str(i) for i in range(n_channels)]
        channel_menu.configure(values=values)
        if channel_var.get() not in values:
            channel_var.set(values[0])

    def run():
        inp = input_file.get()
        if not inp or not Path(inp).exists():
            messagebox.showerror("Error", "Select a valid input file.")
            return
        try:
            sx = float(scale_xy.get())
            sz = float(scale_z.get())
            bx = int(box_entry_var.get())
            ch = int(channel_var.get())
            qa_snr = float(qa_snr_var.get())
            qa_sym = float(qa_sym_var.get())
            aniso = float(anisotropy_var.get()) if anisotropy_var.get().strip() else None
            cp_min_size = int(cellpose_min_size_var.get())
            cp_flow = float(cellpose_flow_threshold_var.get())
            n_avg = int(num_beads_avg_var.get())
        except ValueError:
            messagebox.showerror("Error", "Scale XY, Z, channel, QA, box size, anisotropy, num beads avg, and Cellpose params must be valid numbers.")
            return
        out = output_dir.get() or str(Path(inp).parent)
        na_val = float(na_var.get()) if na_var.get().strip() else None
        fluor = fluorophore_var.get().strip() or None

        def status(msg):
            status_var.set(msg)
            app.update_idletasks()

        cellpose_path = (cellpose_model_var.get().strip() or None) if mode_var.get() == 'cellpose' else None
        fit_mode = fit_mode_var.get()
        fit_gaussian = fit_mode in ('1d', 'both')
        fit_3d = fit_mode in ('3d', 'both')

        def run_thread():
            _run_analysis(
                inp, out, mode_var.get(), sx, sz, na_val, fluor,
                bx, fit_gaussian, subtract_background.get(),
                status,
                cellpose_model_path=cellpose_path,
                channel=ch,
                review_detection=review_detection_var.get(),
                skip_cellpose_review=skip_cellpose_review_var.get(),
                qa_min_snr=qa_snr,
                qa_min_symmetry=qa_sym,
                fit_3d=fit_3d,
                save_diagnostics=save_diagnostics_var.get(),
                qa_auto_reject=qa_auto_reject_var.get(),
                cellpose_do_3d=cellpose_do_3d_var.get(),
                anisotropy=aniso,
                use_blob_fallback=blob_fallback_var.get(),
                local_background=local_background_var.get(),
                robust_fit=robust_fit_var.get(),
                cellpose_min_size=cp_min_size,
                cellpose_flow_threshold=cp_flow,
                num_beads_avg=n_avg,
            )

        settings = {
            'input_file': inp,
            'output_dir': out,
            'mode': mode_var.get(),
            'scale_xy': sx,
            'scale_z': sz,
            'na': na_val,
            'fluorophore': fluor,
            'channel': ch,
            'box_size': bx,
            'fit_mode': fit_mode,
            'subtract_background': subtract_background.get(),
            'review_detection': review_detection_var.get(),
            'skip_cellpose_review': skip_cellpose_review_var.get(),
            'qa_min_snr': qa_snr,
            'qa_min_symmetry': qa_sym,
            'save_diagnostics': save_diagnostics_var.get(),
            'qa_auto_reject': qa_auto_reject_var.get(),
            'cellpose_do_3d': cellpose_do_3d_var.get(),
            'anisotropy': aniso,
            'use_blob_fallback': blob_fallback_var.get(),
            'local_background': local_background_var.get(),
            'robust_fit': robust_fit_var.get(),
            'cellpose_min_size': cp_min_size,
            'cellpose_flow_threshold': cp_flow,
            'num_beads_avg': n_avg,
            'show_advanced': show_advanced.get(),
        }
        if cellpose_path:
            settings['cellpose_model'] = cellpose_path
        # Save to output directory (portable config for CLI --config)
        settings_path = Path(out) / 'bead_analyzer_settings.json'
        try:
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass
        # Save to home directory (auto-restored on next GUI launch)
        try:
            with open(_settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

        status_var.set("Running...")
        threading.Thread(target=run_thread, daemon=True).start()

    # Layout
    pad = {"padx": 12, "pady": 6}
    ctk.CTkLabel(app, text="Bead Analyzer", font=ctk.CTkFont(size=18, weight="bold")).pack(**pad)

    # File section
    frame_file = ctk.CTkFrame(app, fg_color="transparent")
    frame_file.pack(fill="x", **pad)
    ctk.CTkLabel(frame_file, text="Input file:").pack(anchor="w")
    entry_input = ctk.CTkEntry(frame_file, textvariable=input_file, width=400)
    entry_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
    ctk.CTkButton(frame_file, text="Browse", width=80, command=browse_input).pack(side="left")

    ctk.CTkLabel(app, text="Output directory:").pack(anchor="w", **pad)
    frame_out = ctk.CTkFrame(app, fg_color="transparent")
    frame_out.pack(fill="x", **pad)
    entry_out = ctk.CTkEntry(frame_out, textvariable=output_dir, width=400)
    entry_out.pack(side="left", fill="x", expand=True, padx=(0, 6))
    ctk.CTkButton(frame_out, text="Browse", width=80, command=browse_output).pack(side="left")

    # Parameters
    ctk.CTkLabel(app, text="Experimental parameters", font=ctk.CTkFont(weight="bold")).pack(anchor="w", **pad)
    frame_params = ctk.CTkFrame(app, fg_color="transparent")
    frame_params.pack(fill="x", **pad)
    ctk.CTkLabel(frame_params, text="Scaling (um/pix):").pack(side="left", padx=(0, 8))
    ctk.CTkLabel(frame_params, text="XY").pack(side="left", padx=(0, 4))
    ctk.CTkEntry(frame_params, textvariable=scale_xy, width=80).pack(side="left", padx=(0, 12))
    ctk.CTkLabel(frame_params, text="Z").pack(side="left", padx=(0, 4))
    ctk.CTkEntry(frame_params, textvariable=scale_z, width=80).pack(side="left")

    frame_channel = ctk.CTkFrame(app, fg_color="transparent")
    frame_channel.pack(fill="x", **pad)
    ctk.CTkLabel(frame_channel, text="Channel:").pack(side="left", padx=(0, 8))
    channel_menu = ctk.CTkOptionMenu(frame_channel, variable=channel_var, values=["0"], width=100)
    channel_menu.pack(side="left")
    _update_channel_options(input_file.get() if input_file.get() else None)

    # Advanced Options Checkbox (packed at bottom, below status)
    show_advanced_cb = ctk.CTkCheckBox(app, text="Show Advanced Options", variable=show_advanced,
                                       font=ctk.CTkFont(weight="bold", size=13))

    # Mode
    ctk.CTkLabel(app, text="Detection mode", font=ctk.CTkFont(weight="bold")).pack(anchor="w", **pad)
    frame_mode = ctk.CTkFrame(app, fg_color="transparent")
    frame_mode.pack(fill="x", **pad)
    manual_rb = ctk.CTkRadioButton(frame_mode, text="Manual", variable=mode_var, value="manual")
    manual_rb.pack(side="left", padx=(0, 10))
    blob_rb = ctk.CTkRadioButton(frame_mode, text="Blob", variable=mode_var, value="blob")
    blob_rb.pack(side="left", padx=(0, 10))
    trackpy_rb = ctk.CTkRadioButton(frame_mode, text="Trackpy", variable=mode_var, value="trackpy")
    trackpy_rb.pack(side="left", padx=(0, 10))
    stardist_rb = ctk.CTkRadioButton(frame_mode, text="StarDist", variable=mode_var, value="stardist")
    stardist_rb.pack(side="left", padx=(0, 10))
    cellpose_rb = ctk.CTkRadioButton(frame_mode, text="Cellpose", variable=mode_var, value="cellpose")
    cellpose_rb.pack(side="left")

    # --- Analysis options (always enabled) ---
    ctk.CTkLabel(app, text="Analysis options", font=ctk.CTkFont(weight="bold")).pack(anchor="w", **pad)

    # Fitting
    sub = ctk.CTkFont(size=12, weight="bold")
    ctk.CTkLabel(app, text="Fitting", font=sub).pack(anchor="w", padx=12, pady=(4, 2))
    frame_fit = ctk.CTkFrame(app, fg_color="transparent")
    frame_fit.pack(fill="x", **pad)
    ctk.CTkRadioButton(frame_fit, text="1D Gaussian", variable=fit_mode_var, value="1d").grid(row=0, column=0, sticky="w", padx=(0, 18))
    ctk.CTkRadioButton(frame_fit, text="3D Gaussian", variable=fit_mode_var, value="3d").grid(row=0, column=1, sticky="w", padx=(0, 18))
    ctk.CTkRadioButton(frame_fit, text="Both", variable=fit_mode_var, value="both").grid(row=0, column=2, sticky="w", padx=(0, 18))
    ctk.CTkRadioButton(frame_fit, text="No fit", variable=fit_mode_var, value="none").grid(row=0, column=3, sticky="w")
    ctk.CTkLabel(frame_fit, text="", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=0, sticky="w")
    ctk.CTkLabel(frame_fit, text="slower", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=1, sticky="w")
    ctk.CTkLabel(frame_fit, text="", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=2, sticky="w")
    ctk.CTkLabel(frame_fit, text="peak width only", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=3, sticky="w")
    ctk.CTkCheckBox(frame_fit, text="Robust fit (Huber loss)", variable=robust_fit_var).grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

    # General numeric params
    frame_gen = ctk.CTkFrame(app, fg_color="transparent")
    frame_gen.pack(fill="x", **pad)
    box_entry_var = ctk.StringVar(value=str(prev.get('box_size', '15')))
    ctk.CTkLabel(frame_gen, text="Box size (px):").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_gen, textvariable=box_entry_var, width=50).pack(side="left", padx=(0, 16))
    ctk.CTkLabel(frame_gen, text="Beads to avg (0=all):").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_gen, textvariable=num_beads_avg_var, width=50).pack(side="left")

    # Background
    ctk.CTkLabel(app, text="Background", font=sub).pack(anchor="w", padx=12, pady=(4, 2))
    frame_bg = ctk.CTkFrame(app, fg_color="transparent")
    frame_bg.pack(fill="x", **pad)
    ctk.CTkCheckBox(frame_bg, text="Subtract global background", variable=subtract_background).pack(side="left", padx=(0, 16))
    ctk.CTkCheckBox(frame_bg, text="Local background", variable=local_background_var).pack(side="left")

    # Quality & output
    ctk.CTkLabel(app, text="Quality & output", font=sub).pack(anchor="w", padx=12, pady=(4, 2))
    frame_qa_cb = ctk.CTkFrame(app, fg_color="transparent")
    frame_qa_cb.pack(fill="x", **pad)
    ctk.CTkCheckBox(frame_qa_cb, text="Save bead diagnostics", variable=save_diagnostics_var).pack(side="left", padx=(0, 16))
    ctk.CTkCheckBox(frame_qa_cb, text="Auto-reject low QA", variable=qa_auto_reject_var).pack(side="left")
    frame_qa = ctk.CTkFrame(app, fg_color="transparent")
    frame_qa.pack(fill="x", **pad)
    ctk.CTkLabel(frame_qa, text="QA min SNR:").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_qa, textvariable=qa_snr_var, width=60).pack(side="left", padx=(0, 16))
    ctk.CTkLabel(frame_qa, text="QA min symmetry:").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_qa, textvariable=qa_sym_var, width=60).pack(side="left")

    # --- Advanced options (Blob / Trackpy / StarDist only) ---
    detection_header = ctk.CTkLabel(app, text="Advanced options", font=ctk.CTkFont(weight="bold"))
    detection_header.pack(anchor="w", **pad)
    frame_det = ctk.CTkFrame(app, fg_color="transparent")
    frame_det.pack(fill="x", **pad)
    review_detection_cb = ctk.CTkCheckBox(frame_det, text="Review detection overlay", variable=review_detection_var)
    review_detection_cb.pack(side="left", padx=(0, 16))
    blob_fallback_cb = ctk.CTkCheckBox(frame_det, text="Blob fallback (StarDist)", variable=blob_fallback_var)
    blob_fallback_cb.pack(side="left")

    # --- Cellpose options (Cellpose only) ---
    cellpose_header = ctk.CTkLabel(app, text="Cellpose options", font=ctk.CTkFont(weight="bold"))
    cellpose_header.pack(anchor="w", **pad)
    frame_cp_model = ctk.CTkFrame(app, fg_color="transparent")
    frame_cp_model.pack(fill="x", **pad)
    cp_model_label = ctk.CTkLabel(frame_cp_model, text="Model file:")
    cp_model_label.pack(side="left", padx=(0, 8))
    cp_model_entry = ctk.CTkEntry(frame_cp_model, textvariable=cellpose_model_var, width=300)
    cp_model_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    cp_model_browse = ctk.CTkButton(frame_cp_model, text="Browse", width=80, command=browse_cellpose_model)
    cp_model_browse.pack(side="left")
    cp_env_hint = ctk.CTkLabel(app, text="(or set FWHM_CELLPOSE_MODEL env var)", text_color="gray", font=ctk.CTkFont(size=11))
    cp_env_hint.pack(anchor="w", padx=(12, 0))
    frame_cp_checks = ctk.CTkFrame(app, fg_color="transparent")
    frame_cp_checks.pack(fill="x", **pad)
    cp_do_3d_cb = ctk.CTkCheckBox(frame_cp_checks, text="Native 3D", variable=cellpose_do_3d_var)
    cp_do_3d_cb.pack(side="left", padx=(0, 16))
    cp_skip_review_cb = ctk.CTkCheckBox(frame_cp_checks, text="Skip review", variable=skip_cellpose_review_var)
    cp_skip_review_cb.pack(side="left")
    frame_cp_aniso = ctk.CTkFrame(app, fg_color="transparent")
    frame_cp_aniso.pack(fill="x", **pad)
    cp_aniso_label = ctk.CTkLabel(frame_cp_aniso, text="Anisotropy (z/xy):")
    cp_aniso_label.pack(side="left", padx=(0, 8))
    cp_aniso_entry = ctk.CTkEntry(frame_cp_aniso, textvariable=anisotropy_var, width=70)
    cp_aniso_entry.pack(side="left")
    frame_cp_params = ctk.CTkFrame(app, fg_color="transparent")
    frame_cp_params.pack(fill="x", **pad)
    cp_minsize_label = ctk.CTkLabel(frame_cp_params, text="Min size (px):")
    cp_minsize_label.pack(side="left", padx=(0, 8))
    cp_minsize_entry = ctk.CTkEntry(frame_cp_params, textvariable=cellpose_min_size_var, width=50)
    cp_minsize_entry.pack(side="left", padx=(0, 16))
    cp_flow_label = ctk.CTkLabel(frame_cp_params, text="Flow threshold:")
    cp_flow_label.pack(side="left", padx=(0, 8))
    cp_flow_entry = ctk.CTkEntry(frame_cp_params, textvariable=cellpose_flow_threshold_var, width=50)
    cp_flow_entry.pack(side="left")

    # --- Mode-dependent enable/disable ---
    _default_label_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
    _disabled_color = "gray40"
    _sections = {
        'detection': {
            'header': detection_header,
            'widgets': [review_detection_cb, blob_fallback_cb],
            'labels': [],
        },
        'cellpose': {
            'header': cellpose_header,
            'widgets': [cp_model_entry, cp_model_browse, cp_do_3d_cb,
                        cp_skip_review_cb, cp_aniso_entry, cp_minsize_entry,
                        cp_flow_entry],
            'labels': [cp_model_label, cp_aniso_label, cp_minsize_label,
                       cp_flow_label],
        },
    }

    def _set_section_state(name, enabled):
        sec = _sections[name]
        state = "normal" if enabled else "disabled"
        color = _default_label_color if enabled else _disabled_color
        sec['header'].configure(text_color=color)
        for w in sec['widgets']:
            w.configure(state=state)
        for lbl in sec['labels']:
            lbl.configure(text_color=color)

    # Advanced options widgets to hide/show
    _advanced_widgets = {
        'mode_buttons': [stardist_rb, cellpose_rb],
        'sections': {
            'detection': {
                'header': detection_header,
                'widgets': [review_detection_cb, blob_fallback_cb],
                'labels': [],
            },
            'cellpose': {
                'header': cellpose_header,
                'widgets': [cp_model_entry, cp_model_browse, cp_do_3d_cb,
                            cp_skip_review_cb, cp_aniso_entry, cp_minsize_entry,
                            cp_flow_entry],
                'labels': [cp_model_label, cp_aniso_label, cp_minsize_label,
                           cp_flow_label],
                'frames': [frame_cp_model, cp_env_hint, frame_cp_checks,
                          frame_cp_aniso, frame_cp_params],
            },
        },
    }

    def _toggle_advanced_options(*_args):
        """Show/hide advanced options based on checkbox state."""
        is_advanced = show_advanced.get()

        # Show/hide StarDist and Cellpose mode buttons
        for btn in _advanced_widgets['mode_buttons']:
            if is_advanced:
                btn.pack(side="left", padx=(0, 10) if btn == stardist_rb else 0)
            else:
                btn.pack_forget()
                # If user was on advanced mode, switch to blob
                if mode_var.get() in ('stardist', 'cellpose'):
                    mode_var.set('blob')

        # Show/hide detection and cellpose sections
        if is_advanced:
            # Detection section
            detection_header.pack(anchor="w", **pad)
            frame_det.pack(fill="x", **pad)
            # Cellpose section
            cellpose_header.pack(anchor="w", **pad)
            for frame in _advanced_widgets['sections']['cellpose']['frames']:
                if frame == cp_env_hint:
                    frame.pack(anchor="w", padx=(12, 0))
                else:
                    frame.pack(fill="x", **pad)
        else:
            # Hide detection section
            detection_header.pack_forget()
            frame_det.pack_forget()
            # Hide cellpose section
            cellpose_header.pack_forget()
            for frame in _advanced_widgets['sections']['cellpose']['frames']:
                frame.pack_forget()

        # Resize window to fit content
        if is_advanced:
            app.geometry(f"600x{HEIGHT_EXPANDED}")
            app.minsize(550, HEIGHT_EXPANDED)
        else:
            app.geometry(f"600x{HEIGHT_COMPACT}")
            app.minsize(550, HEIGHT_COMPACT)

    def _on_mode_change(*_args):
        mode = mode_var.get()
        is_advanced = show_advanced.get()

        # Only apply mode-dependent logic if advanced mode is on
        if is_advanced:
            detection_on = mode in ('blob', 'trackpy', 'stardist')
            _set_section_state('detection', detection_on)
            if detection_on:
                blob_fallback_cb.configure(state="normal" if mode == 'stardist' else "disabled")
            cellpose_on = mode == 'cellpose'
            _set_section_state('cellpose', cellpose_on)
            cp_env_hint.configure(text_color="gray" if cellpose_on else _disabled_color)

    show_advanced.trace_add("write", _toggle_advanced_options)
    mode_var.trace_add("write", _on_mode_change)
    _toggle_advanced_options()  # Initialize visibility
    _on_mode_change()

    # Run button (extra 10px above to push down)
    ctk.CTkButton(app, text="Analyze beads", command=run, height=36, font=ctk.CTkFont(weight="bold")).pack(padx=12, pady=(16, 6))

    # Status
    ctk.CTkLabel(app, textvariable=status_var, text_color="gray").pack(**pad)

    show_advanced_cb.pack(anchor="w", **pad)

    app.mainloop()
    return 0


if __name__ == "__main__":
    exit(main())
