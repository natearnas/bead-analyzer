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
    # Window size is controlled dynamically by advanced/docs toggles.
    HEIGHT_COMPACT = 910
    HEIGHT_EXPANDED = 1220
    WIDTH_BASE = 620
    WIDTH_WITH_DOCS = 1080
    app.geometry(f"{WIDTH_BASE}x{HEIGHT_COMPACT}")
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
    docs_open_var = ctk.BooleanVar(value=False)

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

    def _apply_window_geometry():
        """Apply width/height based on current UI visibility toggles."""
        target_h = HEIGHT_EXPANDED if show_advanced.get() else HEIGHT_COMPACT
        target_w = WIDTH_WITH_DOCS if docs_open_var.get() else WIDTH_BASE
        app.geometry(f"{target_w}x{target_h}")
        app.minsize(550, target_h)

    # Layout
    pad = {"padx": 12, "pady": 6}
    button_font = ctk.CTkFont(size=14, weight="bold")
    toggle_font = ctk.CTkFont(size=14)
    root_frame = ctk.CTkFrame(app, fg_color="transparent")
    root_frame.pack(fill="both", expand=True)
    left_frame = ctk.CTkFrame(root_frame, fg_color="transparent")
    left_frame.pack(side="left", fill="both", expand=True)

    frame_header = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_header.pack(fill="x", **pad)
    ctk.CTkLabel(frame_header, text="Bead Analyzer", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
    docs_toggle_btn = ctk.CTkButton(frame_header, text="Docs ▸", width=96)
    docs_toggle_btn.pack(side="right")

    # File section
    file_results_label = ctk.CTkLabel(left_frame, text="Input file and Results", font=ctk.CTkFont(weight="bold"))
    file_results_label.pack(anchor="w", **pad)
    frame_file = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_file.pack(fill="x", **pad)
    input_file_label = ctk.CTkLabel(frame_file, text="Input file:")
    input_file_label.pack(anchor="w")
    entry_input = ctk.CTkEntry(frame_file, textvariable=input_file, width=400)
    entry_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
    browse_input_btn = ctk.CTkButton(frame_file, text="Browse", width=80, command=browse_input)
    browse_input_btn.pack(side="left")

    output_dir_label = ctk.CTkLabel(left_frame, text="Results directory:")
    output_dir_label.pack(anchor="w", **pad)
    frame_out = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_out.pack(fill="x", **pad)
    entry_out = ctk.CTkEntry(frame_out, textvariable=output_dir, width=400)
    entry_out.pack(side="left", fill="x", expand=True, padx=(0, 6))
    browse_out_btn = ctk.CTkButton(frame_out, text="Browse", width=80, command=browse_output)
    browse_out_btn.pack(side="left")

    # Parameters
    experimental_params_label = ctk.CTkLabel(left_frame, text="Experimental parameters", font=ctk.CTkFont(weight="bold"))
    experimental_params_label.pack(anchor="w", **pad)
    frame_params = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_params.pack(fill="x", **pad)
    scaling_label = ctk.CTkLabel(frame_params, text="Scaling (um/pix):")
    scaling_label.pack(side="left", padx=(0, 8))
    ctk.CTkLabel(frame_params, text="XY").pack(side="left", padx=(0, 4))
    ctk.CTkEntry(frame_params, textvariable=scale_xy, width=80).pack(side="left", padx=(0, 12))
    ctk.CTkLabel(frame_params, text="Z").pack(side="left", padx=(0, 4))
    ctk.CTkEntry(frame_params, textvariable=scale_z, width=80).pack(side="left")

    frame_channel = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_channel.pack(fill="x", **pad)
    channel_label = ctk.CTkLabel(frame_channel, text="Channel:")
    channel_label.pack(side="left", padx=(0, 8))
    channel_menu = ctk.CTkOptionMenu(frame_channel, variable=channel_var, values=["0"], width=100)
    channel_menu.pack(side="left")
    _update_channel_options(input_file.get() if input_file.get() else None)

    # Advanced Options Checkbox (packed at bottom, below status)
    show_advanced_cb = ctk.CTkCheckBox(left_frame, text="Show Advanced Options", variable=show_advanced,
                                       font=ctk.CTkFont(weight="bold", size=13))

    # Mode
    detection_mode_label = ctk.CTkLabel(left_frame, text="Detection mode", font=ctk.CTkFont(weight="bold"))
    detection_mode_label.pack(anchor="w", **pad)
    frame_mode = ctk.CTkFrame(left_frame, fg_color="transparent")
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
    analysis_options_label = ctk.CTkLabel(left_frame, text="Analysis options", font=ctk.CTkFont(weight="bold"))
    analysis_options_label.pack(anchor="w", **pad)

    # Fitting
    sub = ctk.CTkFont(size=12, weight="bold")
    fitting_label = ctk.CTkLabel(left_frame, text="Fitting", font=sub)
    fitting_label.pack(anchor="w", padx=12, pady=(4, 2))
    frame_fit = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_fit.pack(fill="x", **pad)
    fit_1d_rb = ctk.CTkRadioButton(frame_fit, text="1D Gaussian", variable=fit_mode_var, value="1d")
    fit_1d_rb.grid(row=0, column=0, sticky="w", padx=(0, 18))
    fit_3d_rb = ctk.CTkRadioButton(frame_fit, text="3D Gaussian", variable=fit_mode_var, value="3d")
    fit_3d_rb.grid(row=0, column=1, sticky="w", padx=(0, 18))
    fit_both_rb = ctk.CTkRadioButton(frame_fit, text="Both", variable=fit_mode_var, value="both")
    fit_both_rb.grid(row=0, column=2, sticky="w", padx=(0, 18))
    fit_none_rb = ctk.CTkRadioButton(frame_fit, text="No fit", variable=fit_mode_var, value="none")
    fit_none_rb.grid(row=0, column=3, sticky="w")
    ctk.CTkLabel(frame_fit, text="", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=0, sticky="w")
    ctk.CTkLabel(frame_fit, text="slower", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=1, sticky="w")
    ctk.CTkLabel(frame_fit, text="", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=2, sticky="w")
    ctk.CTkLabel(frame_fit, text="peak width only", font=ctk.CTkFont(size=11), text_color="gray").grid(row=1, column=3, sticky="w")
    robust_fit_cb = ctk.CTkCheckBox(frame_fit, text="Robust fit (Huber loss)", variable=robust_fit_var)
    robust_fit_cb.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

    # General numeric params
    frame_gen = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_gen.pack(fill="x", **pad)
    box_entry_var = ctk.StringVar(value=str(prev.get('box_size', '15')))
    box_size_label = ctk.CTkLabel(frame_gen, text="Box size (px):")
    box_size_label.pack(side="left", padx=(0, 8))
    box_entry = ctk.CTkEntry(frame_gen, textvariable=box_entry_var, width=50)
    box_entry.pack(side="left", padx=(0, 16))
    num_beads_label = ctk.CTkLabel(frame_gen, text="Beads to avg (0=all):")
    num_beads_label.pack(side="left", padx=(0, 8))
    num_beads_entry = ctk.CTkEntry(frame_gen, textvariable=num_beads_avg_var, width=50)
    num_beads_entry.pack(side="left")

    # Background
    background_label = ctk.CTkLabel(left_frame, text="Background", font=sub)
    background_label.pack(anchor="w", padx=12, pady=(4, 2))
    frame_bg = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_bg.pack(fill="x", **pad)
    subtract_bg_cb = ctk.CTkCheckBox(frame_bg, text="Subtract global background", variable=subtract_background)
    subtract_bg_cb.pack(side="left", padx=(0, 16))
    local_bg_cb = ctk.CTkCheckBox(frame_bg, text="Local background", variable=local_background_var)
    local_bg_cb.pack(side="left")

    # Quality & output
    quality_output_label = ctk.CTkLabel(left_frame, text="Quality & output", font=sub)
    quality_output_label.pack(anchor="w", padx=12, pady=(4, 2))
    frame_qa_cb = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_qa_cb.pack(fill="x", **pad)
    save_diag_cb = ctk.CTkCheckBox(frame_qa_cb, text="Save bead diagnostics", variable=save_diagnostics_var)
    save_diag_cb.pack(side="left", padx=(0, 16))
    qa_reject_cb = ctk.CTkCheckBox(frame_qa_cb, text="Auto-reject low QA", variable=qa_auto_reject_var)
    qa_reject_cb.pack(side="left")
    frame_qa = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_qa.pack(fill="x", **pad)
    qa_snr_label = ctk.CTkLabel(frame_qa, text="QA min SNR:")
    qa_snr_label.pack(side="left", padx=(0, 8))
    qa_snr_entry = ctk.CTkEntry(frame_qa, textvariable=qa_snr_var, width=60)
    qa_snr_entry.pack(side="left", padx=(0, 16))
    qa_sym_label = ctk.CTkLabel(frame_qa, text="QA min symmetry:")
    qa_sym_label.pack(side="left", padx=(0, 8))
    qa_sym_entry = ctk.CTkEntry(frame_qa, textvariable=qa_sym_var, width=60)
    qa_sym_entry.pack(side="left")

    # --- Advanced options (Blob / Trackpy / StarDist only) ---
    detection_header = ctk.CTkLabel(left_frame, text="Advanced options", font=ctk.CTkFont(weight="bold"))
    detection_header.pack(anchor="w", **pad)
    frame_det = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_det.pack(fill="x", **pad)
    review_detection_cb = ctk.CTkCheckBox(frame_det, text="Review detection overlay", variable=review_detection_var)
    review_detection_cb.pack(side="left", padx=(0, 16))
    blob_fallback_cb = ctk.CTkCheckBox(frame_det, text="Blob fallback (StarDist)", variable=blob_fallback_var)
    blob_fallback_cb.pack(side="left")

    # --- Cellpose options (Cellpose only) ---
    cellpose_header = ctk.CTkLabel(left_frame, text="Cellpose options", font=ctk.CTkFont(weight="bold"))
    cellpose_header.pack(anchor="w", **pad)
    frame_cp_model = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_cp_model.pack(fill="x", **pad)
    cp_model_label = ctk.CTkLabel(frame_cp_model, text="Model file:")
    cp_model_label.pack(side="left", padx=(0, 8))
    cp_model_entry = ctk.CTkEntry(frame_cp_model, textvariable=cellpose_model_var, width=300)
    cp_model_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    cp_model_browse = ctk.CTkButton(frame_cp_model, text="Browse", width=80, command=browse_cellpose_model)
    cp_model_browse.pack(side="left")
    cp_env_hint = ctk.CTkLabel(left_frame, text="(or set FWHM_CELLPOSE_MODEL env var)", text_color="gray", font=ctk.CTkFont(size=11))
    cp_env_hint.pack(anchor="w", padx=(12, 0))
    frame_cp_checks = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_cp_checks.pack(fill="x", **pad)
    cp_do_3d_cb = ctk.CTkCheckBox(frame_cp_checks, text="Native 3D", variable=cellpose_do_3d_var)
    cp_do_3d_cb.pack(side="left", padx=(0, 16))
    cp_skip_review_cb = ctk.CTkCheckBox(frame_cp_checks, text="Skip review", variable=skip_cellpose_review_var)
    cp_skip_review_cb.pack(side="left")
    frame_cp_aniso = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_cp_aniso.pack(fill="x", **pad)
    cp_aniso_label = ctk.CTkLabel(frame_cp_aniso, text="Anisotropy (z/xy):")
    cp_aniso_label.pack(side="left", padx=(0, 8))
    cp_aniso_entry = ctk.CTkEntry(frame_cp_aniso, textvariable=anisotropy_var, width=70)
    cp_aniso_entry.pack(side="left")
    frame_cp_params = ctk.CTkFrame(left_frame, fg_color="transparent")
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

        _apply_window_geometry()

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
    # Run button (extra 10px above to push down)
    analyze_btn = ctk.CTkButton(left_frame, text="Analyze beads", command=run, height=36, font=ctk.CTkFont(weight="bold"))
    analyze_btn.pack(padx=12, pady=(16, 6))

    # Status
    ctk.CTkLabel(left_frame, textvariable=status_var, text_color="gray").pack(**pad)

    show_advanced_cb.pack(anchor="w", **pad)

    # Right-side expandable docs panel
    docs_panel = ctk.CTkFrame(root_frame, width=430)
    docs_panel.pack_propagate(False)

    docs_header = ctk.CTkFrame(docs_panel, fg_color="transparent")
    docs_header.pack(fill="x", padx=12, pady=(12, 8))
    ctk.CTkLabel(docs_header, text="Settings and Use Cases", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")

    docs_tabs = ctk.CTkTabview(docs_panel)
    docs_tabs.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    docs_tabs.add("Settings")
    docs_tabs.add("Use Cases")

    settings_scroll = ctk.CTkScrollableFrame(docs_tabs.tab("Settings"))
    settings_scroll.pack(fill="both", expand=True)
    use_cases_scroll = ctk.CTkScrollableFrame(docs_tabs.tab("Use Cases"))
    use_cases_scroll.pack(fill="both", expand=True)

    setting_anchors = {}
    _active_highlight_key = None
    _clear_highlight_job = None

    def _add_setting_doc(key, title, body):
        section = ctk.CTkFrame(settings_scroll)
        section.pack(fill="x", padx=6, pady=(0, 10))
        default_text_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
        title_lbl = ctk.CTkLabel(section, text=title, font=ctk.CTkFont(weight="bold"), text_color=default_text_color)
        title_lbl.pack(anchor="w", padx=8, pady=(8, 2))
        body_lbl = ctk.CTkLabel(section, text=body, justify="left", wraplength=330, text_color=default_text_color)
        body_lbl.pack(anchor="w", padx=(26, 12), pady=(0, 8))
        divider = ctk.CTkFrame(section, height=1, fg_color=("gray78", "gray28"))
        divider.pack(fill="x", padx=(22, 14), pady=(0, 2))
        setting_anchors[key] = {
            "section": section,
            "title": title_lbl,
            "body": body_lbl,
            "title_color": default_text_color,
            "body_color": default_text_color,
            "divider": divider,
        }

    _add_setting_doc("section_file_results", "Input file and Results", "Select the image stack to analyze and where the results directory for outputs should be saved.")
    _add_setting_doc("input_file", "Input file", "Path to the TIFF/OME-TIFF bead image stack to analyze.")
    _add_setting_doc("output_dir", "Results directory", "Folder where CSV tables, figures, overlays, and config JSON files are saved.")
    _add_setting_doc("section_experimental", "Experimental parameters", "Core microscope acquisition settings used to convert pixel measurements into physical units.")
    _add_setting_doc("scale", "Scaling (um/pix): XY, Z", "Physical pixel size. Accurate values are critical for valid FWHM in microns.")
    _add_setting_doc("channel", "Channel", "Selects image channel index used for detection and measurement.")
    _add_setting_doc("show_advanced", "Show Advanced Options", "Reveals advanced detector/model options and additional controls.")
    _add_setting_doc("section_detection_mode", "Detection mode", "Select which bead-detection backend finds candidate bead centers for downstream FWHM analysis.")
    _add_setting_doc("mode", "Detection mode", "Manual click, Blob, Trackpy, StarDist, or Cellpose bead detection backend.")
    _add_setting_doc("section_analysis_options", "Analysis options", "Shared measurement settings controlling fitting, averaging, background handling, and quality filtering.")
    _add_setting_doc("section_fitting", "Fitting", "Controls how widths are estimated from profiles/volumes (parametric fit vs. prominence).")
    _add_setting_doc("fit_mode", "Fitting mode", "1D Gaussian, 3D Gaussian, Both, or No fit (prominence-only width).")
    _add_setting_doc("robust_fit", "Robust fit", "Huber-loss Gaussian fitting for better stability when profiles include outliers.")
    _add_setting_doc("box_size", "Box size (px)", "Crop size around each bead for local profile extraction.")
    _add_setting_doc("num_beads_avg", "Beads to avg (0=all)", "Number of beads to include in average bead/profile outputs; 0 uses all.")
    _add_setting_doc("section_background", "Background", "Options to remove baseline/background signal before width measurement.")
    _add_setting_doc("subtract_background", "Subtract global background", "Subtracts image-wide baseline before profile analysis.")
    _add_setting_doc("local_background", "Local background", "Uses local neighborhood baseline; useful with nonuniform haze/illumination.")
    _add_setting_doc("section_quality_output", "Quality & output", "Options for diagnostics and automatic rejection of low-quality bead measurements.")
    _add_setting_doc("save_diagnostics", "Save bead diagnostics", "Writes per-bead diagnostic figures for QA and troubleshooting.")
    _add_setting_doc("qa_auto_reject", "Auto-reject low QA", "Automatically excludes beads below SNR/symmetry thresholds.")
    _add_setting_doc("qa_snr", "QA min SNR", "Minimum signal-to-noise ratio to accept a bead (when auto-reject is enabled).")
    _add_setting_doc("qa_sym", "QA min symmetry", "Minimum symmetry metric for bead acceptance (when auto-reject is enabled).")
    _add_setting_doc("section_advanced", "Advanced options", "Detector-specific controls used primarily for automatic detection workflows.")
    _add_setting_doc("review_detection", "Review detection overlay", "Shows a visual overlay so you can validate automatic detections.")
    _add_setting_doc("blob_fallback", "Blob fallback (StarDist)", "Fallback to classical blob detector if StarDist misses beads.")
    _add_setting_doc("section_cellpose", "Cellpose options", "Model and inference controls specific to Cellpose detection mode.")
    _add_setting_doc("cellpose_model", "Cellpose model file", "Path to trained Cellpose model used in Cellpose mode.")
    _add_setting_doc("cellpose_native_3d", "Cellpose Native 3D", "Uses Cellpose 3D inference instead of tiled 2D inference.")
    _add_setting_doc("cellpose_skip_review", "Cellpose Skip review", "Skips interactive mask review step for faster runs.")
    _add_setting_doc("anisotropy", "Anisotropy (z/xy)", "Voxel anisotropy ratio for 3D Cellpose and depth-aware operations.")
    _add_setting_doc("cellpose_min_size", "Cellpose Min size", "Minimum object size kept by Cellpose post-processing.")
    _add_setting_doc("cellpose_flow_threshold", "Cellpose Flow threshold", "Flow error threshold controlling Cellpose mask acceptance.")

    for title, body in [
        ("Fast default workflow", "Use Blob + 1D Gaussian, keep robust fit ON, set correct XY/Z scale, and run with default QA."),
        ("High-quality publication workflow", "Use Both fitting modes, save diagnostics, and enable auto-reject with tuned QA thresholds."),
        ("Difficult background workflow", "Enable local background, inspect detection overlay, and tighten QA thresholds."),
        ("Cellpose workflow", "Enable Show Advanced Options, choose Cellpose mode, set model path, then tune min size and flow threshold."),
        ("Small beads workflow", "Start with Blob or Trackpy instead of DL models when beads are only a few pixels wide."),
    ]:
        box = ctk.CTkFrame(use_cases_scroll)
        box.pack(fill="x", padx=6, pady=(0, 10))
        ctk.CTkLabel(box, text=title, font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(8, 2))
        ctk.CTkLabel(box, text=body, justify="left", wraplength=330).pack(anchor="w", padx=(26, 12), pady=(0, 8))
        ctk.CTkFrame(box, height=1, fg_color=("gray78", "gray28")).pack(fill="x", padx=(22, 14), pady=(0, 2))

    def _toggle_docs_panel():
        if docs_open_var.get():
            docs_panel.pack_forget()
            docs_open_var.set(False)
            docs_toggle_btn.configure(text="Docs ▸")
        else:
            docs_panel.pack(side="right", fill="y", padx=(0, 8), pady=8)
            docs_open_var.set(True)
            docs_toggle_btn.configure(text="Docs ◂")
        _apply_window_geometry()

    docs_toggle_btn.configure(command=_toggle_docs_panel)

    # Larger text for interactive controls.
    for w in [docs_toggle_btn, browse_input_btn, browse_out_btn, cp_model_browse, analyze_btn]:
        w.configure(font=button_font)
    for w in [
        show_advanced_cb, manual_rb, blob_rb, trackpy_rb, stardist_rb, cellpose_rb,
        fit_1d_rb, fit_3d_rb, fit_both_rb, fit_none_rb, robust_fit_cb,
        subtract_bg_cb, local_bg_cb, save_diag_cb, qa_reject_cb,
        review_detection_cb, blob_fallback_cb, cp_do_3d_cb, cp_skip_review_cb,
    ]:
        w.configure(font=toggle_font)
    channel_menu.configure(font=toggle_font, dropdown_font=toggle_font)
    if hasattr(docs_tabs, "_segmented_button"):
        docs_tabs._segmented_button.configure(font=toggle_font)

    def _clear_setting_highlight():
        nonlocal _active_highlight_key, _clear_highlight_job
        if _active_highlight_key is None:
            _clear_highlight_job = None
            return
        card = setting_anchors.get(_active_highlight_key)
        if not card:
            _active_highlight_key = None
            _clear_highlight_job = None
            return
        card["section"].configure(fg_color="transparent")
        card["title"].configure(text_color=card["title_color"])
        card["body"].configure(text_color=card["body_color"])
        _active_highlight_key = None
        _clear_highlight_job = None

    def _highlight_setting(key):
        nonlocal _active_highlight_key, _clear_highlight_job
        if key not in setting_anchors:
            return
        if _clear_highlight_job is not None:
            try:
                app.after_cancel(_clear_highlight_job)
            except Exception:
                # Timer may already have fired; safe to ignore.
                pass
            _clear_highlight_job = None
        _clear_setting_highlight()
        card = setting_anchors[key]
        card["section"].configure(fg_color=("gray90", "gray24"))
        card["title"].configure(text_color=("#1f6aa5", "#5fa8e6"))
        card["body"].configure(text_color=("#1f1f1f", "#d6d6d6"))
        _active_highlight_key = key
        _clear_highlight_job = app.after(1800, _clear_setting_highlight)

    def _jump_to_setting(key):
        if key not in setting_anchors:
            return
        # Keep panel collapsed by default; only jump/highlight when user opens Docs.
        if not docs_open_var.get():
            return
        docs_tabs.set("Settings")
        app.update_idletasks()
        canvas = getattr(settings_scroll, "_parent_canvas", None)
        anchor = setting_anchors[key]["section"]
        if canvas is None:
            return
        bbox = canvas.bbox("all")
        if not bbox:
            return
        content_h = max(1, bbox[3] - bbox[1])
        y_pos = max(0, anchor.winfo_y() - 8)
        # yview_moveto expects a fraction of the full scrollregion, not max_scroll.
        canvas.yview_moveto(min(1.0, y_pos / content_h))
        _highlight_setting(key)

    def _bind_hover(widget, key):
        widget.bind("<Enter>", lambda _e, k=key: _jump_to_setting(k), add="+")

    _bind_hover(file_results_label, "section_file_results")
    _bind_hover(entry_input, "input_file")
    _bind_hover(input_file_label, "input_file")
    _bind_hover(output_dir_label, "output_dir")
    _bind_hover(entry_out, "output_dir")
    _bind_hover(experimental_params_label, "section_experimental")
    _bind_hover(scaling_label, "scale")
    _bind_hover(channel_label, "channel")
    _bind_hover(channel_menu, "channel")
    _bind_hover(detection_mode_label, "section_detection_mode")
    _bind_hover(manual_rb, "mode")
    _bind_hover(blob_rb, "mode")
    _bind_hover(trackpy_rb, "mode")
    _bind_hover(stardist_rb, "mode")
    _bind_hover(cellpose_rb, "mode")
    _bind_hover(analysis_options_label, "section_analysis_options")
    _bind_hover(fitting_label, "section_fitting")
    _bind_hover(fit_1d_rb, "fit_mode")
    _bind_hover(fit_3d_rb, "fit_mode")
    _bind_hover(fit_both_rb, "fit_mode")
    _bind_hover(fit_none_rb, "fit_mode")
    _bind_hover(robust_fit_cb, "robust_fit")
    _bind_hover(box_size_label, "box_size")
    _bind_hover(box_entry, "box_size")
    _bind_hover(num_beads_label, "num_beads_avg")
    _bind_hover(num_beads_entry, "num_beads_avg")
    _bind_hover(background_label, "section_background")
    _bind_hover(subtract_bg_cb, "subtract_background")
    _bind_hover(local_bg_cb, "local_background")
    _bind_hover(quality_output_label, "section_quality_output")
    _bind_hover(save_diag_cb, "save_diagnostics")
    _bind_hover(qa_reject_cb, "qa_auto_reject")
    _bind_hover(qa_snr_label, "qa_snr")
    _bind_hover(qa_snr_entry, "qa_snr")
    _bind_hover(qa_sym_label, "qa_sym")
    _bind_hover(qa_sym_entry, "qa_sym")
    _bind_hover(detection_header, "section_advanced")
    _bind_hover(review_detection_cb, "review_detection")
    _bind_hover(blob_fallback_cb, "blob_fallback")
    _bind_hover(cellpose_header, "section_cellpose")
    _bind_hover(cp_model_label, "cellpose_model")
    _bind_hover(cp_model_entry, "cellpose_model")
    _bind_hover(cp_do_3d_cb, "cellpose_native_3d")
    _bind_hover(cp_skip_review_cb, "cellpose_skip_review")
    _bind_hover(cp_aniso_label, "anisotropy")
    _bind_hover(cp_aniso_entry, "anisotropy")
    _bind_hover(cp_minsize_label, "cellpose_min_size")
    _bind_hover(cp_minsize_entry, "cellpose_min_size")
    _bind_hover(cp_flow_label, "cellpose_flow_threshold")
    _bind_hover(cp_flow_entry, "cellpose_flow_threshold")
    _bind_hover(show_advanced_cb, "show_advanced")

    _toggle_advanced_options()  # Initialize visibility
    _on_mode_change()
    _apply_window_geometry()

    app.mainloop()
    return 0


if __name__ == "__main__":
    exit(main())
