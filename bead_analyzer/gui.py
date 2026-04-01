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
import sys
import threading
from pathlib import Path

try:
    import customtkinter as ctk
    from tkinter import PhotoImage, filedialog, messagebox
except ImportError:
    ctk = None


def _run_analysis(input_file, output_dir, mode, scale_xy, scale_z, na, fluorophore,
                  box_size, fit_gaussian, subtract_background, status_callback,
                  cellpose_model_path=None, channel=0, review_detection=False,
                  skip_cellpose_review=False, qa_min_snr=3.0, qa_min_symmetry=0.6,
                  fit_3d=False, save_diagnostics=False, qa_auto_reject=False,
                  cellpose_do_3d=False, anisotropy=None, use_blob_fallback=False,
                  local_background=False, robust_fit=False,
                  trackpy_diameter=5, trackpy_minmass=5000, trackpy_separation=None,
                  cellpose_min_size=3, cellpose_flow_threshold=0.4,
                  num_beads_avg=20, sample_fraction=100, center_mode='peak',
                  run_on_main=None, run_settings=None):
    """Run analysis in background thread. run_on_main: callable to run interactive matplotlib on main thread."""
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
            'status_callback': status_callback,
            'run_on_main': run_on_main,
            'sample_fraction': sample_fraction,
            'center_mode': center_mode,
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
            kwargs['trackpy_diameter'] = trackpy_diameter
            kwargs['trackpy_minmass'] = trackpy_minmass
            kwargs['trackpy_separation'] = trackpy_separation
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
            stack=stack,
            profiles=profiles,
            num_beads_avg=num_beads_avg,
            center_mode=center_mode,
            run_settings=run_settings,
        )
        rej_msg = f" ({len(rejected)} rejected)" if rejected else ""
        status_callback(f"Done. {len(results)} beads analyzed{rej_msg}.")
    except Exception as e:
        status_callback(f"Error: {e}")


def main():
    if ctk is None:
        print("Install customtkinter: pip install customtkinter")
        return 1

    # Use TkAgg so review windows appear (not Agg which is non-interactive)
    import matplotlib
    matplotlib.use('TkAgg')

    from . import __version__

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "ArnasTechnologies.BeadAnalyzer"
            )
        except Exception:
            pass

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = ctk.CTk()
    app.title(f"Bead Analyzer v{__version__}")

    def _set_app_icon():
        assets_dir = Path(__file__).resolve().parent / "assets"
        ico_path = assets_dir / "app_icon.ico"
        png_path = assets_dir / "app_icon.png"

        try:
            if ico_path.exists():
                app.iconbitmap(str(ico_path))
        except Exception:
            pass

        try:
            if png_path.exists():
                # Keep a reference so Tk does not release the icon image.
                app._icon_photo = PhotoImage(file=str(png_path))
                app.iconphoto(True, app._icon_photo)
        except Exception:
            pass

    _set_app_icon()
    # Window size is controlled dynamically by advanced/docs toggles.
    # Values are empirical: measured to fit all widgets without scrolling.
    # Adjust if widgets are added/removed.
    HEIGHT_COMPACT = 1040      # core controls only
    HEIGHT_EXPANDED = 1260     # core + two-column advanced options visible
    WIDTH_BASE = 420           # controls panel only (compact)
    WIDTH_ADVANCED = 550       # controls panel when advanced options visible (wider)
    DOCS_PANEL_WIDTH = 820
    launch_x = max((app.winfo_screenwidth() - WIDTH_BASE) // 2, 0)
    launch_y = 10
    app.geometry(f"{WIDTH_BASE}x{HEIGHT_COMPACT}+{launch_x}+{launch_y}")
    app.minsize(550, HEIGHT_COMPACT)

    # Run Matplotlib review windows on the main thread to avoid "outside of the main thread" warnings
    from . import detectors
    def _run_on_main_and_wait(fn, *args, **kwargs):
        result = [None]
        exc = [None]
        done = threading.Event()
        def on_main():
            try:
                result[0] = fn(*args, **kwargs)
            except Exception as e:
                exc[0] = e
            finally:
                # Run GC and Tk update on main thread so any Tk/Matplotlib cleanup
                # (e.g. from closed review window) runs here, avoiding
                # "main thread is not in main loop" in worker thread.
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                except Exception:
                    pass
                import gc
                gc.collect()
                app.update_idletasks()
                done.set()
        app.after(0, on_main)
        done.wait()
        if exc[0]:
            raise exc[0]
        return result[0]
    detectors._main_thread_runner = _run_on_main_and_wait

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
    center_mode_prev = str(prev.get('center_mode', 'peak')).lower()
    if center_mode_prev not in ('peak', 'centroid', 'radial'):
        center_mode_prev = 'peak'
    center_mode_var = ctk.StringVar(value=center_mode_prev)
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
    trackpy_diameter_var = ctk.StringVar(value=str(prev.get('trackpy_diameter', '5')))
    trackpy_minmass_var = ctk.StringVar(value=str(prev.get('trackpy_minmass', '5000')))
    trackpy_separation_var = ctk.StringVar(value=str(prev.get('trackpy_separation', '')) if prev.get('trackpy_separation') is not None else '')
    cellpose_min_size_var = ctk.StringVar(value=str(prev.get('cellpose_min_size', '3')))
    cellpose_flow_threshold_var = ctk.StringVar(value=str(prev.get('cellpose_flow_threshold', '0.4')))
    prev_num_beads_avg = prev.get('num_beads_avg', 20)
    try:
        prev_num_beads_avg = max(1, int(prev_num_beads_avg))
    except (TypeError, ValueError):
        prev_num_beads_avg = 20
    num_beads_avg_var = ctk.StringVar(value=str(prev_num_beads_avg))
    sample_fraction_var = ctk.StringVar(value=str(prev.get('sample_fraction', '100')))
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
            tp_diameter = int(trackpy_diameter_var.get())
            tp_minmass = float(trackpy_minmass_var.get())
            tp_separation = int(trackpy_separation_var.get()) if trackpy_separation_var.get().strip() else None
            n_avg = max(1, int(num_beads_avg_var.get()))
            sample_frac = float(sample_fraction_var.get())
        except ValueError:
            messagebox.showerror("Error", "Scale XY, Z, channel, QA, box width, anisotropy, num beads avg, sample fraction, Trackpy params, and Cellpose params must be valid numbers.")
            return
        num_beads_avg_var.set(str(n_avg))
        out = output_dir.get() or str(Path(inp).parent)
        na_val = float(na_var.get()) if na_var.get().strip() else None
        fluor = fluorophore_var.get().strip() or None

        def status(msg):
            def update():
                status_var.set(msg)
                app.update_idletasks()
            app.after(0, update)

        cellpose_path = (cellpose_model_var.get().strip() or None) if mode_var.get() == 'cellpose' else None
        fit_mode = fit_mode_var.get()
        fit_gaussian = fit_mode in ('1d', 'both')
        fit_3d = fit_mode in ('3d', 'both')

        def run_thread():
            try:
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
                    trackpy_diameter=tp_diameter,
                    trackpy_minmass=tp_minmass,
                    trackpy_separation=tp_separation,
                    cellpose_min_size=cp_min_size,
                    cellpose_flow_threshold=cp_flow,
                    num_beads_avg=n_avg,
                    sample_fraction=min(100, max(1, sample_frac)),
                    center_mode=center_mode_var.get(),
                    run_on_main=_run_on_main_and_wait,
                    run_settings=settings,
                )
            finally:
                app.after(0, lambda: analyze_btn.configure(state="normal"))

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
            'trackpy_diameter': tp_diameter,
            'trackpy_minmass': tp_minmass,
            'trackpy_separation': tp_separation,
            'cellpose_min_size': cp_min_size,
            'cellpose_flow_threshold': cp_flow,
            'num_beads_avg': n_avg,
            'sample_fraction': min(100, max(1, sample_frac)),
            'center_mode': center_mode_var.get(),
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
        analyze_btn.configure(state="disabled")
        threading.Thread(target=run_thread, daemon=True).start()

    def _apply_window_geometry():
        """Apply width/height based on current UI visibility toggles."""
        target_h = HEIGHT_EXPANDED if show_advanced.get() else HEIGHT_COMPACT
        base_w = WIDTH_ADVANCED if show_advanced.get() else WIDTH_BASE
        app.geometry(f"{base_w}x{target_h}")
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
    file_results_label.pack(anchor="w", padx=12, pady=(6, 2))
    frame_file = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_file.pack(fill="x", padx=12, pady=(0, 4))
    input_file_label = ctk.CTkLabel(frame_file, text="Input file:")
    input_file_label.pack(anchor="w")
    entry_input = ctk.CTkEntry(frame_file, textvariable=input_file, width=400)
    entry_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
    browse_input_btn = ctk.CTkButton(frame_file, text="Browse", width=80, command=browse_input)
    browse_input_btn.pack(side="left")

    output_dir_label = ctk.CTkLabel(left_frame, text="Results directory:")
    output_dir_label.pack(anchor="w", padx=12, pady=(2, 2))
    frame_out = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_out.pack(fill="x", padx=12, pady=(0, 6))
    entry_out = ctk.CTkEntry(frame_out, textvariable=output_dir, width=400)
    entry_out.pack(side="left", fill="x", expand=True, padx=(0, 6))
    browse_out_btn = ctk.CTkButton(frame_out, text="Browse", width=80, command=browse_output)
    browse_out_btn.pack(side="left")
    ctk.CTkFrame(left_frame, height=1, fg_color=("gray78", "gray28")).pack(fill="x", padx=(18, 18), pady=(4, 8))

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
    ctk.CTkFrame(left_frame, height=1, fg_color=("gray78", "gray28")).pack(fill="x", padx=(18, 18), pady=(4, 8))

    # Advanced Options Checkbox (packed at bottom, below status)
    show_advanced_cb = ctk.CTkCheckBox(left_frame, text="Show Advanced Options", variable=show_advanced,
                                       font=ctk.CTkFont(weight="bold", size=13))

    # Mode
    detection_mode_label = ctk.CTkLabel(left_frame, text="Bead detection", font=ctk.CTkFont(weight="bold"))
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
    ctk.CTkFrame(left_frame, height=1, fg_color=("gray78", "gray28")).pack(fill="x", padx=(18, 18), pady=(4, 8))

    # Fitting method
    sub = ctk.CTkFont(size=12, weight="bold")
    fitting_label = ctk.CTkLabel(left_frame, text="Fitting method", font=sub)
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

    extraction_avg_label = ctk.CTkLabel(left_frame, text="Extraction & averaging", font=sub)
    extraction_avg_label.pack(anchor="w", padx=12, pady=(4, 2))

    # General numeric params (Extraction & averaging: Box width only)
    frame_gen = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_gen.pack(fill="x", **pad)
    box_entry_var = ctk.StringVar(value=str(prev.get('box_size', '15')))
    box_size_label = ctk.CTkLabel(frame_gen, text="Box width (px):")
    box_size_label.pack(side="left", padx=(0, 8))
    box_entry = ctk.CTkEntry(frame_gen, textvariable=box_entry_var, width=50)
    box_entry.pack(side="left")
    center_mode_label = ctk.CTkLabel(frame_gen, text="Center mode:")
    center_mode_label.pack(side="left", padx=(16, 8))
    center_mode_menu = ctk.CTkOptionMenu(
        frame_gen,
        values=["peak", "centroid", "radial"],
        variable=center_mode_var,
        width=110,
    )
    center_mode_menu.pack(side="left")

    # Background
    background_label = ctk.CTkLabel(left_frame, text="Background subtraction", font=sub)
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
    sample_fraction_label = ctk.CTkLabel(frame_qa_cb, text="Analyze % of beads (1-100):")
    sample_fraction_label.pack(side="left", padx=(0, 8))
    sample_fraction_entry = ctk.CTkEntry(frame_qa_cb, textvariable=sample_fraction_var, width=50)
    sample_fraction_entry.pack(side="left")
    frame_qa_reject = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_qa_reject.pack(fill="x", **pad)
    qa_reject_cb = ctk.CTkCheckBox(frame_qa_reject, text="Auto-reject low QA beads", variable=qa_auto_reject_var)
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

    frame_sample = ctk.CTkFrame(left_frame, fg_color="transparent")
    frame_sample.pack(fill="x", **pad)
    num_beads_label = ctk.CTkLabel(frame_sample, text="Percentage of beads to avg (1-100):")
    num_beads_label.pack(side="left", padx=(0, 8))
    num_beads_entry = ctk.CTkEntry(frame_sample, textvariable=num_beads_avg_var, width=50)
    num_beads_entry.pack(side="left")

    # --- Advanced options container
    advanced_container = ctk.CTkFrame(left_frame, fg_color="transparent")
    advanced_container.pack(fill="x")

    adv_left = ctk.CTkFrame(advanced_container, fg_color="transparent")
    adv_left.pack(side="left", fill="y", anchor="nw", padx=(0, 12))
    adv_right = ctk.CTkFrame(advanced_container, fg_color="transparent")
    adv_right.pack(side="left", fill="y", anchor="nw")

    # Left column: Detection review + Trackpy + StarDist options
    detection_header = ctk.CTkLabel(adv_left, text="Detection review", font=ctk.CTkFont(weight="bold"))
    detection_header.pack(anchor="w", **pad)
    frame_det = ctk.CTkFrame(adv_left, fg_color="transparent")
    frame_det.pack(fill="x", **pad)
    review_detection_cb = ctk.CTkCheckBox(frame_det, text="Review detection overlay", variable=review_detection_var)
    review_detection_cb.pack(side="left", padx=(0, 16))

    trackpy_header = ctk.CTkLabel(adv_left, text="Trackpy options", font=ctk.CTkFont(weight="bold"))
    trackpy_header.pack(anchor="w", **pad)
    frame_trackpy = ctk.CTkFrame(adv_left, fg_color="transparent")
    frame_trackpy.pack(fill="x", **pad)
    tp_diameter_label = ctk.CTkLabel(frame_trackpy, text="Diameter (px):")
    tp_diameter_label.pack(side="left", padx=(0, 8))
    tp_diameter_entry = ctk.CTkEntry(frame_trackpy, textvariable=trackpy_diameter_var, width=50)
    tp_diameter_entry.pack(side="left", padx=(0, 16))
    tp_minmass_label = ctk.CTkLabel(frame_trackpy, text="Minmass:")
    tp_minmass_label.pack(side="left", padx=(0, 8))
    tp_minmass_entry = ctk.CTkEntry(frame_trackpy, textvariable=trackpy_minmass_var, width=70)
    tp_minmass_entry.pack(side="left")
    frame_trackpy_sep = ctk.CTkFrame(adv_left, fg_color="transparent")
    frame_trackpy_sep.pack(fill="x", **pad)
    tp_sep_label = ctk.CTkLabel(frame_trackpy_sep, text="Separation (px, optional):")
    tp_sep_label.pack(side="left", padx=(0, 8))
    tp_sep_entry = ctk.CTkEntry(frame_trackpy_sep, textvariable=trackpy_separation_var, width=70)
    tp_sep_entry.pack(side="left")

    stardist_header = ctk.CTkLabel(adv_left, text="StarDist options", font=ctk.CTkFont(weight="bold"))
    stardist_header.pack(anchor="w", **pad)
    frame_stardist = ctk.CTkFrame(adv_left, fg_color="transparent")
    frame_stardist.pack(fill="x", **pad)
    blob_fallback_cb = ctk.CTkCheckBox(frame_stardist, text="Blob fallback", variable=blob_fallback_var)
    blob_fallback_cb.pack(side="left")

    # Right column: Cellpose options
    cellpose_header = ctk.CTkLabel(adv_right, text="Cellpose options (requires model file)", font=ctk.CTkFont(weight="bold"))
    cellpose_header.pack(anchor="w", **pad)
    frame_cp_model = ctk.CTkFrame(adv_right, fg_color="transparent")
    frame_cp_model.pack(fill="x", **pad)
    cp_model_label = ctk.CTkLabel(frame_cp_model, text="Model file:")
    cp_model_label.pack(side="left", padx=(0, 8))
    cp_model_entry = ctk.CTkEntry(frame_cp_model, textvariable=cellpose_model_var, width=300)
    cp_model_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    cp_model_browse = ctk.CTkButton(frame_cp_model, text="Browse", width=80, command=browse_cellpose_model)
    cp_model_browse.pack(side="left")
    cp_env_hint = ctk.CTkLabel(adv_right, text="(or set FWHM_CELLPOSE_MODEL env var)", text_color="gray", font=ctk.CTkFont(size=11))
    cp_env_hint.pack(anchor="w", padx=(12, 0))
    frame_cp_checks = ctk.CTkFrame(adv_right, fg_color="transparent")
    frame_cp_checks.pack(fill="x", **pad)
    cp_do_3d_cb = ctk.CTkCheckBox(frame_cp_checks, text="Native 3D", variable=cellpose_do_3d_var)
    cp_do_3d_cb.pack(side="left", padx=(0, 16))
    cp_skip_review_cb = ctk.CTkCheckBox(frame_cp_checks, text="Skip review", variable=skip_cellpose_review_var)
    cp_skip_review_cb.pack(side="left")
    frame_cp_aniso = ctk.CTkFrame(adv_right, fg_color="transparent")
    frame_cp_aniso.pack(fill="x", **pad)
    cp_aniso_label = ctk.CTkLabel(frame_cp_aniso, text="Anisotropy (z/xy):")
    cp_aniso_label.pack(side="left", padx=(0, 8))
    cp_aniso_entry = ctk.CTkEntry(frame_cp_aniso, textvariable=anisotropy_var, width=70)
    cp_aniso_entry.pack(side="left")
    frame_cp_params = ctk.CTkFrame(adv_right, fg_color="transparent")
    frame_cp_params.pack(fill="x", **pad)
    cp_minsize_label = ctk.CTkLabel(frame_cp_params, text="Min size (px):")
    cp_minsize_label.pack(side="left", padx=(0, 8))
    cp_minsize_entry = ctk.CTkEntry(frame_cp_params, textvariable=cellpose_min_size_var, width=50)
    cp_minsize_entry.pack(side="left", padx=(0, 16))
    cp_flow_label = ctk.CTkLabel(frame_cp_params, text="Flow threshold:")
    cp_flow_label.pack(side="left", padx=(0, 8))
    cp_flow_entry = ctk.CTkEntry(frame_cp_params, textvariable=cellpose_flow_threshold_var, width=50)
    cp_flow_entry.pack(side="left")

    # --- Mode-dependent enable/disable & advanced toggle ---
    _default_label_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
    _disabled_color = "gray40"
    _sections = {
        'detection': {
            'header': detection_header,
            'widgets': [review_detection_cb],
            'labels': [],
        },
        'stardist': {
            'header': stardist_header,
            'widgets': [blob_fallback_cb],
            'labels': [],
        },
        'trackpy': {
            'header': trackpy_header,
            'widgets': [tp_diameter_entry, tp_minmass_entry, tp_sep_entry],
            'labels': [tp_diameter_label, tp_minmass_label, tp_sep_label],
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
    _advanced_mode_buttons = [stardist_rb, cellpose_rb]

    def _set_section_state(name, enabled):
        sec = _sections[name]
        state = "normal" if enabled else "disabled"
        color = _default_label_color if enabled else _disabled_color
        sec['header'].configure(text_color=color)
        for w in sec['widgets']:
            w.configure(state=state)
        for lbl in sec['labels']:
            lbl.configure(text_color=color)

    def _toggle_advanced_options(*_args):
        """Show/hide advanced options based on checkbox state."""
        is_advanced = show_advanced.get()

        # Show/hide StarDist and Cellpose mode buttons
        for btn in _advanced_mode_buttons:
            if is_advanced:
                btn.pack(side="left", padx=(0, 10) if btn == stardist_rb else 0)
            else:
                btn.pack_forget()
                # If user was on advanced mode, switch to blob
                if mode_var.get() in ('stardist', 'cellpose'):
                    mode_var.set('blob')

        # Show/hide the entire advanced container as a single unit.
        # Use before= to keep it above the checkbox / run button.
        if is_advanced:
            advanced_container.pack(fill="x", before=show_advanced_cb)
        else:
            advanced_container.pack_forget()

        _apply_window_geometry()

    def _on_mode_change(*_args):
        mode = mode_var.get()
        is_advanced = show_advanced.get()

        # Only apply mode-dependent logic if advanced mode is on
        if is_advanced:
            detection_on = mode in ('blob', 'trackpy', 'stardist')
            _set_section_state('detection', detection_on)
            trackpy_on = mode == 'trackpy'
            _set_section_state('trackpy', trackpy_on)
            stardist_on = mode == 'stardist'
            _set_section_state('stardist', stardist_on)
            cellpose_on = mode == 'cellpose'
            _set_section_state('cellpose', cellpose_on)
            cp_env_hint.configure(text_color="gray" if cellpose_on else _disabled_color)

    show_advanced.trace_add("write", _toggle_advanced_options)
    mode_var.trace_add("write", _on_mode_change)
    show_advanced_cb.pack(anchor="w", padx=12, pady=(16, 6))

    # Run button (extra 10px above to push down)
    analyze_btn = ctk.CTkButton(left_frame, text="Analyze beads", command=run, height=36, font=ctk.CTkFont(weight="bold"))
    analyze_btn.pack(padx=12, pady=(16, 6))

    # Status
    ctk.CTkLabel(left_frame, textvariable=status_var, text_color="gray").pack(**pad)

    # Docs popout window (separate toplevel, does not affect main window width)
    docs_window = ctk.CTkToplevel(app)
    docs_window.title("Bead Analyzer – Docs")
    docs_window.geometry(f"{DOCS_PANEL_WIDTH}x{HEIGHT_COMPACT}")
    docs_window.withdraw()  # start hidden
    docs_window.protocol("WM_DELETE_WINDOW", lambda: _toggle_docs_panel())

    docs_header = ctk.CTkFrame(docs_window, fg_color="transparent")
    docs_header.pack(fill="x", padx=12, pady=(12, 8))
    ctk.CTkLabel(docs_header, text="Settings and Use Cases", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")

    docs_tabs = ctk.CTkTabview(docs_window)
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

    def _add_setting_doc(key, title, body, is_child=False):
        section = ctk.CTkFrame(settings_scroll)
        section.pack(fill="x", padx=6, pady=(0, 10))
        default_text_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
        title_text = f"  - {title}" if is_child else title
        title_lbl = ctk.CTkLabel(section, text=title_text, font=ctk.CTkFont(weight="bold"), text_color=default_text_color)
        title_lbl.pack(anchor="w", padx=(18, 8) if is_child else (8, 8), pady=(8, 2))
        # Derive wraplength from panel width minus padding/indentation
        body_wrap = (DOCS_PANEL_WIDTH - 135) if is_child else (DOCS_PANEL_WIDTH - 110)
        body_lbl = ctk.CTkLabel(section, text=body, justify="left", wraplength=body_wrap, text_color=default_text_color)
        body_lbl.pack(anchor="w", padx=(38, 12) if is_child else (26, 12), pady=(0, 8))
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

    _add_setting_doc("section_file_results", "Input file and Results",
        "Select the 3D bead image stack to analyze and choose where CSV tables, "
        "figures, detection overlays, and settings JSON are saved.")
    _add_setting_doc("input_file", "Input file",
        "Path to a TIFF or OME-TIFF 3D bead stack (ZYX or CZYX). "
        "Multi-channel files are supported; select the channel below.",
        is_child=True)
    _add_setting_doc("output_dir", "Results directory",
        "Folder for all outputs: per-bead CSV, summary statistics, average-bead "
        "profile plots, detection overlay images, and a copy of the run settings. "
        "Defaults to the same folder as the input file.",
        is_child=True)
    _add_setting_doc("section_experimental", "Experimental parameters",
        "Microscope acquisition metadata that converts pixel measurements into "
        "physical units. Incorrect values will scale every FWHM result.")
    _add_setting_doc("scale", "Scaling (um/pix): XY, Z",
        "Lateral (XY) and axial (Z) pixel sizes in microns. Find these in your "
        "microscope software (NIS-Elements, ZEN, FluoView) or via ImageJ "
        "Image > Show Info. Typical values: 0.065\u20130.26 um for XY, 0.1\u20132.0 um for Z.",
        is_child=True)
    _add_setting_doc("channel", "Channel",
        "Zero-based channel index for multi-channel (CZYX) stacks. "
        "The dropdown auto-updates when you select an input file. "
        "For single-channel stacks, leave at 0.",
        is_child=True)
    _add_setting_doc("section_detection_mode", "Bead detection",
        "Choose how candidate bead centers are found before FWHM measurement. "
        "Each mode suits different bead sizes, densities, and image quality.")
    _add_setting_doc("mode_manual", "Manual",
        "Right-click on beads interactively in a MIP window; press Escape when "
        "done. Best for sparse slides, edge cases, or targeted re-analysis of "
        "specific beads.",
        is_child=True)
    _add_setting_doc("mode_blob", "Blob",
        "Gaussian-smoothed local-maximum detector. Fast, no dependencies beyond "
        "scipy. Recommended default, especially for small beads (< 10 px diameter).",
        is_child=True)
    _add_setting_doc("mode_trackpy", "Trackpy",
        "Bandpass filter + centroid refinement. Handles non-uniform backgrounds and "
        "intensity gradients better than Blob. Good for low-NA objectives or dim beads.",
        is_child=True)
    _add_setting_doc("mode_stardist", "StarDist (advanced)",
        "Pre-trained deep-learning star-convex polygon detector. Works best when "
        "beads are ~15+ px in diameter and well-separated. Requires the stardist package.",
        is_child=True)
    _add_setting_doc("mode_cellpose", "Cellpose (advanced)",
        "Instance segmentation with a user-trained Cellpose model. Handles dense "
        "or overlapping bead fields where classical detectors struggle. Requires a "
        "trained model file (see Cellpose options below).",
        is_child=True)
    _add_setting_doc("section_fitting", "Fitting method",
        "Controls how FWHM is extracted from each bead. Gaussian fitting gives "
        "sub-pixel precision; prominence mode is faster but less accurate.")
    _add_setting_doc("fit_1d", "1D Gaussian",
        "Fits a 1D Gaussian curve independently to X, Y, and Z intensity profiles "
        "through the bead center. Fast and gives sub-pixel FWHM. Good default for "
        "most use cases.",
        is_child=True)
    _add_setting_doc("fit_3d", "3D Gaussian",
        "Fits an axis-aligned 3D Gaussian (separate sigma_x, sigma_y, sigma_z) to "
        "the entire bead volume. More accurate for asymmetric PSFs but significantly "
        "slower. No rotation parameters.",
        is_child=True)
    _add_setting_doc("fit_both", "Both",
        "Runs 1D and 3D fitting sequentially. Use this to compare methods or for "
        "thorough publication-quality analysis. Results for both appear in the CSV.",
        is_child=True)
    _add_setting_doc("fit_none", "No fit",
        "Measures peak width at half-prominence using scipy peak finding. No Gaussian "
        "model is fitted. Fastest option; useful for quick screening or when fitting "
        "fails on noisy data.",
        is_child=True)
    _add_setting_doc("robust_fit", "Robust fit",
        "Uses soft L1 (Huber-like) loss in the Gaussian curve_fit optimizer instead "
        "of least-squares. Downweights outlier pixels from clipped edges or nearby "
        "beads. Recommended ON unless beads are very clean and isolated.",
        is_child=True)
    _add_setting_doc("section_extraction_avg", "Extraction & averaging",
        "Controls how bead sub-volumes are cropped and which beads contribute to "
        "the averaged bead profile and composite outputs.")
    _add_setting_doc("box_size", "Box width (px)",
        "Full width of the square crop around each bead center (half_box = box_size // 2 "
        "per side). Default 15 px. Increase for large beads or high-mag objectives; "
        "decrease if beads are densely packed to avoid overlap.",
        is_child=True)
    _add_setting_doc("center_mode", "Center mode",
        "How XY center is refined after initial detection. peak: brightest local voxel "
        "(default). centroid: intensity-weighted center. radial: radial-symmetry style "
        "centering, often better for annular/hollow-looking beads.",
        is_child=True)
    _add_setting_doc("num_beads_avg", "Percentage of beads to avg (1-100)",
        "Number of beads used for the average bead profile. Beads are ranked by "
        "distance from the median Z-FWHM, so the most representative beads are "
        "selected first. Minimum: 1. Default: 20.",
        is_child=True)
    _add_setting_doc("section_background", "Background subtraction",
        "Remove baseline signal before measuring peak widths. Important when "
        "background fluorescence would artificially broaden FWHM estimates.")
    _add_setting_doc("subtract_background", "Subtract global background",
        "Opens an interactive window where you draw an ROI over a background region. "
        "Use the right mouse button to click and drag to draw a background region. "
        "The mean intensity of that ROI is subtracted from the entire stack before "
        "analysis.",
        is_child=True)
    _add_setting_doc("local_background", "Local background",
        "Estimates per-bead background from an annulus around each bead (inner radius = "
        "half_box, outer radius = inner + half_box/2). The annulus median is subtracted "
        "from each bead's profile. Best for light-sheet data or images with spatially "
        "varying haze.",
        is_child=True)
    _add_setting_doc("section_quality_output", "Quality & output",
        "Filter out poor measurements and save diagnostic plots for per-bead inspection.")
    _add_setting_doc("save_diagnostics", "Save bead diagnostics",
        "Writes a diagnostic figure for every bead to a bead_diagnostics/ subfolder. "
        "Each figure shows the XYZ intensity profiles, fitted curves, and QA metrics. "
        "Useful for troubleshooting unexpected FWHM values.",
        is_child=True)
    _add_setting_doc("qa_auto_reject", "Auto-reject low QA beads",
        "Automatically excludes beads that fall below the SNR or symmetry thresholds. "
        "Rejected beads are logged in the CSV but excluded from averages and summary "
        "statistics.",
        is_child=True)
    _add_setting_doc("qa_snr", "QA min SNR",
        "Minimum signal-to-noise ratio (peak intensity / noise floor) to accept a "
        "bead. Default: 3.0. Increase if you see dim artifacts being counted as beads.",
        is_child=True)
    _add_setting_doc("qa_sym", "QA min symmetry",
        "Z-profile symmetry score from 0 (asymmetric) to 1 (perfectly symmetric). "
        "Computed as 1 minus the mean absolute difference between normalized left and "
        "right halves of the Z profile. Default: 0.6.",
        is_child=True)
    _add_setting_doc("sample_fraction", "Analyze % of beads (1-100)",
        "Random uniform sample of detected beads to analyze. 100 = analyze all. "
        "Lower values (e.g. 20 or 50) speed up runs while giving a representative sample. "
        "CSV and bead diagnostics only include the analyzed (sampled) beads.",
        is_child=True)
    _add_setting_doc("section_advanced", "Detection review",
        "Visual verification of automatic detections before committing to full analysis.")
    _add_setting_doc("show_advanced", "Show Advanced Options",
        "Reveals Trackpy/StarDist/Cellpose configuration sections. "
        "Also widens the window to a two-column advanced layout.",
        is_child=True)
    _add_setting_doc("review_detection", "Review detection overlay",
        "Displays a MIP with yellow circle markers at each detected bead center. "
        "Press 'y' to accept and continue, or 'n' to abort and retry with different "
        "settings.",
        is_child=True)
    _add_setting_doc("section_stardist", "StarDist options",
        "Controls specific to the StarDist deep-learning detection mode.")
    _add_setting_doc("section_trackpy", "Trackpy options",
        "Controls for Trackpy bandpass + centroid detection. Important for large beads "
        "or high-magnification datasets where default small-bead values underperform.")
    _add_setting_doc("trackpy_diameter", "Trackpy Diameter (px)",
        "Expected feature diameter in pixels (odd integer). For large resolved beads, "
        "set this near the apparent bead diameter in pixels.",
        is_child=True)
    _add_setting_doc("trackpy_minmass", "Trackpy Minmass",
        "Minimum integrated brightness for accepted features. Increase to suppress "
        "dim/background detections; decrease if true beads are missed.",
        is_child=True)
    _add_setting_doc("trackpy_separation", "Trackpy Separation (px, optional)",
        "Minimum spacing between feature centers. Leave blank for Trackpy default "
        "(diameter + 1). Increase to avoid splitting one large bead into multiple detections.",
        is_child=True)
    _add_setting_doc("blob_fallback", "Blob fallback",
        "If StarDist detects zero beads, automatically falls back to the Blob "
        "detector instead of aborting. Useful as a safety net when StarDist "
        "confidence thresholds are too strict.",
        is_child=True)
    _add_setting_doc("section_cellpose", "Cellpose options (requires model file)",
        "Model path, inference settings, and post-processing controls for "
        "Cellpose-based bead segmentation.")
    _add_setting_doc("cellpose_model", "Cellpose model file",
        "Path to your trained Cellpose model. Alternatively, set the "
        "FWHM_CELLPOSE_MODEL environment variable before launching. "
        "See the README for training instructions using annotate_beads.py "
        "and train_cellpose.py.",
        is_child=True)
    _add_setting_doc("cellpose_native_3d", "Cellpose Native 3D",
        "Runs Cellpose on the full 3D volume instead of a 2D maximum intensity "
        "projection. More accurate for overlapping beads in Z but significantly "
        "slower and uses more memory.",
        is_child=True)
    _add_setting_doc("cellpose_skip_review", "Cellpose Skip review",
        "Skips the interactive mask overlay window where you approve/reject "
        "detected masks. Enable for batch runs or when you trust the model output.",
        is_child=True)
    _add_setting_doc("anisotropy", "Anisotropy (z/xy)",
        "Ratio of Z step size to XY pixel size (e.g., 2.0/0.26 = 7.7). "
        "Used by Cellpose 3D to rescale the volume so voxels appear isotropic. "
        "Leave blank to let Cellpose use its default.",
        is_child=True)
    _add_setting_doc("cellpose_min_size", "Cellpose Min size",
        "Minimum mask area in pixels. Masks smaller than this are discarded as "
        "noise. Default: 3 px. Increase if you see many tiny false detections.",
        is_child=True)
    _add_setting_doc("cellpose_flow_threshold", "Cellpose Flow threshold",
        "Maximum flow error for mask acceptance. Lower values are stricter and "
        "reject more uncertain masks. Default: 0.4. Reduce to 0.2\u20130.3 for "
        "cleaner but potentially fewer detections.",
        is_child=True)

    for title, body in [
        ("Fast screening",
         "Blob detection + 1D Gaussian + Robust fit ON. Leave QA defaults as-is. "
         "Good for a quick first look at a new bead slide. Takes seconds per stack."),
        ("Publication-quality analysis",
         "Use 'Both' fitting to get 1D and 3D FWHM side by side. Enable 'Save bead "
         "diagnostics' to inspect every bead. Turn on 'Auto-reject low QA beads' with SNR >= 5 "
         "and symmetry >= 0.7 for strict filtering. Report median and IQR from the CSV."),
        ("Light-sheet / uneven background",
         "Enable 'Local background' so each bead is baselined from its own annulus "
         "neighborhood. Combine with 'Review detection overlay' to verify that the "
         "detector is not picking up background blobs. Consider Trackpy if Blob "
         "over-detects on intensity gradients."),
        ("Dense / overlapping beads",
         "Enable 'Show Advanced Options' and choose Cellpose with your trained model. "
         "Set Min size to filter debris and lower Flow threshold (0.2\u20130.3) for stricter "
         "mask quality. Use Native 3D if beads overlap in Z. Reduce box width if crops "
         "overlap neighboring beads."),
        ("Small sub-resolution beads",
         "Use Blob or Trackpy (not StarDist/Cellpose, which need ~15+ px beads). "
         "Keep box width small (7\u201311 px). 1D Gaussian is usually sufficient; 3D Gaussian "
         "may fail to converge on very small volumes."),
        ("Validating a new microscope",
         "Image a standard bead slide (e.g., 100 nm TetraSpeck). Run 'Both' fitting with "
         "'Save diagnostics' ON. Compare measured XY and Z FWHM against the theoretical "
         "diffraction limit for your objective NA and wavelength."),
        ("Batch processing (CLI)",
         "Use the GUI for initial parameter tuning, then copy the saved "
         "bead_analyzer_settings.json to script your CLI runs. The CLI supports "
         "all GUI options plus additional controls (blob_sigma, z_range, detrending). "
         "See CLI.md for details."),
    ]:
        box = ctk.CTkFrame(use_cases_scroll)
        box.pack(fill="x", padx=6, pady=(0, 10))
        ctk.CTkLabel(box, text=title, font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(8, 2))
        ctk.CTkLabel(box, text=body, justify="left", wraplength=DOCS_PANEL_WIDTH - 100).pack(anchor="w", padx=(26, 12), pady=(0, 8))
        ctk.CTkFrame(box, height=1, fg_color=("gray78", "gray28")).pack(fill="x", padx=(22, 14), pady=(0, 2))

    def _toggle_docs_panel():
        if docs_open_var.get():
            docs_window.withdraw()
            docs_open_var.set(False)
            docs_toggle_btn.configure(text="Docs ▸")
        else:
            app.update_idletasks()
            x = app.winfo_x() + app.winfo_width() + 8
            y = app.winfo_y()
            docs_window.geometry(f"{DOCS_PANEL_WIDTH}x{app.winfo_height()}+{x}+{y}")
            docs_window.deiconify()
            docs_open_var.set(True)
            docs_toggle_btn.configure(text="Docs ◂")

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
        # _parent_canvas is a CustomTkinter internal; fall back gracefully if
        # a future CTk version renames or removes it.
        canvas = getattr(settings_scroll, "_parent_canvas", None)
        anchor = setting_anchors[key]["section"]
        if canvas is None:
            _highlight_setting(key)
            return
        try:
            bbox = canvas.bbox("all")
        except Exception:
            _highlight_setting(key)
            return
        if not bbox:
            _highlight_setting(key)
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
    _bind_hover(manual_rb, "mode_manual")
    _bind_hover(blob_rb, "mode_blob")
    _bind_hover(trackpy_rb, "mode_trackpy")
    _bind_hover(stardist_rb, "mode_stardist")
    _bind_hover(cellpose_rb, "mode_cellpose")
    _bind_hover(fitting_label, "section_fitting")
    _bind_hover(fit_1d_rb, "fit_1d")
    _bind_hover(fit_3d_rb, "fit_3d")
    _bind_hover(fit_both_rb, "fit_both")
    _bind_hover(fit_none_rb, "fit_none")
    _bind_hover(robust_fit_cb, "robust_fit")
    _bind_hover(extraction_avg_label, "section_extraction_avg")
    _bind_hover(box_size_label, "box_size")
    _bind_hover(box_entry, "box_size")
    _bind_hover(center_mode_label, "center_mode")
    _bind_hover(center_mode_menu, "center_mode")
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
    _bind_hover(sample_fraction_label, "sample_fraction")
    _bind_hover(sample_fraction_entry, "sample_fraction")
    _bind_hover(detection_header, "section_advanced")
    _bind_hover(review_detection_cb, "review_detection")
    _bind_hover(trackpy_header, "section_trackpy")
    _bind_hover(tp_diameter_label, "trackpy_diameter")
    _bind_hover(tp_diameter_entry, "trackpy_diameter")
    _bind_hover(tp_minmass_label, "trackpy_minmass")
    _bind_hover(tp_minmass_entry, "trackpy_minmass")
    _bind_hover(tp_sep_label, "trackpy_separation")
    _bind_hover(tp_sep_entry, "trackpy_separation")
    _bind_hover(stardist_header, "section_stardist")
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
