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
                  cellpose_min_size=3, cellpose_flow_threshold=0.4):
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
        )
        rej_msg = f" ({len(rejected)} rejected)" if rejected else ""
        status_callback(f"Done. {len(results)} beads analyzed{rej_msg}.")
    except Exception as e:
        status_callback(f"Error: {e}")


def main():
    if ctk is None:
        print("Install customtkinter: pip install customtkinter")
        return 1

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = ctk.CTk()
    app.title("Bead Analyzer")
    app.geometry("560x760")
    app.minsize(450, 600)

    # Variables
    input_file = ctk.StringVar()
    output_dir = ctk.StringVar()
    scale_xy = ctk.StringVar(value="0.26")
    scale_z = ctk.StringVar(value="2.0")
    channel_var = ctk.StringVar(value="0")
    na_var = ctk.StringVar(value="")
    fluorophore_var = ctk.StringVar(value="")
    mode_var = ctk.StringVar(value="manual")
    cellpose_model_var = ctk.StringVar(value="")
    status_var = ctk.StringVar(value="Ready")
    fit_gaussian = ctk.BooleanVar(value=True)
    subtract_background = ctk.BooleanVar(value=False)
    review_detection_var = ctk.BooleanVar(value=True)
    skip_cellpose_review_var = ctk.BooleanVar(value=False)
    qa_snr_var = ctk.StringVar(value="3.0")
    qa_sym_var = ctk.StringVar(value="0.6")
    fit_3d_var = ctk.BooleanVar(value=False)
    save_diagnostics_var = ctk.BooleanVar(value=False)
    qa_auto_reject_var = ctk.BooleanVar(value=False)
    cellpose_do_3d_var = ctk.BooleanVar(value=False)
    anisotropy_var = ctk.StringVar(value="")
    blob_fallback_var = ctk.BooleanVar(value=False)
    local_background_var = ctk.BooleanVar(value=False)
    robust_fit_var = ctk.BooleanVar(value=False)
    cellpose_min_size_var = ctk.StringVar(value="3")
    cellpose_flow_threshold_var = ctk.StringVar(value="0.4")

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

    def browse_output():
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            output_dir.set(path)

    def browse_cellpose_model():
        path = filedialog.askopenfilename(title="Select Cellpose model file")
        if path:
            cellpose_model_var.set(path)

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
        except ValueError:
            messagebox.showerror("Error", "Scale XY, Z, channel, QA, box size, anisotropy, and Cellpose params must be valid numbers.")
            return
        out = output_dir.get() or str(Path(inp).parent)
        na_val = float(na_var.get()) if na_var.get().strip() else None
        fluor = fluorophore_var.get().strip() or None

        def status(msg):
            status_var.set(msg)
            app.update_idletasks()

        cellpose_path = (cellpose_model_var.get().strip() or None) if mode_var.get() == 'cellpose' else None

        def run_thread():
            _run_analysis(
                inp, out, mode_var.get(), sx, sz, na_val, fluor,
                bx, fit_gaussian.get(), subtract_background.get(),
                status,
                cellpose_model_path=cellpose_path,
                channel=ch,
                review_detection=review_detection_var.get(),
                skip_cellpose_review=skip_cellpose_review_var.get(),
                qa_min_snr=qa_snr,
                qa_min_symmetry=qa_sym,
                fit_3d=fit_3d_var.get(),
                save_diagnostics=save_diagnostics_var.get(),
                qa_auto_reject=qa_auto_reject_var.get(),
                cellpose_do_3d=cellpose_do_3d_var.get(),
                anisotropy=aniso,
                use_blob_fallback=blob_fallback_var.get(),
                local_background=local_background_var.get(),
                robust_fit=robust_fit_var.get(),
                cellpose_min_size=cp_min_size,
                cellpose_flow_threshold=cp_flow,
            )

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
    ctk.CTkLabel(frame_params, text="Scale XY (µm/px):").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)
    ctk.CTkEntry(frame_params, textvariable=scale_xy, width=80).grid(row=0, column=1, padx=(0, 16), pady=2)
    ctk.CTkLabel(frame_params, text="Scale Z (µm/px):").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=2)
    ctk.CTkEntry(frame_params, textvariable=scale_z, width=80).grid(row=1, column=1, padx=(0, 16), pady=2)
    ctk.CTkLabel(frame_params, text="NA:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=2)
    ctk.CTkEntry(frame_params, textvariable=na_var, width=80).grid(row=2, column=1, padx=(0, 16), pady=2)
    ctk.CTkLabel(frame_params, text="Fluorophore:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=2)
    ctk.CTkEntry(frame_params, textvariable=fluorophore_var, width=120).grid(row=3, column=1, padx=(0, 16), pady=2)
    ctk.CTkLabel(frame_params, text="Channel (0-based):").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=2)
    ctk.CTkEntry(frame_params, textvariable=channel_var, width=80).grid(row=4, column=1, padx=(0, 16), pady=2)

    # Mode
    ctk.CTkLabel(app, text="Detection mode", font=ctk.CTkFont(weight="bold")).pack(anchor="w", **pad)
    frame_mode = ctk.CTkFrame(app, fg_color="transparent")
    frame_mode.pack(fill="x", **pad)
    ctk.CTkRadioButton(frame_mode, text="Manual", variable=mode_var, value="manual").pack(side="left", padx=(0, 16))
    ctk.CTkRadioButton(frame_mode, text="StarDist", variable=mode_var, value="stardist").pack(side="left", padx=(0, 16))
    ctk.CTkRadioButton(frame_mode, text="Cellpose", variable=mode_var, value="cellpose").pack(side="left")

    # Cellpose model
    frame_cellpose = ctk.CTkFrame(app, fg_color="transparent")
    frame_cellpose.pack(fill="x", **pad)
    ctk.CTkLabel(frame_cellpose, text="Cellpose model (file):").pack(anchor="w")
    frame_cp = ctk.CTkFrame(app, fg_color="transparent")
    frame_cp.pack(fill="x", **pad)
    ctk.CTkEntry(frame_cp, textvariable=cellpose_model_var, width=350).pack(side="left", fill="x", expand=True, padx=(0, 6))
    ctk.CTkButton(frame_cp, text="Browse", width=80, command=browse_cellpose_model).pack(side="left")
    ctk.CTkLabel(app, text="(or set FWHM_CELLPOSE_MODEL env var)", text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=(12, 0))

    # Options
    ctk.CTkLabel(app, text="Options", font=ctk.CTkFont(weight="bold")).pack(anchor="w", **pad)
    frame_opts1 = ctk.CTkFrame(app, fg_color="transparent")
    frame_opts1.pack(fill="x", **pad)
    ctk.CTkCheckBox(frame_opts1, text="Fit Gaussian (1D)", variable=fit_gaussian).pack(side="left", padx=(0, 16))
    ctk.CTkCheckBox(frame_opts1, text="Fit 3D Gaussian", variable=fit_3d_var).pack(side="left", padx=(0, 16))
    ctk.CTkCheckBox(frame_opts1, text="Subtract background", variable=subtract_background).pack(side="left")
    
    frame_opts2 = ctk.CTkFrame(app, fg_color="transparent")
    frame_opts2.pack(fill="x", **pad)
    ctk.CTkCheckBox(frame_opts2, text="Save bead diagnostics", variable=save_diagnostics_var).pack(side="left", padx=(0, 16))
    ctk.CTkCheckBox(frame_opts2, text="Auto-reject low QA", variable=qa_auto_reject_var).pack(side="left")
    
    frame_opts3 = ctk.CTkFrame(app, fg_color="transparent")
    frame_opts3.pack(fill="x", **pad)
    ctk.CTkCheckBox(frame_opts3, text="Review StarDist detection", variable=review_detection_var).pack(side="left", padx=(0, 16))
    ctk.CTkCheckBox(frame_opts3, text="Blob fallback (StarDist)", variable=blob_fallback_var).pack(side="left")
    frame_opts4 = ctk.CTkFrame(app, fg_color="transparent")
    frame_opts4.pack(fill="x", **pad)
    ctk.CTkCheckBox(frame_opts4, text="Cellpose native 3D", variable=cellpose_do_3d_var).pack(side="left", padx=(0, 16))
    ctk.CTkCheckBox(frame_opts4, text="Skip Cellpose review", variable=skip_cellpose_review_var).pack(side="left")
    frame_opts5 = ctk.CTkFrame(app, fg_color="transparent")
    frame_opts5.pack(fill="x", **pad)
    ctk.CTkCheckBox(frame_opts5, text="Local background", variable=local_background_var).pack(side="left", padx=(0, 16))
    ctk.CTkCheckBox(frame_opts5, text="Robust fit (Huber loss)", variable=robust_fit_var).pack(side="left")
    frame_box = ctk.CTkFrame(app, fg_color="transparent")
    frame_box.pack(fill="x", **pad)
    box_entry_var = ctk.StringVar(value="15")
    ctk.CTkLabel(frame_box, text="Box size (px):").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_box, textvariable=box_entry_var, width=50).pack(side="left")
    frame_qa = ctk.CTkFrame(app, fg_color="transparent")
    frame_qa.pack(fill="x", **pad)
    ctk.CTkLabel(frame_qa, text="QA min SNR:").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_qa, textvariable=qa_snr_var, width=60).pack(side="left", padx=(0, 16))
    ctk.CTkLabel(frame_qa, text="QA min symmetry:").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_qa, textvariable=qa_sym_var, width=60).pack(side="left")
    frame_aniso = ctk.CTkFrame(app, fg_color="transparent")
    frame_aniso.pack(fill="x", **pad)
    ctk.CTkLabel(frame_aniso, text="Cellpose anisotropy (z/xy):").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_aniso, textvariable=anisotropy_var, width=70).pack(side="left")
    frame_cp_params = ctk.CTkFrame(app, fg_color="transparent")
    frame_cp_params.pack(fill="x", **pad)
    ctk.CTkLabel(frame_cp_params, text="Cellpose min size (px):").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_cp_params, textvariable=cellpose_min_size_var, width=50).pack(side="left", padx=(0, 16))
    ctk.CTkLabel(frame_cp_params, text="Flow threshold:").pack(side="left", padx=(0, 8))
    ctk.CTkEntry(frame_cp_params, textvariable=cellpose_flow_threshold_var, width=50).pack(side="left")

    # Run button
    ctk.CTkButton(app, text="Run Analysis", command=run, height=36, font=ctk.CTkFont(weight="bold")).pack(**pad)

    # Status
    ctk.CTkLabel(app, textvariable=status_var, text_color="gray").pack(**pad)

    app.mainloop()
    return 0


if __name__ == "__main__":
    exit(main())
