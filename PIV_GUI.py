# piv_gui_v22_GUI_Plot.py
# - Sequential processing, Green Channel, Help buttons, Calibration for cropped inputs.
# - REMOVED all coordinate correction logic and outputs.
# - Saves ONLY original world/scaled/pixel data CSV.
# - MODIFIED: Average flow plot now displayed within the GUI on a Matplotlib canvas.
# - ADDED: Controls to adjust quiver arrow scale multiplier in GUI.
# - ADDED: Button to refresh plot display and save the current view to PNG.
# - Other plots removed. Maintained readable formatting.

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
import logging
import time
import math
import traceback # For detailed error logging

import cv2
import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend BEFORE importing pyplot MUST BE HERE
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Attempt imports
try:
    from openpiv import tools, pyprocess, validation, filters, preprocess
except ImportError:
    root_check = tk.Tk()
    root_check.withdraw()
    messagebox.showerror("Missing Library", "Error: 'openpiv-python' library not found.\nPlease install it: pip install openpiv-python")
    root_check.destroy()
    exit()

try:
    # Assumes calibration.py contains pixel_to_world_via_plane AND create_world_view_image_fixed_canvas
    import calibration
    if not hasattr(calibration, 'pixel_to_world_via_plane'):
        raise AttributeError("Function 'pixel_to_world_via_plane' not found.")
    # create_world_view is still needed for the plot background warping
    if not hasattr(calibration, 'create_world_view_image_fixed_canvas'):
        raise AttributeError("Function 'create_world_view_image_fixed_canvas' not found.")
    CALIBRATION_AVAILABLE = True
    logging.info("Successfully imported calibration functions.")
except (ImportError, AttributeError) as e:
    logging.warning(f"Calibration module/function not found: {e}. Calibration/Warping disabled.")
    CALIBRATION_AVAILABLE = False # Disable if essential functions missing

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Parameter Help Texts ---
PARAM_HELP = {
    "window_size": "Size (pixels) of the square interrogation window in Frame A.",
    "overlap": "Overlap (pixels) between adjacent windows (e.g., 50% of window_size).",
    "search_area_size": "Size (pixels) of the search area in Frame B (must be >= window_size).",
    "sig2noise_method": "Method for S/N calculation ('peak2peak' or 'matrix_comparison').",
    "noise_threshold": "Minimum S/N ratio threshold for vector validation (e.g., 1.0-1.5).",
    "filter_method": "Method to replace invalid vectors ('localmean', 'median', 'gaussian').",
    "max_iter": "Max iterations for the outlier replacement filter.",
    "kernel_size": "Neighborhood size (kernel_size x kernel_size) for outlier filter.",
    "processing_fps": "Target analysis FPS (determines frame step).",
    "displacement_frames": "Frame difference (dt) between Frame A and B.",
    "start_frame": "First Frame A index (0-based).",
    "end_frame": "Last potential Frame A index (inclusive).",
    "units_per_pixel": "(Fallback Scaling) Used if no calibration file is loaded.",
    "arrow_scale_multiplier": "Adjusts arrow length. >1 makes arrows shorter, <1 makes arrows longer relative to the default scale (based on 95th percentile velocity)."
}


# --- PIV Core Processing Logic ---
def process_piv_pair(frame_a, frame_b, window_size, overlap, search_area_size, sig2noise_method, dt=1.0, calib_params=None):
    """Processes PIV for a frame pair, uses GREEN channel, optionally undistorts."""
    try:
        if not isinstance(frame_a, np.ndarray) or not isinstance(frame_b, np.ndarray):
            raise TypeError("Inputs must be NumPy arrays.")
        if frame_a.ndim < 3 or frame_b.ndim < 3 or frame_a.shape[2] < 3 or frame_b.shape[2] < 3:
            raise ValueError("Inputs must be 3-channel color images.")
        frame_a_processed = frame_a.copy(); frame_b_processed = frame_b.copy()
        img_shape = frame_a.shape[:2]
        if calib_params and 'K' in calib_params and 'dist' in calib_params:
            K, dist = calib_params['K'], calib_params['dist']
            frame_a_processed = cv2.undistort(frame_a_processed, K, dist, None, K)
            frame_b_processed = cv2.undistort(frame_b_processed, K, dist, None, K)
            img_shape = frame_a_processed.shape[:2]
        piv_input_a = frame_a_processed[:, :, 1].astype(np.int32) # Green channel
        piv_input_b = frame_b_processed[:, :, 1].astype(np.int32) # Green channel
        u, v, sig2noise = pyprocess.extended_search_area_piv(
            piv_input_a, piv_input_b, window_size=window_size, overlap=overlap, dt=dt,
            search_area_size=search_area_size, sig2noise_method=sig2noise_method
        )
        if u is not None: return u.copy(), v.copy(), sig2noise.copy(), img_shape
        else: logger.warning("PIV returned None."); return None, None, None, None
    except Exception as e: logger.error(f"Error in process_piv_pair: {e}", exc_info=True); return None, None, None, None

def validate_and_filter_piv(u, v, sig2noise, noise_threshold, filter_method, max_iter, kernel_size):
    """Validates and filters PIV results."""
    if u is None: return None, None, -1
    try:
        invalid_mask = validation.sig2noise_val(sig2noise, threshold=noise_threshold)
        invalid_count = invalid_mask.sum()
        u_f, v_f = filters.replace_outliers(u.copy(), v.copy(), invalid_mask, method=filter_method, max_iter=max_iter, kernel_size=kernel_size)
        if u_f is None: raise RuntimeError("replace_outliers failed.")
        return u_f, v_f, invalid_count
    except Exception as e: logger.error(f"Error in validation/filtering: {e}", exc_info=True); return u, v, -1


# --- GUI Class ---
class PivGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenPIV GUI Processor (v22 - GUI Plot)")
        self.root.geometry("800x950") # Increased height for plot

        # --- State Variables ---
        self.input_video_path = tk.StringVar()
        self.output_base_name = tk.StringVar()
        self.video_properties = {}
        self.calib_file_path = tk.StringVar(value="No calib file loaded.")
        self.calib_params = {}
        self.piv_parameters = {}
        self.piv_param_vars = {}
        self.processing_thread = None
        self.cancel_processing = threading.Event()
        # Plotting related state
        self.plot_data = {} # Store averaged data for plotting
        self.arrow_scale_multiplier_var = tk.DoubleVar(value=1.0) # Multiplier for arrow scale

        # --- Style ---
        style = ttk.Style()
        style.configure("TButton", padding=5)
        style.configure("TLabel", padding=2)
        style.configure("TEntry", padding=2)
        style.configure("TLabelframe.Label", font=('Helvetica', 10, 'bold'))
        style.configure("Help.TButton", padding=(1, 1), font=('Helvetica', 8))

        # --- Main Frame ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(5, weight=1) # Make plot row expandable
        main_frame.columnconfigure(0, weight=1)

        # --- Row 0-3: Input, Params, Calib, Run (Layout mostly same) ---
        # Row 0
        frame_input = ttk.LabelFrame(main_frame, text="1. Input Cropped Video", padding=(10, 5))
        frame_input.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        frame_input.columnconfigure(1, weight=1)
        btn_load = ttk.Button(frame_input, text="Load Cropped Video", command=self.load_video)
        btn_load.grid(row=0, column=0, padx=5, pady=5)
        self.lbl_filename = ttk.Label(frame_input, text="No cropped video selected.", anchor="w", relief="sunken", padding=3)
        self.lbl_filename.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.lbl_video_info = ttk.Label(frame_input, text="Video Info: -", anchor="w")
        self.lbl_video_info.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=2)

        # Row 1
        frame_params = ttk.LabelFrame(main_frame, text="2. PIV Parameters", padding=(10, 5))
        frame_params.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        frame_params.columnconfigure(2, weight=1); frame_params.columnconfigure(6, weight=1)
        param_row = 0
        self.add_param("Window Size", "window_size", 64, param_row, 0, unit="px", frame_widget=frame_params)
        self.add_param("Overlap", "overlap", 32, param_row, 1, unit="px", frame_widget=frame_params)
        param_row += 1
        self.add_param("Search Area Size", "search_area_size", 76, param_row, 0, unit="px", frame_widget=frame_params)
        self.add_param("S/N Method", 'sig2noise_method', 'peak2peak', param_row, 1, width=12, is_combo=True, combo_values=['peak2peak', 'matrix_comparison'], frame_widget=frame_params)
        param_row += 1
        ttk.Separator(frame_params, orient='horizontal').grid(row=param_row, columnspan=8, sticky='ew', pady=(10, 5))
        param_row += 1
        self.add_param("S/N Threshold", "noise_threshold", 1.15, param_row, 0, frame_widget=frame_params)
        self.add_param("Filter Method", 'filter_method', 'localmean', param_row, 1, width=12, is_combo=True, combo_values=['localmean', 'median', 'gaussian'], frame_widget=frame_params)
        param_row += 1
        self.add_param("Filter Max Iter", "max_iter", 3, param_row, 0, frame_widget=frame_params)
        self.add_param("Filter Kernel Size", "kernel_size", 2, param_row, 1, frame_widget=frame_params)
        param_row += 1
        ttk.Separator(frame_params, orient='horizontal').grid(row=param_row, columnspan=8, sticky='ew', pady=(10, 5))
        param_row += 1
        self.add_param("Processing FPS", "processing_fps", 10, param_row, 0, unit="frames/sec", frame_widget=frame_params)
        self.add_param("Displacement (dt)", "displacement_frames", 1, param_row, 1, unit="frames", frame_widget=frame_params)
        param_row += 1
        self.add_param("Start Frame", "start_frame", 0, param_row, 0, frame_widget=frame_params)
        self.add_param("End Frame", "end_frame", 0, param_row, 1, frame_widget=frame_params)

        # Row 2
        frame_calib = ttk.LabelFrame(main_frame, text="3. Calibration (for Cropped Frame)", padding=(10, 5))
        frame_calib.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        frame_calib.columnconfigure(1, weight=1)
        btn_load_calib = ttk.Button(frame_calib, text="Load Calib File (.npz)", command=self.load_calibration_file, state=tk.NORMAL if CALIBRATION_AVAILABLE else tk.DISABLED)
        btn_load_calib.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lbl_calib_file = ttk.Label(frame_calib, textvariable=self.calib_file_path, anchor="w", relief="sunken", padding=3)
        self.lbl_calib_file.grid(row=0, column=1, columnspan=3, sticky="ew", padx=5, pady=5)
        if not CALIBRATION_AVAILABLE: self.calib_file_path.set("Calibration module missing.")
        self.units_var, self.entry_units = self.add_param("Units/Pixel (Fallback)", "units_per_pixel", 1.0, 1, 0, unit="(e.g., mm/px)", frame_widget=frame_calib)
        self.entry_units.grid(sticky="w")

        # Row 3
        frame_run = ttk.LabelFrame(main_frame, text="4. Run Analysis", padding=(10, 5))
        frame_run.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        frame_run.columnconfigure(1, weight=1)
        ttk.Label(frame_run, text="Output Base Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entry_output = ttk.Entry(frame_run, textvariable=self.output_base_name, width=60)
        entry_output.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        button_frame = ttk.Frame(frame_run)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 5))
        self.btn_run = ttk.Button(button_frame, text="Run PIV Analysis", command=self.run_piv_analysis, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=10)
        self.btn_cancel = ttk.Button(button_frame, text="Cancel", command=self.cancel_piv_analysis, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=10)

        # --- Row 4: Status & Progress ---
        frame_status = ttk.LabelFrame(main_frame, text="5. Status", padding=(10, 5))
        frame_status.grid(row=4, column=0, sticky="ew", padx=5, pady=5) # Changed row configure weight above
        frame_status.columnconfigure(0, weight=1)
        self.lbl_status = ttk.Label(frame_status, text="Load a cropped video to begin.", anchor="w", wraplength=700)
        self.lbl_status.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(frame_status, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, expand=True, padx=5, pady=5)

        # --- Row 5: Embedded Plot ---
        frame_plot = ttk.LabelFrame(main_frame, text="6. Average Velocity Plot", padding=(10, 5))
        frame_plot.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)
        frame_plot.rowconfigure(1, weight=1) # Canvas row expands
        frame_plot.columnconfigure(0, weight=1) # Canvas col expands

        # Plot Controls Frame
        plot_controls_frame = ttk.Frame(frame_plot)
        plot_controls_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(plot_controls_frame, text="Arrow Scale Multiplier:").pack(side=tk.LEFT, padx=(0, 2))
        scale_entry = ttk.Entry(plot_controls_frame, textvariable=self.arrow_scale_multiplier_var, width=8)
        scale_entry.pack(side=tk.LEFT, padx=(0, 5))
        scale_help_button = ttk.Button(plot_controls_frame, text="?", width=2, style="Help.TButton",
                                       command=lambda k='arrow_scale_multiplier': self.show_help(k))
        scale_help_button.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_refresh_save = ttk.Button(plot_controls_frame, text="Refresh Plot & Save PNG",
                                           command=self.refresh_and_save_plot, state=tk.DISABLED)
        self.btn_refresh_save.pack(side=tk.LEFT, padx=5)

        # Matplotlib Figure and Canvas
        self.plot_fig = Figure(figsize=(6, 5), dpi=100) # Initial size
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.plot_ax.set_title("Plot will appear here after analysis")
        self.plot_ax.text(0.5, 0.5, "No data yet", ha='center', va='center', transform=self.plot_ax.transAxes)

        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=frame_plot)
        self.canvas_widget = self.plot_canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Matplotlib Toolbar
        toolbar = NavigationToolbar2Tk(self.plot_canvas, frame_plot)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)


    # --- add_param Helper defined as a METHOD now ---
    def add_param(self, label, key, default_value, row, col_group=0, width=10, unit="", columnspan=1, is_combo=False, combo_values=None, state=tk.NORMAL, frame_widget=None, **kwargs):
        """Adds labeled parameter widget with help button to specified frame."""
        # Default to self.frame_params if no frame_widget is provided
        if frame_widget is None:
            frame_widget = self.frame_params

        base_col = col_group * 4 # Label, Help, Widget, Unit
        ttk.Label(frame_widget, text=f"{label}:").grid(row=row, column=base_col, sticky="w", padx=(10, 0), pady=3)
        # Only add help button if key exists in PARAM_HELP
        if key in PARAM_HELP:
            help_button = ttk.Button(frame_widget, text="?", width=2, style="Help.TButton", command=lambda k=key: self.show_help(k))
            help_button.grid(row=row, column=base_col + 1, sticky="w", padx=(2, 5), pady=3)
        else: # Add empty label for alignment if no help
             ttk.Label(frame_widget, text="").grid(row=row, column=base_col + 1, padx=(2, 5), pady=3)

        var = tk.StringVar(value=str(default_value))
        if is_combo:
            widget = ttk.Combobox(frame_widget, textvariable=var, values=combo_values, state='readonly', width=width, **kwargs)
        else:
            widget = ttk.Entry(frame_widget, textvariable=var, width=width, state=state, **kwargs)
        widget.grid(row=row, column=base_col + 2, sticky="ew", padx=0, pady=3, columnspan=columnspan)
        if unit:
            ttk.Label(frame_widget, text=unit).grid(row=row, column=base_col + 3, sticky="w", padx=(2, 10), pady=3)
        self.piv_param_vars[key] = var
        return var, widget

    # --- Help Button Action ---
    def show_help(self, param_key):
        """Displays help text for the given parameter key."""
        help_text = PARAM_HELP.get(param_key, "No help available for this parameter.")
        messagebox.showinfo(f"Help: {param_key}", help_text)

    # --- GUI Methods ---
    def load_video(self):
        """Loads CROPPED video file."""
        filepath = filedialog.askopenfilename(title="Select CROPPED Video File", filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")))
        if not filepath: return
        self.input_video_path.set(filepath); self.video_properties = {}; self.calib_params = {}; self.calib_file_path.set("No calib file loaded."); self.entry_units.config(state=tk.NORMAL); self.piv_param_vars["units_per_pixel"].set("1.0")
        dir_name = os.path.dirname(filepath); base_name = os.path.splitext(os.path.basename(filepath))[0]
        if base_name.lower().endswith('_cropped'): base_name = base_name[:-8]
        self.output_base_name.set(os.path.join(dir_name, base_name))
        cap = None
        try:
            cap = cv2.VideoCapture(filepath); assert cap.isOpened(), "Cannot open video."
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps, frame_count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); assert width > 0 and height > 0 and frame_count > 0 and fps > 0, "Video props invalid."
            duration = frame_count / fps; self.video_properties = {"width": width, "height": height, "fps": fps, "frame_count": frame_count, "duration": duration}
            self.lbl_filename.config(text=os.path.basename(filepath)); info_text = f"Info (Cropped): {width}x{height} | {frame_count} frames | {fps:.2f} FPS | {duration:.2f}s"; self.lbl_video_info.config(text=info_text)
            win_size = max(16, int(min(width, height)/32)*2); overlap = win_size//2; search_size = win_size+12
            self.piv_param_vars["window_size"].set(str(win_size)); self.piv_param_vars["overlap"].set(str(overlap)); self.piv_param_vars["search_area_size"].set(str(search_size))
            self.piv_param_vars["processing_fps"].set(f"{fps:.1f}"); disp_frames = int(self.piv_param_vars["displacement_frames"].get()) if self.piv_param_vars["displacement_frames"].get().isdigit() else 1; default_end = max(0, frame_count - disp_frames - 1)
            self.piv_param_vars["end_frame"].set(str(default_end)); self.piv_param_vars["start_frame"].set("0")
            self.lbl_status.config(text=f"Cropped video loaded. Adjust params, optionally load calib, and Run."); self.btn_run.config(state=tk.NORMAL)
            self.plot_data = {} # Clear previous plot data
            self._update_gui_plot() # Clear plot canvas
            self.btn_refresh_save.config(state=tk.DISABLED) # Disable refresh until run
        except Exception as e: messagebox.showerror("Video Load Error", f"Failed: {e}"); self.lbl_filename.config(text="Load failed."); self.lbl_video_info.config(text="Video Info: -"); self.btn_run.config(state=tk.DISABLED); logger.error(f"Video loading failed: {e}", exc_info=True)
        finally:
            if cap: cap.release()

    def load_calibration_file(self):
        """Loads NPZ calibration file (intended for cropped frame)."""
        if not CALIBRATION_AVAILABLE: messagebox.showerror("Error", "Calibration module not loaded."); return
        filepath = filedialog.askopenfilename(title="Select Calibration File (for Cropped Frame)", filetypes=(("NumPy NPZ files", "*.npz"), ("All files", "*.*")))
        if not filepath: return
        try:
             with np.load(filepath) as data:
                 required_keys = ['K', 'dist', 'R', 'T']; assert all(key in data for key in required_keys), f"NPZ missing key(s): {required_keys}"
                 self.calib_params = {key: data[key] for key in data}; logger.info(f"Loaded calib from {os.path.basename(filepath)}")
                 self.calib_file_path.set(os.path.basename(filepath)); self.update_status(f"Calibration loaded: {os.path.basename(filepath)}. 'Units per Pixel' disabled."); self.entry_units.config(state=tk.DISABLED)
                 if 'image_size' in self.calib_params and self.video_properties: # Dimension check
                     calib_w, calib_h = self.calib_params['image_size']; vid_w, vid_h = self.video_properties['width'], self.video_properties['height']
                     if calib_w != vid_w or calib_h != vid_h: msg = (f"Info: Video dims ({vid_w}x{vid_h}) != calib size ({calib_w}x{calib_h}). Ensure calib file matches."); messagebox.showinfo("Dimension Info", msg); logger.info(msg)
        except Exception as e: messagebox.showerror("Load Error", f"Failed to load calib file:\n{e}"); self.calib_params={}; self.calib_file_path.set("Load failed."); self.entry_units.config(state=tk.NORMAL); logger.error(f"Calib loading failed: {e}", exc_info=True)

    def update_status(self, message):
        if self.root:
            try:
                self.root.after(0, lambda: self.lbl_status.config(text=message))
            except tk.TclError:
                pass
        logger.info(message)

    def update_progress(self, value):
        if self.root:
            try:
                safe_value = max(0, min(100, int(value)))
                self.root.after(0, lambda: self.progress_bar.config(value=safe_value))
            except tk.TclError:
                pass

    def run_piv_analysis(self):
        """Validates parameters and starts the PIV processing thread (SEQUENTIAL)."""
        if not self.input_video_path.get() or not os.path.exists(self.input_video_path.get()):
            messagebox.showerror("Error", "Load cropped video first.")
            return
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Processing is already running.")
            return

        params = {}
        try: # Param Validation
            int_params=["window_size","overlap","search_area_size","max_iter","kernel_size","displacement_frames","start_frame","end_frame"]
            for k in int_params:
                params[k]=int(self.piv_param_vars[k].get())
            float_params=["noise_threshold","processing_fps"]
            for k in float_params:
                params[k]=float(self.piv_param_vars[k].get())
            params["sig2noise_method"]=self.piv_param_vars["sig2noise_method"].get()
            params["filter_method"]=self.piv_param_vars["filter_method"].get()

            if not self.calib_params:
                params["units_per_pixel"]=float(self.piv_param_vars["units_per_pixel"].get())
                assert params["units_per_pixel"] > 0, "Units per Pixel must be positive."
            else:
                params["units_per_pixel"]=None

            # --- Logic Checks ---
            assert params["window_size"] > 0 and params["overlap"] >= 0 and params["search_area_size"] > 0, "Sizes must be positive."
            assert params["overlap"] < params["window_size"], "Overlap < Window Size."
            assert params["processing_fps"] > 0, "Processing FPS > 0."
            assert params["start_frame"] >= 0, "Start Frame >= 0."
            assert params["end_frame"] > params["start_frame"], "End Frame > Start Frame."

            max_f = self.video_properties.get("frame_count", float('inf')) - 1
            if params["end_frame"] > max_f:
                logger.warning(f"End Frame adjusted to {int(max_f)}")
                params["end_frame"]=int(max_f)
                self.piv_param_vars["end_frame"].set(str(params["end_frame"]))
            assert params["start_frame"]+params["displacement_frames"] <= params["end_frame"], "Start+Displacement <= End Frame."

        except Exception as e:
            messagebox.showerror("Invalid Parameter", f"{e}")
            return

        # --- Start Background Processing Thread ---
        self.cancel_processing.clear()
        self.btn_run.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.btn_refresh_save.config(state=tk.DISABLED) # Disable refresh during run
        status_prefix = "Starting PIV analysis" + (" (Calibrated)" if self.calib_params else "")
        self.update_status(status_prefix + "...")
        self.update_progress(0)
        self.piv_parameters = params.copy()
        thread_calib_params = self.calib_params.copy() if self.calib_params else None

        # Run the sequential worker in a thread to keep GUI responsive
        self.processing_thread = threading.Thread(target=self._piv_processing_worker_sequential, args=(thread_calib_params,), daemon=True)
        self.processing_thread.start()

    def cancel_piv_analysis(self):
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Cancel request received by GUI.")
            self.cancel_processing.set()
            self.update_status("Cancellation requested... finishing current pair.")
            self.btn_cancel.config(state=tk.DISABLED)
        else:
            self.update_status("No active processing to cancel.")

    # --- PIV Worker Thread (Sequential) ---
    def _piv_processing_worker_sequential(self, calib_params):
        """
        SEQUENTIAL PIV processing loop. Stores results for GUI plot.
        """
        start_time_proc = time.time()
        # --- Get parameters ---
        video_path = self.input_video_path.get()
        output_base = self.output_base_name.get()
        params = self.piv_parameters
        vid_props = self.video_properties

        # --- Basic Setup & Validation ---
        if not vid_props or vid_props.get("fps", 0) <= 0:
            self.update_status("ERROR: Video properties invalid.")
            self.root.after(0, self._piv_finished, "ERROR: Video properties invalid.")
            return
        video_fps = vid_props["fps"]
        time_step_frames = params.get("displacement_frames", 1)
        dt_sec = time_step_frames / video_fps
        dt_process_frames = max(1, int(round(video_fps / params.get("processing_fps", video_fps))))
        start_frame_idx = params.get("start_frame", 0)
        end_frame_idx = min(params.get("end_frame", vid_props["frame_count"] - 1),
                            vid_props["frame_count"] - 1 - time_step_frames)

        # --- Determine Units ---
        use_calibration = bool(calib_params and CALIBRATION_AVAILABLE)
        units_label, vel_units_label = ("pixels", "pixels/frame")
        units_per_pixel = params.get("units_per_pixel")
        K, dist, R, T = None, None, None, None # Define for scope fix
        if use_calibration:
            K, dist, R, T = calib_params['K'], calib_params['dist'], calib_params['R'], calib_params['T']
            units_label, vel_units_label = "mm", "mm/sec"
        elif units_per_pixel:
            units_label, vel_units_label = f"units(scale={units_per_pixel:.3g})", f"units/sec"
        if dt_sec <= 0 and "/sec" in vel_units_label:
            logger.warning(f"dt={dt_sec:.4f}s zero/negative. Velocity incorrect.")

        # --- Prepare Task Indices & Result Storage ---
        frame_indices_to_process = list(range(start_frame_idx, end_frame_idx + 1, dt_process_frames))
        total_tasks = len(frame_indices_to_process)
        if total_tasks == 0:
             self.update_status("No frame pairs to process.")
             self.root.after(0, self._piv_finished, "No frame pairs.")
             return

        logger.info(f"Total pairs to process sequentially: {total_tasks}")
        self.update_status(f"Starting sequential processing for {total_tasks} pairs...")

        # Store results per frame
        results = {'xw': None, 'yw': None, 'uw': [], 'vw': [], 'frame_index': [], 'sig2noise': []}
        first_frame_display = None
        processed_count = 0
        x_pix, y_pix = None, None

        # --- Open Video Capture Once ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.update_status("ERROR: Failed to open video file.")
            self.root.after(0, self._piv_finished, "ERROR: Failed to open video.")
            return

        try:
            # --- Main Sequential Processing Loop ---
            for frame_a_idx in frame_indices_to_process:
                if self.cancel_processing.is_set():
                    self.update_status("Cancelled.")
                    break # Exit loop

                frame_b_idx = frame_a_idx + time_step_frames
                # Read frame pair
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_a_idx)
                ret_a, frame_a = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_b_idx)
                ret_b, frame_b = cap.read()

                if not ret_a or not ret_b:
                    logger.warning(f"Failed to read frames {frame_a_idx}/{frame_b_idx}. Stopping.")
                    break # Stop if frames can't be read

                # Store first frame for plot background
                if processed_count == 0:
                    first_frame_to_store = frame_a.copy()
                    # Use K, dist defined in the outer scope
                    if use_calibration and K is not None:
                        first_frame_to_store = cv2.undistort(first_frame_to_store, K, dist, None, K)
                    first_frame_display = first_frame_to_store

                # Update status
                status_msg = f"Processing PIV: Frames {frame_a_idx} & {frame_b_idx} ({processed_count + 1}/{total_tasks})..."
                self.update_status(status_msg)

                # --- PIV Core ---
                u_pix, v_pix, sig2noise_map, processed_shape = process_piv_pair(
                    frame_a, frame_b, params["window_size"], params["overlap"],
                    params["search_area_size"], params["sig2noise_method"], dt=1.0,
                    calib_params=calib_params
                )
                if u_pix is None:
                    logger.warning(f"Skipping pair {frame_a_idx}: Core PIV failed.")
                    continue # Skip to next pair

                # --- Validation & Filtering ---
                u_pix_f, v_pix_f, invalid_count = validate_and_filter_piv(
                    u_pix, v_pix, sig2noise_map, params["noise_threshold"],
                    params["filter_method"], params["max_iter"], params["kernel_size"]
                )
                if u_pix_f is None:
                    logger.warning(f"Skipping pair {frame_a_idx}: Validation failed.")
                    continue
                logger.debug(f"Pair {frame_a_idx}: {invalid_count} vectors replaced.")

                # --- Coordinate & Velocity Transformation ---
                if x_pix is None: # Calculate grids only once
                    x_pix, y_pix = pyprocess.get_coordinates(processed_shape, params["search_area_size"], params["overlap"])
                    if use_calibration:
                        pixel_points_grid = np.vstack((x_pix.ravel(), y_pix.ravel())).T
                        try:
                            world_points_grid = calibration.pixel_to_world_via_plane(pixel_points_grid, K, dist, R, T)
                            results['xw'] = world_points_grid[:, 0].reshape(x_pix.shape)
                            results['yw'] = world_points_grid[:, 1].reshape(x_pix.shape)
                        except Exception as e:
                            raise RuntimeError(f"World coordinate transformation failed: {e}")
                    elif units_per_pixel:
                        results['xw'] = x_pix * units_per_pixel
                        results['yw'] = y_pix * units_per_pixel
                    else:
                        results['xw'] = x_pix.copy()
                        results['yw'] = y_pix.copy()
                    if results['xw'] is None:
                        raise RuntimeError("Coord generation failed.")

                # Calculate Velocities (Original World/Scaled/Pixel)
                uw_frame = np.full_like(x_pix, np.nan); vw_frame = np.full_like(x_pix, np.nan)
                if use_calibration:
                    start_pixel = np.vstack((x_pix.ravel(), y_pix.ravel())).T
                    end_pixel = np.vstack(((x_pix + u_pix_f).ravel(), (y_pix + v_pix_f).ravel())).T
                    try:
                        start_world = calibration.pixel_to_world_via_plane(start_pixel, K, dist, R, T)
                        end_world = calibration.pixel_to_world_via_plane(end_pixel, K, dist, R, T)
                        world_disp = end_world - start_world
                        if dt_sec > 1e-9:
                            uw_frame=(world_disp[:,0]/dt_sec).reshape(x_pix.shape)
                            vw_frame=(world_disp[:,1]/dt_sec).reshape(x_pix.shape)
                    except Exception as e:
                        logger.error(f"Velocity transform frame {frame_a_idx} failed: {e}", exc_info=True) # Keep NaNs
                elif units_per_pixel:
                    if dt_sec > 1e-9: uw_frame=(u_pix_f*units_per_pixel)/dt_sec; vw_frame=(v_pix_f*units_per_pixel)/dt_sec
                else:
                    uw_frame, vw_frame = u_pix_f.copy(), v_pix_f.copy() # Velocity is pixel displacement / frame

                # --- Store results for this frame pair ---
                results['uw'].append(uw_frame); results['vw'].append(vw_frame)
                results['sig2noise'].append(sig2noise_map)
                results['frame_index'].append(frame_a_idx)

                processed_count += 1
                progress = min(100, int(100 * processed_count / total_tasks))
                self.update_progress(progress)
            # --- End of Sequential Processing Loop ---

        except Exception as e:
            logger.error("Error during sequential PIV processing loop:", exc_info=True)
            error_msg = f"ERROR during processing: {e}"
            self.update_status(error_msg)
            self.root.after(0, self._piv_finished, error_msg)
            return # Stop processing
        finally:
            if cap: cap.release(); logger.info("Video capture released.")

        # --- Post-Processing (after sequential execution) ---
        if self.cancel_processing.is_set():
            self.root.after(0, self._piv_finished, "Processing Cancelled.")
            return
        if processed_count == 0 or not results['uw']: # Check if any results were actually stored
            self.update_status("ERROR: No results generated.")
            self.root.after(0, self._piv_finished, "ERROR: No results generated.")
            return

        self.update_status("Processing complete. Preparing data for display...")

        # --- Assemble final arrays ---
        xw_coords = results['xw']; yw_coords = results['yw'] # Original world/scaled/pixel coords
        uw_all = np.stack(results['uw'], axis=0); vw_all = np.stack(results['vw'], axis=0) # Original world/scaled/pixel vels
        sig2noise_all = np.stack(results['sig2noise'], axis=0)
        frame_indices = np.array(results['frame_index'])

        # --- REMOVED Coordinate Correction Logic ---

        logger.info(f"Successfully processed {processed_count} pairs sequentially.")
        self.update_status("Processing complete. Calculating averages...")

        # --- Calculate Averages ---
        uw_avg = np.nanmean(uw_all, axis=0); vw_avg = np.nanmean(vw_all, axis=0)
        sig2noise_avg = np.nanmean(sig2noise_all, axis=0)

        # --- Prepare Data for GUI Plot ---
        self.plot_data = {
            'xw': xw_coords,
            'yw': yw_coords,
            'uw_avg': uw_avg,
            'vw_avg': vw_avg,
            'background': first_frame_display, # Potentially undistorted
            'calib_params': calib_params, # Needed for potential warping
            'units_label': units_label,
            'vel_units_label': vel_units_label
        }

        # Calculate base scale for arrows (95th percentile)
        magnitude_avg = np.sqrt(uw_avg**2 + vw_avg**2)
        valid_mag = np.isfinite(magnitude_avg)
        if np.any(valid_mag):
            nz_mag = magnitude_avg[valid_mag & (magnitude_avg > 1e-9)]
            if len(nz_mag) > 0:
                self.plot_data['base_scale'] = np.percentile(nz_mag, 95)
            else:
                self.plot_data['base_scale'] = 1.0 # Default if all magnitudes are zero
        else:
            self.plot_data['base_scale'] = 1.0 # Default if no valid magnitudes

        # Reset GUI scale multiplier to 1.0 for new data
        self.arrow_scale_multiplier_var.set(1.0)

        # --- Save Data Files ---
        # 1. Raw Data CSV (Original World Coords + S/N)
        csv_path = f"{output_base}_world_data.csv"; self.update_status(f"Saving raw data CSV ({csv_path})..."); logger.info("Saving raw data CSV...")
        try:
            n_steps, ny, nx = uw_all.shape; n_vec = ny * nx; xw_flat = np.tile(xw_coords.flatten(), n_steps); yw_flat = np.tile(yw_coords.flatten(), n_steps)
            frame_col = np.repeat(frame_indices, n_vec); uw_flat = uw_all.flatten(); vw_flat = vw_all.flatten()
            sig2noise_flat = sig2noise_all.flatten() # Flatten S/N
            df = pd.DataFrame({f'x_pos_{units_label}': xw_flat, f'y_pos_{units_label}': yw_flat, f'u_vel_{vel_units_label}': uw_flat, f'v_vel_{vel_units_label}': vw_flat, 'sig2noise': sig2noise_flat}, index=frame_col); df.index.name = 'frame_index' # Add S/N
            df.to_csv(csv_path, float_format='%.5g', na_rep='NaN'); logger.info(f"Saved world data: {csv_path}")
        except Exception as e: logger.error(f"CSV saving failed: {e}", exc_info=True); self.update_status(f"Warn: Error saving CSV: {e}")

        # 2. REMOVED Corrected Raw Data CSV Saving

        # 3. Stats Calc (Using ORIGINAL world/scaled velocities)
        self.update_status("Calculating final statistics..."); logger.info("Calculating stats...")
        avg_vel_mag, avg_rms_fluctuation = np.nan, np.nan; valid_stats = np.isfinite(uw_avg) & np.isfinite(vw_avg)
        if processed_count > 0 and np.any(valid_stats):
            mag_avg_map = np.sqrt(uw_avg**2 + vw_avg**2); valid_mags = mag_avg_map[valid_stats]
            if len(valid_mags) > 0: perc_5, perc_95 = np.percentile(valid_mags, [5, 95]); mask_mag_perc = (valid_mags >= perc_5) & (valid_mags <= perc_95); avg_vel_mag = np.mean(valid_mags[mask_mag_perc]) if np.any(mask_mag_perc) else np.mean(valid_mags); logger.info(f"Avg Vel Mag (orig, spatially filtered): {avg_vel_mag:.5f} {vel_units_label}")
            if processed_count > 1:
                 uw_prime = uw_all - uw_avg; vw_prime = vw_all - vw_avg; mag_prime_sq_tmean = np.nanmean(uw_prime**2 + vw_prime**2, axis=0); rms_map = np.sqrt(mag_prime_sq_tmean); valid_rms = rms_map[valid_stats]
                 if len(valid_rms) > 0: perc_5_rms, perc_95_rms = np.percentile(valid_rms, [5, 95]); mask_rms_perc = (valid_rms >= perc_5_rms) & (valid_rms <= perc_95_rms); avg_rms_fluctuation = np.mean(valid_rms[mask_rms_perc]) if np.any(mask_rms_perc) else np.mean(valid_rms); logger.info(f"Avg RMS Fluct (orig, spatially filtered): {avg_rms_fluctuation:.5f} {vel_units_label}")
        else: logger.warning("Skipping stats calc: no valid vectors.")
        avg_vel_msg_part = f"Avg Vel Mag: {avg_vel_mag:.4f}, Avg RMS Fluct: {avg_rms_fluctuation:.4f} {vel_units_label}"

        # --- REMOVED PLOTTING FROM WORKER ---

        # 4. Summary TXT
        summary_txt_path = f"{output_base}_summary.txt"; self.update_status(f"Saving summary ({summary_txt_path})..."); logger.info("Saving summary...")
        try: # Update summary notes
            with open(summary_txt_path, 'w') as f:
                f.write(f"PIV Analysis Summary\nInput Video: {os.path.basename(video_path)}\n" + "="*40 + "\n")
                if use_calibration: f.write(f"Calibration File: {self.calib_file_path.get()}\n -> Units: mm, mm/sec\n")
                elif units_per_pixel: f.write(f"Scaling: Fallback = {units_per_pixel} units/pixel\n -> Units: units, units/sec\n")
                else: f.write("Scaling: None\n -> Units: pixels, pixels/frame\n")
                # Removed reference to corrected CSV
                f.write(f"Avg Flow Plot: {os.path.basename(output_base)}_average_flow_world_overlay.png (Generated via GUI button)\n") # Updated plot name
                # Removed references to other plots
                f.write("-" * 40 + "\n"); f.write(f"Processed Pairs: {processed_count}\n"); f.write(f"Frame Range (A): {frame_indices[0]}-{frame_indices[-1]}\n" if frame_indices.size>0 else "Frame Range: N/A\n")
                f.write(f"Time Step (dt): {dt_sec:.5f} sec ({time_step_frames} frames @ {video_fps:.2f} FPS)\n"); f.write("-" * 40 + "\nParams:\n")
                params_to_write = {k:v for k,v in params.items() if not (k=='units_per_pixel' and use_calibration)}
                for k,v in params_to_write.items(): f.write(f"  {k}: {v}\n")
                f.write("-" * 40 + "\nResults (Original World/Scaled Coords, Spatial Outliers Excluded):\n"); f.write(f"Avg Vel Mag: {avg_vel_mag:.5f} ({vel_units_label})\n"); f.write(f"Avg RMS Fluct: {avg_rms_fluctuation:.5f} ({vel_units_label})\n\nNotes:\n")
                f.write("- PIV analysis performed on GREEN channel.\n"); f.write("- Stats calculated on valid vectors.\n"); f.write("- Averages exclude spatial outliers (5th-95th percentile).\n")
            logger.info(f"Saved summary: {summary_txt_path}")
        except Exception as e: logger.error(f"Summary saving failed: {e}", exc_info=True); self.update_status(f"Warn: Error saving summary: {e}")

        # --- Final Status Update & Schedule GUI Plot ---
        total_time = time.time() - start_time_proc
        final_msg = f"Processing Complete! ({processed_count} pairs in {total_time:.1f}s). Ready to plot. {avg_vel_msg_part}"
        # Schedule plot update and final GUI state update on main thread
        self.root.after(0, self._update_gui_plot) # Update plot first
        self.root.after(10, self._piv_finished, final_msg) # Then update rest of GUI


    def _update_gui_plot(self):
        """Updates the embedded Matplotlib canvas with the latest results."""
        # Clear previous plot
        self.plot_ax.cla()

        # Check if plot data is available
        if not hasattr(self, 'plot_data') or not self.plot_data:
            self.plot_ax.set_title("Plot will appear here after analysis")
            self.plot_ax.text(0.5, 0.5, "No data yet", ha='center', va='center', transform=self.plot_ax.transAxes)
            self.plot_canvas.draw()
            return

        # Retrieve stored data
        xw_coords = self.plot_data.get('xw')
        yw_coords = self.plot_data.get('yw')
        uw_avg = self.plot_data.get('uw_avg')
        vw_avg = self.plot_data.get('vw_avg')
        first_frame_plot_bg = self.plot_data.get('background')
        calib_params = self.plot_data.get('calib_params')
        units_label = self.plot_data.get('units_label', 'pixels')
        vel_units_label = self.plot_data.get('vel_units_label', 'pixels/frame')
        base_scale = self.plot_data.get('base_scale', 1.0)
        use_calibration = bool(calib_params and CALIBRATION_AVAILABLE)

        # Get current scale multiplier from GUI
        try:
            scale_multiplier = self.arrow_scale_multiplier_var.get()
            if scale_multiplier <= 0:
                logger.warning("Arrow scale multiplier must be positive. Using 1.0.")
                scale_multiplier = 1.0
        except tk.TclError: # Handle case where variable might not be ready
             scale_multiplier = 1.0

        # Calculate final scale for quiver
        final_quiver_scale = base_scale / scale_multiplier if base_scale > 1e-9 else None
        if final_quiver_scale == 0: final_quiver_scale = None # Use auto-scaling if base is zero

        plot_title = f"Avg Velocity ({vel_units_label})"
        plot_generated = False

        # Try to warp background if calibration available
        world_canvas = None
        plot_extent = None
        if use_calibration and first_frame_plot_bg is not None and CALIBRATION_AVAILABLE:
            try:
                logger.info("Warping background frame for GUI plot...")
                warp_output_size = (1000, 1000) # Or use a different size?
                K, dist, R, T = calib_params['K'], calib_params['dist'], calib_params['R'], calib_params['T']
                warp_result = calibration.create_world_view_image_fixed_canvas(
                    first_frame_plot_bg, K, dist, R, T, warp_output_size
                )
                if warp_result:
                    world_canvas, resolution, canvas_origin = warp_result
                    canvas_h, canvas_w = world_canvas.shape[:2]
                    origin_x, origin_y = canvas_origin
                    plot_extent = [origin_x, origin_x + canvas_w * resolution,
                                   origin_y + canvas_h * resolution, origin_y]
                    logger.info("Warping successful for GUI plot.")
                    self.plot_ax.imshow(cv2.cvtColor(world_canvas, cv2.COLOR_BGR2RGB),
                                      extent=plot_extent, origin='upper')
                    plot_title += " on Warped Coords"
                    plot_generated = True
                else:
                    logger.warning("Image warping failed for GUI plot.")
                    plot_title += " (Warping Failed)"
            except Exception as warp_err:
                 logger.error(f"Error during GUI plot warping: {warp_err}", exc_info=True)
                 plot_title += " (Warping Error)"

        # Fallback or non-calibrated: Use extent scaling
        if not plot_generated and first_frame_plot_bg is not None:
            x_min, x_max = np.nanmin(xw_coords), np.nanmax(xw_coords)
            y_min, y_max = np.nanmin(yw_coords), np.nanmax(yw_coords)
            if np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max) and x_max > x_min and y_max > y_min:
                plot_extent = [x_min, x_max, y_max, y_min]
                self.plot_ax.imshow(cv2.cvtColor(first_frame_plot_bg, cv2.COLOR_BGR2RGB), extent=plot_extent)
                plot_title += " on Scaled Coords" if not use_calibration else " (Warping Failed - Scaled BG)"
                plot_generated = True
            else:
                 plot_title += " (No Background)"
        elif not plot_generated:
             plot_title += " (No Background)"


        # --- Plot Quiver (Common logic) ---
        valid = np.isfinite(xw_coords) & np.isfinite(yw_coords) & np.isfinite(uw_avg) & np.isfinite(vw_avg)
        if np.any(valid):
            self.plot_ax.quiver(xw_coords[valid], yw_coords[valid], uw_avg[valid], vw_avg[valid],
                                color='red', scale=final_quiver_scale, scale_units='xy',
                                angles='xy', width=0.003, pivot='mid')
            plot_generated = True
        else:
            logger.warning("No valid avg vectors to plot in GUI.")
            if not plot_generated:
                 self.plot_ax.text(0.5, 0.5, 'No valid vectors', transform=self.plot_ax.transAxes, ha='center', va='center')

        # Common plot settings
        self.plot_ax.set_title(plot_title)
        self.plot_ax.set_xlabel(f"Position ({units_label})")
        self.plot_ax.set_ylabel(f"Position ({units_label})")
        self.plot_ax.set_aspect('equal')
        self.plot_ax.invert_yaxis() # Invert Y axis

        # Set limits based on plot extent if background was plotted, else based on data
        if plot_extent:
             self.plot_ax.set_xlim(plot_extent[0], plot_extent[1])
             self.plot_ax.set_ylim(plot_extent[2], plot_extent[3]) # extent is L,R,B,T; ylim is B,T
        elif np.any(valid):
            x_min_plot, x_max_plot = np.nanmin(xw_coords[valid]), np.nanmax(xw_coords[valid])
            y_min_plot, y_max_plot = np.nanmin(yw_coords[valid]), np.nanmax(yw_coords[valid])
            if np.isfinite(x_min_plot) and np.isfinite(x_max_plot) and x_max_plot > x_min_plot:
                self.plot_ax.set_xlim(x_min_plot - 0.05 * (x_max_plot - x_min_plot), x_max_plot + 0.05 * (x_max_plot - x_min_plot))
            if np.isfinite(y_min_plot) and np.isfinite(y_max_plot) and y_max_plot > y_min_plot:
                self.plot_ax.set_ylim(y_min_plot - 0.05 * (y_max_plot - y_min_plot), y_max_plot + 0.05 * (y_max_plot - y_min_plot))

        # Draw the canvas
        self.plot_canvas.draw()
        self.btn_refresh_save.config(state=tk.NORMAL) # Enable button after plotting

    def refresh_and_save_plot(self):
        """Updates the plot display and saves the current view to PNG."""
        if not hasattr(self, 'plot_data') or not self.plot_data:
            messagebox.showwarning("No Data", "No PIV results available to plot or save.")
            return

        # Update plot display with current settings
        self._update_gui_plot()

        # Save the current figure
        output_base = self.output_base_name.get()
        if not output_base:
             messagebox.showwarning("No Output Name", "Please ensure an output base name is set (load video first).")
             return
        save_path = f"{output_base}_average_flow_world_overlay.png"
        try:
            self.plot_fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved current GUI plot to: {save_path}")
            self.update_status(f"Saved current plot to {os.path.basename(save_path)}")
        except Exception as e:
            logger.error(f"Failed to save GUI plot: {e}", exc_info=True)
            messagebox.showerror("Save Error", f"Failed to save plot:\n{e}")
            self.update_status(f"Error saving plot: {e}")


    def _piv_finished(self, final_message):
        """Called in main thread when PIV processing finishes, errors, or cancels."""
        success = "ERROR" not in final_message and "Cancelled" not in final_message
        # Re-enable buttons
        is_video_loaded = bool(self.input_video_path.get() and os.path.exists(self.input_video_path.get()))
        self.btn_run.config(state=tk.NORMAL if is_video_loaded else tk.DISABLED)
        self.btn_cancel.config(state=tk.DISABLED)
        # Enable refresh/save button only if processing completed successfully and data exists
        self.btn_refresh_save.config(state=tk.NORMAL if success and hasattr(self, 'plot_data') and self.plot_data else tk.DISABLED)

        # Update progress and status
        if success and "Complete" in final_message:
             self.update_progress(100)
        else:
             self.update_progress(0) # Reset progress on error/cancel
             # Show appropriate message box
             if "ERROR" in final_message:
                  short_error = final_message.split("ERROR during processing:")[-1].strip()
                  # Schedule messagebox to run after status update
                  self.root.after(50, lambda: messagebox.showerror("Processing Error", f"PIV Processing Failed:\n{short_error}"))
             elif "Cancelled" in final_message:
                  self.root.after(50, lambda: messagebox.showwarning("Processing Cancelled", final_message))
        # Update status label last
        self.update_status(final_message)
        # Clear thread reference and cancellation flag
        self.processing_thread = None
        self.cancel_processing.clear()

# --- Main Execution ---
if __name__ == "__main__":
    # No longer need freeze_support for sequential threading
    # multiprocessing.freeze_support()

    if 'pyprocess' not in globals(): # Check OpenPIV import before starting GUI
        exit()
    root = tk.Tk()
    app = PivGuiApp(root)
    root.mainloop()
