# piv_gui_v3.py
# Creates a GUI for PIV analysis using openpiv-python
# Added more .copy() calls to attempt to fix read-only errors.
# Current time: Wednesday, April 9, 2025 at 1:00 AM EDT (Toronto, Ontario, Canada)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
import logging
import time
import math

import cv2
import numpy as np
import pandas as pd  # For CSV output
import matplotlib

# Use non-interactive backend BEFORE importing pyplot MUST BE HERE
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Attempt to import openpiv, handle if missing
try:
    from openpiv import tools, pyprocess, validation, filters, scaling
except ImportError:
    # Use tkinter messagebox BEFORE initializing the main Tk window if possible
    root_check = tk.Tk()
    root_check.withdraw() # Hide the small root window
    messagebox.showerror("Missing Library",
                         "Error: 'openpiv-python' library not found.\nPlease install it: pip install openpiv-python")
    root_check.destroy()
    exit()

# --- Logging Setup ---
logger = logging.getLogger(__name__)  # Use specific logger
logger.setLevel(logging.INFO)
# Prevent adding multiple handlers if script is re-run in some environments
if not logger.hasHandlers():
    handler = logging.StreamHandler()  # Output to console/terminal
    # More detailed format including levelname
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --- PIV Core Processing Logic ---
def process_piv_pair(frame_a, frame_b, window_size, overlap, search_area_size, sig2noise_method, dt=1.0):
    """Processes a single pair of frames for PIV."""
    try:
        if not isinstance(frame_a, np.ndarray) or not isinstance(frame_b, np.ndarray):
             raise TypeError("Input frames must be NumPy arrays.")

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.int32)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.int32)

        u, v, sig2noise = pyprocess.extended_search_area_piv(
            gray_a, gray_b,
            window_size=window_size, overlap=overlap, dt=dt,
            search_area_size=search_area_size, sig2noise_method=sig2noise_method
        )

        # --- FIX ATTEMPT 2: Return copies immediately after generation ---
        if u is not None and v is not None and sig2noise is not None:
             # Copy arrays to ensure they are writable later
             return u.copy(), v.copy(), sig2noise.copy(), gray_a.shape
        else:
             # If PIV failed, return Nones
             logger.warning("pyprocess.extended_search_area_piv returned None values.")
             return None, None, None, None
        # ---------------------------------------------------------------

    except Exception as e:
        logger.error(f"Error during core PIV processing for a frame pair: {e}", exc_info=True)
        return None, None, None, None


def validate_and_filter_piv(u, v, sig2noise, noise_threshold, filter_method, max_iter, kernel_size):
    """Validates and filters PIV results."""
    if u is None or v is None or sig2noise is None:
         logger.warning("Skipping validation/filtering due to missing input data.")
         return None, None, -1
    try:
        # Calculate mask based on S/N ratio
        invalid_mask = validation.sig2noise_val(sig2noise, threshold=noise_threshold)
        invalid_count_initial = invalid_mask.sum()

        # --- FIX ATTEMPT 1 (Keep for safety): Ensure u and v are writable copies before filtering ---
        u_writable = u.copy()
        v_writable = v.copy()
        # ------------------------------------------------------------------------------------------

        # Replace outliers using the specified method
        u_filtered, v_filtered = filters.replace_outliers(
            u_writable, v_writable, # Pass the writable copies
            invalid_mask,
            method=filter_method,
            max_iter=max_iter,
            kernel_size=kernel_size
        )
        # Check if filtering returned valid arrays
        if u_filtered is None or v_filtered is None:
             raise RuntimeError("filters.replace_outliers returned None.")

        return u_filtered, v_filtered, invalid_count_initial # Return count of invalid vectors *before* replacement
    except Exception as e:
        logger.error(f"Error during PIV validation/filtering: {e}", exc_info=True)
        # Return original vectors but indicate error with count -1
        return u, v, -1


def scale_piv_results(x, y, u, v, scaling_factor):
    """Scales PIV results to physical units."""
    if x is None or y is None or u is None or v is None:
         logger.warning("Skipping scaling due to missing input data.")
         return None, None, None, None
    try:
        x_scaled, y_scaled, u_scaled, v_scaled = scaling.uniform(x, y, u, v, scaling_factor=scaling_factor)
        # Ensure scaled results are also writable if needed later (unlikely needed, but safe)
        # return x_scaled.copy(), y_scaled.copy(), u_scaled.copy(), v_scaled.copy()
        # Let's skip copying here unless proven necessary.
        return x_scaled, y_scaled, u_scaled, v_scaled
    except Exception as e:
        logger.error(f"Error during PIV scaling: {e}", exc_info=True)
        # Return original vectors on error
        return x, y, u, v


# --- GUI Class ---
class PivGuiApp:
    """
    A Tkinter GUI application for performing PIV analysis on video files
    using the openpiv-python library.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("OpenPIV GUI Processor")
        self.root.geometry("700x700") # Adjust as needed

        # --- State Variables ---
        self.input_video_path = tk.StringVar()
        self.output_base_name = tk.StringVar()
        self.video_properties = {} # Store fps, width, height, frame_count
        self.piv_parameters = {} # Store GUI entries for the processing thread
        self.piv_param_vars = {} # Store Tkinter StringVars linked to GUI entries
        self.processing_thread = None
        self.cancel_processing = threading.Event()

        # --- Style ---
        style = ttk.Style()
        style.configure("TButton", padding=5)
        style.configure("TLabel", padding=2)
        style.configure("TEntry", padding=2)
        style.configure("TLabelframe.Label", font=('Helvetica', 10, 'bold'))


        # --- Main Frame ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(4, weight=1) # Allow status row to expand
        main_frame.columnconfigure(0, weight=1) # Allow columns to expand


        # --- Row 0: Input Video ---
        frame_input = ttk.LabelFrame(main_frame, text="1. Input Video", padding=(10, 5))
        frame_input.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        frame_input.columnconfigure(1, weight=1) # Allow filename label to expand

        btn_load = ttk.Button(frame_input, text="Load Video", command=self.load_video)
        btn_load.grid(row=0, column=0, padx=5, pady=5)

        self.lbl_filename = ttk.Label(frame_input, text="No file selected.", anchor="w", relief="sunken", padding=3)
        self.lbl_filename.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.lbl_video_info = ttk.Label(frame_input, text="Video Info: -", anchor="w")
        self.lbl_video_info.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=2)


        # --- Row 1: PIV Parameters ---
        frame_params = ttk.LabelFrame(main_frame, text="2. PIV Parameters", padding=(10, 5))
        frame_params.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        for i in [1, 4]: frame_params.columnconfigure(i, weight=1) # Let entry columns expand

        param_row = 0

        # Helper to add parameter widgets
        def add_param(label, key, default_value, row, col=0, width=10, unit="", columnspan=1, is_combo=False, combo_values=None, **kwargs):
            ttk.Label(frame_params, text=f"{label}:").grid(row=row, column=col, sticky="w", padx=(10, 2), pady=3)
            var = tk.StringVar(value=str(default_value))
            if is_combo:
                widget = ttk.Combobox(frame_params, textvariable=var, values=combo_values, state='readonly', width=width, **kwargs)
            else:
                widget = ttk.Entry(frame_params, textvariable=var, width=width, **kwargs)
            widget.grid(row=row, column=col + 1, sticky="ew", padx=2, pady=3, columnspan=columnspan)
            if unit:
                ttk.Label(frame_params, text=unit).grid(row=row, column=col + 2 + columnspan - 1, sticky="w", padx=(0, 10), pady=3)
            self.piv_param_vars[key] = var
            return var, widget

        # Define parameters using the helper
        add_param("Window Size", "window_size", 64, param_row, 0, unit="px")
        add_param("Overlap", "overlap", 32, param_row, 3, unit="px")
        param_row += 1
        add_param("Search Area Size", "search_area_size", 76, param_row, 0, unit="px")
        add_param("S/N Method", 'sig2noise_method', 'peak2peak', param_row, 3, width=12, is_combo=True, combo_values=['peak2peak', '_matrix_comparison'])
        param_row += 1
        ttk.Separator(frame_params, orient='horizontal').grid(row=param_row, columnspan=6, sticky='ew', pady=(10, 5))
        param_row += 1
        add_param("S/N Threshold", "noise_threshold", 1.15, param_row, 0)
        add_param("Filter Method", 'filter_method', 'localmean', param_row, 3, width=12, is_combo=True, combo_values=['localmean', 'median', 'gaussian'])
        param_row += 1
        add_param("Filter Max Iter", "max_iter", 3, param_row, 0)
        add_param("Filter Kernel Size", "kernel_size", 2, param_row, 3)
        param_row += 1
        ttk.Separator(frame_params, orient='horizontal').grid(row=param_row, columnspan=6, sticky='ew', pady=(10, 5))
        param_row += 1
        add_param("Processing FPS", "processing_fps", 10, param_row, 0, unit="frames/sec")
        add_param("Displacement (dt)", "displacement_frames", 1, param_row, 3, unit="frames")
        param_row += 1
        add_param("Start Frame", "start_frame", 0, param_row, 0)
        add_param("End Frame", "end_frame", 0, param_row, 3)


        # --- Row 2: Calibration ---
        frame_calib = ttk.LabelFrame(main_frame, text="3. Calibration", padding=(10, 5))
        frame_calib.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        add_param("Units per Pixel", "units_per_pixel", 1.0, 0, 0, unit="(e.g., mm/px)")


        # --- Row 3: Output & Run ---
        frame_run = ttk.LabelFrame(main_frame, text="4. Run Analysis", padding=(10, 5))
        frame_run.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        frame_run.columnconfigure(1, weight=1) # Let entry expand

        ttk.Label(frame_run, text="Output Base Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entry_output = ttk.Entry(frame_run, textvariable=self.output_base_name, width=60)
        entry_output.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        button_frame = ttk.Frame(frame_run) # Center buttons
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 5))
        self.btn_run = ttk.Button(button_frame, text="Run PIV Analysis", command=self.run_piv_analysis, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=10)
        self.btn_cancel = ttk.Button(button_frame, text="Cancel", command=self.cancel_piv_analysis, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=10)


        # --- Row 4: Status & Progress ---
        frame_status = ttk.LabelFrame(main_frame, text="5. Status", padding=(10, 5))
        frame_status.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        frame_status.columnconfigure(0, weight=1) # Label expand

        self.lbl_status = ttk.Label(frame_status, text="Load a video to begin.", anchor="w", wraplength=650)
        self.lbl_status.pack(fill=tk.X, expand=True, padx=5, pady=5)

        self.progress_bar = ttk.Progressbar(frame_status, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, expand=True, padx=5, pady=5)


    # --- GUI Methods ---
    def load_video(self):
        """Opens file dialog, loads video info, calculates default params."""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
        )
        if not filepath: return

        self.input_video_path.set(filepath)
        dir_name = os.path.dirname(filepath)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        self.output_base_name.set(os.path.join(dir_name, base_name))

        cap = None
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened(): raise IOError("Cannot open video file.")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0:
                 raise ValueError("Video properties invalid or could not be read.")

            self.video_properties = {"width": width, "height": height, "fps": fps,
                                     "frame_count": frame_count, "duration": duration}

            self.lbl_filename.config(text=os.path.basename(filepath))
            info_text = f"Info: {width}x{height} | {frame_count} frames | {fps:.2f} FPS | {duration:.2f}s duration"
            self.lbl_video_info.config(text=info_text)

            # Set Default PIV Parameters
            win_size = max(16, int(min(width, height) / 32) * 2)
            overlap = win_size // 2
            search_size = win_size + 12

            self.piv_param_vars["window_size"].set(str(win_size))
            self.piv_param_vars["overlap"].set(str(overlap))
            self.piv_param_vars["search_area_size"].set(str(search_size))
            self.piv_param_vars["processing_fps"].set(f"{fps:.1f}")
            default_end = max(0, frame_count - int(self.piv_param_vars["displacement_frames"].get()) - 1)
            self.piv_param_vars["end_frame"].set(str(default_end))
            self.piv_param_vars["start_frame"].set("0")
            self.piv_param_vars["units_per_pixel"].set("1.0") # Reset calibration

            self.lbl_status.config(text=f"Video loaded. Adjust parameters and Run Analysis.")
            self.btn_run.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Video Load Error", f"Failed to load video details:\n{e}")
            self.lbl_filename.config(text="Load failed.")
            self.lbl_video_info.config(text="Video Info: -")
            self.btn_run.config(state=tk.DISABLED)
            logger.error(f"Video loading failed for '{filepath}': {e}", exc_info=True)
        finally:
            if cap: cap.release()

    def update_status(self, message):
        """Safely update status label from any thread using Tkinter's after()."""
        if self.root:
             try:
                  if self.lbl_status.winfo_exists():
                       self.root.after(0, lambda: self.lbl_status.config(text=message))
             except tk.TclError: pass
        logger.info(message)

    def update_progress(self, value):
        """Safely update progress bar from any thread."""
        if self.root:
             try:
                 safe_value = max(0, min(100, int(value)))
                 if self.progress_bar.winfo_exists():
                     self.root.after(0, lambda: self.progress_bar.config(value=safe_value))
             except tk.TclError: pass


    def run_piv_analysis(self):
        """Validates parameters and starts the PIV processing thread."""
        if not self.input_video_path.get() or not os.path.exists(self.input_video_path.get()):
            messagebox.showerror("Error", "Please load a valid video file first.")
            return
        if self.processing_thread and self.processing_thread.is_alive():
             messagebox.showwarning("Busy", "PIV processing is already running.")
             return

        # --- Parameter Validation ---
        params = {}
        try:
            # Convert GUI vars to correct types
            int_params = ["window_size", "overlap", "search_area_size", "max_iter", "kernel_size",
                          "displacement_frames", "start_frame", "end_frame"]
            for key in int_params: params[key] = int(self.piv_param_vars[key].get())
            float_params = ["noise_threshold", "processing_fps", "units_per_pixel"]
            for key in float_params: params[key] = float(self.piv_param_vars[key].get())
            params["sig2noise_method"] = self.piv_param_vars["sig2noise_method"].get()
            params["filter_method"] = self.piv_param_vars["filter_method"].get()

            # Logic Checks
            if params["window_size"] <= 0 or params["overlap"] < 0 or params["search_area_size"] <= 0:
                raise ValueError("Window, Overlap, Search sizes must be positive.")
            if params["overlap"] >= params["window_size"]: raise ValueError("Overlap must be < Window Size.")
            if params["search_area_size"] < params["window_size"]: logger.warning("Search Area Size < Window Size.")
            if params["processing_fps"] <= 0 or params["units_per_pixel"] <= 0:
                raise ValueError("Processing FPS and Units per Pixel must be positive.")
            if params["start_frame"] < 0: raise ValueError("Start Frame cannot be negative.")
            if params["end_frame"] <= params["start_frame"]: raise ValueError("End Frame must be > Start Frame.")

            max_frame = self.video_properties.get("frame_count", float('inf')) - 1
            if params["end_frame"] > max_frame:
                logger.warning(f"End Frame adjusted to {int(max_frame)}")
                params["end_frame"] = int(max_frame)
                self.piv_param_vars["end_frame"].set(str(params["end_frame"]))
            if params["start_frame"] + params["displacement_frames"] > params["end_frame"]:
                 raise ValueError(f"Start+Displacement ({params['start_frame'] + params['displacement_frames']}) > End Frame ({params['end_frame']}). No pairs.")

        except ValueError as e: messagebox.showerror("Invalid Parameter", f"{e}"); return
        except KeyError as e: messagebox.showerror("Missing Parameter", f"Var {e}"); return

        # --- Start Background Processing ---
        self.cancel_processing.clear()
        self.btn_run.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)
        self.update_status("Starting PIV analysis...")
        self.update_progress(0)
        self.piv_parameters = params.copy() # Store validated params
        self.processing_thread = threading.Thread(target=self._piv_processing_worker, daemon=True)
        self.processing_thread.start()

    def cancel_piv_analysis(self):
         """Signals the background thread to cancel processing."""
         if self.processing_thread and self.processing_thread.is_alive():
              logger.info("Cancel request received.")
              self.cancel_processing.set()
              self.update_status("Cancellation requested... finishing current step.")
              self.btn_cancel.config(state=tk.DISABLED)
         else:
              self.update_status("No active processing to cancel.")


    def _piv_processing_worker(self):
        """
        Main PIV processing loop in background thread. Includes final calculations and saving.
        """
        start_time_proc = time.time()
        video_path = self.input_video_path.get()
        output_base = self.output_base_name.get()
        params = self.piv_parameters
        vid_props = self.video_properties

        # --- Setup ---
        frame_step_vid = vid_props.get("fps", 30) / params.get("processing_fps", 10)
        dt_frames = max(1, int(round(frame_step_vid)))
        disp_frames = params.get("displacement_frames", 1)
        start_frame_idx = params.get("start_frame", 0)
        end_frame_idx = params.get("end_frame", vid_props.get("frame_count", 1) - disp_frames - 1)
        units_per_pixel = params.get("units_per_pixel", 1.0)
        total_frames_to_process_approx = max(1, math.ceil((end_frame_idx - start_frame_idx + 1) / dt_frames))
        logger.info(f"Effective dt_frames: {dt_frames}, Disp: {disp_frames}, Frames: {start_frame_idx}-{end_frame_idx}, Approx Pairs: {total_frames_to_process_approx}")

        cap = None
        results = {'x': None, 'y': None, 'u': [], 'v': [], 'frame_index': []}
        first_frame = None
        processed_count = 0

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise IOError("Cannot open video.")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            ret_first, first_frame_read = cap.read()
            if not ret_first: raise IOError(f"Cannot read frame at index {start_frame_idx}.")
            first_frame = first_frame_read.copy()

            # --- Main Processing Loop ---
            current_pos_a = start_frame_idx
            while current_pos_a <= end_frame_idx:
                 if self.cancel_processing.is_set():
                      self.update_status("Processing cancelled by user.")
                      logger.info("Cancellation confirmed in worker thread.")
                      break

                 current_pos_b = current_pos_a + disp_frames
                 if current_pos_b > end_frame_idx or current_pos_b >= vid_props["frame_count"]:
                      logger.info(f"Frame B index ({current_pos_b}) exceeds limits. Ending loop.")
                      break

                 cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos_a); ret_a, frame_a = cap.read()
                 cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos_b); ret_b, frame_b = cap.read()
                 if not ret_a or not ret_b:
                     logger.warning(f"Could not read pair {current_pos_a}/{current_pos_b}. Stopping.")
                     break

                 status_msg = f"Processing PIV: Frames {current_pos_a} & {current_pos_b} ({processed_count + 1}/{total_frames_to_process_approx})..."
                 self.update_status(status_msg)
                 u, v, sig2noise, img_shape = process_piv_pair(
                     frame_a, frame_b, params["window_size"], params["overlap"],
                     params["search_area_size"], params["sig2noise_method"], dt=1.0
                 )
                 if u is None: logger.warning(f"Skipping pair {current_pos_a} (Core PIV error)."); current_pos_a += dt_frames; continue

                 u_f, v_f, invalid_count = validate_and_filter_piv(
                     u, v, sig2noise, params["noise_threshold"], params["filter_method"],
                     params["max_iter"], params["kernel_size"]
                 )
                 if u_f is None or invalid_count == -1: logger.warning(f"Skipping pair {current_pos_a} (Validation error)."); current_pos_a += dt_frames; continue
                 logger.debug(f"Pair {current_pos_a}: {invalid_count} vectors replaced.")

                 if results['x'] is None:
                      x_pix, y_pix = pyprocess.get_coordinates(img_shape, params["search_area_size"], params["overlap"])
                      results['x'], results['y'], _, _ = scale_piv_results(x_pix, y_pix, x_pix, y_pix, units_per_pixel)
                      if results['x'] is None: raise RuntimeError("Coordinate scaling failed.")

                 _, _, u_s, v_s = scale_piv_results(results['x'], results['y'], u_f, v_f, units_per_pixel)
                 if u_s is None: logger.warning(f"Skipping pair {current_pos_a} (Velocity scaling error)."); current_pos_a += dt_frames; continue

                 results['u'].append(u_s); results['v'].append(v_s); results['frame_index'].append(current_pos_a)
                 processed_count += 1
                 progress = min(100, int(100 * processed_count / total_frames_to_process_approx))
                 self.update_progress(progress)
                 current_pos_a += dt_frames
            # --- End of Loop ---

            if self.cancel_processing.is_set():
                 self.root.after(0, self._piv_finished, "Processing Cancelled.")
                 return

            self.update_progress(100)
            self.update_status("PIV calculation complete. Post-processing results...")

            # --- Post-Processing & Saving ---
            if not results['u']: raise ValueError("No PIV results generated.")

            logger.info("Stacking results..."); self.update_status("Stacking results...")
            u_all = np.stack(results['u'], axis=0); v_all = np.stack(results['v'], axis=0)
            frame_indices = np.array(results['frame_index']); x_coords = results['x']; y_coords = results['y']

            # --- 1. Average PNG ---
            self.update_status("Calculating time average & saving plot...")
            logger.info("Calculating time average & saving plot...")
            u_avg = np.mean(u_all, axis=0) if processed_count > 0 else np.zeros_like(x_coords)
            v_avg = np.mean(v_all, axis=0) if processed_count > 0 else np.zeros_like(y_coords)
            avg_png_path = f"{output_base}_average.png"
            try:
                fig, ax = plt.subplots(figsize=(10, 10 * vid_props['height'] / vid_props['width']))
                ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
                mag_max = np.max(np.sqrt(u_avg**2 + v_avg**2)) if processed_count > 0 else 1.0
                quiver_scale = mag_max * 20 if mag_max > 1e-9 else 1.0 # Avoid zero scale
                ax.quiver(x_coords, y_coords, u_avg, v_avg, color='red',
                          scale=quiver_scale, scale_units='width', angles='xy', width=0.0025, pivot='mid')
                title_suffix = f"({processed_count} pairs: frames {frame_indices[0]}-{frame_indices[-1]})" if processed_count > 0 else "(No pairs processed)"
                ax.set_title(f"Time-Averaged PIV {title_suffix}")
                ax.set_xlabel(f"Position ({units_per_pixel:.3g} units/pixel)")
                ax.set_ylabel(f"Position ({units_per_pixel:.3g} units/pixel)")
                fig.savefig(avg_png_path, dpi=150, bbox_inches='tight')
                plt.close(fig); logger.info(f"Saved average plot: {avg_png_path}")
            except Exception as e: logger.error(f"Failed to save average plot: {e}", exc_info=True); self.update_status(f"Warn: Error saving plot: {e}")

            # --- 2. Raw CSV ---
            self.update_status("Saving raw data to CSV...")
            logger.info("Saving raw data to CSV...")
            csv_path = f"{output_base}_raw_data.csv"
            try:
                if processed_count > 0:
                    n_steps, ny, nx = u_all.shape; n_vec = ny * nx
                    x_flat=np.tile(x_coords.flatten(),n_steps); y_flat=np.tile(y_coords.flatten(),n_steps)
                    frame_col = np.repeat(frame_indices, n_vec); u_flat=u_all.flatten(); v_flat=v_all.flatten()
                    df = pd.DataFrame({'frame_index':frame_col,'x_position_units':x_flat,'y_position_units':y_flat,
                                       'u_velocity_units_time':u_flat,'v_velocity_units_time':v_flat})
                    df.to_csv(csv_path, index=False, float_format='%.5f'); logger.info(f"Saved raw data: {csv_path}")
                else: logger.warning("No data to save to CSV.")
            except Exception as e: logger.error(f"Failed to save CSV: {e}", exc_info=True); self.update_status(f"Warn: Error saving CSV: {e}")

            # --- 3. Stats Calc (Outlier Filtered) ---
            self.update_status("Calculating final statistics...")
            logger.info("Calculating final statistics...")
            avg_vel_mag = 0.0; avg_rms_fluctuation = 0.0
            if processed_count > 0:
                magnitude_avg_map = np.sqrt(u_avg**2 + v_avg**2)
                perc_5, perc_95 = np.percentile(magnitude_avg_map, [5, 95])
                mask_mag = (magnitude_avg_map >= perc_5) & (magnitude_avg_map <= perc_95)
                avg_vel_mag = np.mean(magnitude_avg_map[mask_mag]) if np.any(mask_mag) else np.mean(magnitude_avg_map)
                logger.info(f"Avg Vel Mag (5-95 perc spatially): {avg_vel_mag:.5f}")
            if processed_count > 1:
                 u_prime=u_all-u_avg; v_prime=v_all-v_avg; mag_prime_sq=u_prime**2+v_prime**2
                 mean_mag_prime_sq=np.mean(mag_prime_sq, axis=0)
                 rms_map=np.sqrt(mean_mag_prime_sq + 1e-12)
                 perc_5_rms, perc_95_rms = np.percentile(rms_map, [5, 95])
                 mask_rms = (rms_map >= perc_5_rms) & (rms_map <= perc_95_rms)
                 avg_rms_fluctuation = np.mean(rms_map[mask_rms]) if np.any(mask_rms) else np.mean(rms_map)
                 logger.info(f"Avg RMS Fluct (5-95 perc spatially): {avg_rms_fluctuation:.5f}")
            avg_vel_msg_part = f"Avg Vel Mag: {avg_vel_mag:.4f}, Avg RMS Fluct: {avg_rms_fluctuation:.4f} units/time_step (spatially filtered 5-95th perc)"

            # --- 4. Summary TXT ---
            self.update_status("Saving summary text file...")
            logger.info("Saving summary text file...")
            summary_txt_path = f"{output_base}_summary.txt"
            try:
                with open(summary_txt_path, 'w') as f:
                    f.write(f"PIV Analysis Summary\nInput Video: {os.path.basename(video_path)}\n" + "="*40 + "\n")
                    f.write(f"Processed Frame Pairs: {processed_count}\n")
                    if frame_indices.size > 0: f.write(f"Frame Index Range (Start): {frame_indices[0]} to {frame_indices[-1]}\n")
                    else: f.write("Frame Index Range: N/A\n")
                    f.write("-" * 40 + "\nKey Parameters Used:\n")
                    for k, v in params.items(): f.write(f"  {k}: {v}\n") # Write all used params
                    f.write("-" * 40 + "\nResults (Spatial Outliers Excluded, typically 5th-95th percentile):\n")
                    f.write(f"Average Velocity Magnitude: {avg_vel_mag:.5f} (units/time_step)\n")
                    f.write(f"Average RMS Fluctuation: {avg_rms_fluctuation:.5f} (units/time_step)\n\nNotes:\n")
                    f.write(f"- 'units' based on calibration ({units_per_pixel:.4g} units/pixel).\n")
                    fps_val = vid_props.get('fps'); fps_str = f"{fps_val:.1f}" if fps_val else 'N/A'
                    time_step_sec = params['displacement_frames'] / fps_val if fps_val else float('nan')
                    f.write(f"- 'time_step' = interval between frames A & B ({params['displacement_frames']} frame(s) / {fps_str} FPS = ~{time_step_sec:.4f} sec).\n")
                    f.write("- Avg RMS Fluct = spatially averaged RMS of velocity deviations from the time-mean.\n")
                    f.write("- Averages exclude spatial outliers (values outside central 90% percentile range).\n")
                logger.info(f"Saved summary: {summary_txt_path}")
            except Exception as e: logger.error(f"Failed to save summary TXT: {e}", exc_info=True); self.update_status(f"Warn: Error saving summary TXT: {e}")

            # --- Final Status Update (Success) ---
            total_time = time.time() - start_time_proc
            final_msg = f"Processing Complete! ({processed_count} pairs in {total_time:.1f}s). Outputs saved. {avg_vel_msg_part}"
            self.root.after(0, self._piv_finished, final_msg)

        except Exception as e:
            logger.error("Error in PIV processing worker:", exc_info=True)
            self.update_status(f"ERROR during processing: {e}")
            self.root.after(0, self._piv_finished, f"ERROR during processing: {e}")
        finally:
            if cap: cap.release(); logger.info("Video capture released.")
            plt.close('all')

    def _piv_finished(self, final_message):
        """Called in main thread when PIV processing finishes, errors, or cancels."""
        success = "ERROR" not in final_message and "Cancelled" not in final_message
        if success and "Complete" in final_message:
             final_message += " Summary text file also created."
             self.update_progress(100)
             # Optionally show success messagebox after a short delay
             # self.root.after(100, lambda: messagebox.showinfo("Processing Complete", final_message))
        else:
             self.update_progress(0)
             if "ERROR" in final_message: self.root.after(100, lambda: messagebox.showerror("Processing Error", final_message))
             elif "Cancelled" in final_message: self.root.after(100, lambda: messagebox.showwarning("Processing Cancelled", final_message))

        self.update_status(final_message)
        is_video_loaded = bool(self.input_video_path.get() and os.path.exists(self.input_video_path.get()))
        self.btn_run.config(state=tk.NORMAL if is_video_loaded else tk.DISABLED)
        self.btn_cancel.config(state=tk.DISABLED)
        self.processing_thread = None
        self.cancel_processing.clear()


# --- Main Execution ---
if __name__ == "__main__":
    if 'pyprocess' not in globals(): # Check if openpiv imported correctly
        print("OpenPIV library failed to import earlier. Exiting.")
        try: # Attempt to show GUI error message
              root_err = tk.Tk(); root_err.withdraw()
              messagebox.showerror("Startup Error", "OpenPIV library not found or failed to import.")
              root_err.destroy()
        except Exception: pass # Ignore if Tkinter itself fails
        exit()

    root = tk.Tk()
    app = PivGuiApp(root)
    root.mainloop()