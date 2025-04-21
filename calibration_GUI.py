import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Scale, HORIZONTAL
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk, ImageDraw
import os
import threading # To avoid freezing the GUI during processing
import logging # Import logging library
import math # For measurement tool calculation
# Import Matplotlib and necessary Tkinter backend components
import matplotlib
matplotlib.use('TkAgg') # Must be called before importing pyplot and creating Tk window
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Setup Logging ---
log_format = '%(asctime)s - %(levelname)s - [CalibGUI] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
# Prevent duplicate handlers if run multiple times in interactive session
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False # Prevent root logger from also printing

# --- Import your calibration functions ---
# Assume calibration.py is in the same directory
try:
    import calibration
    # Check if the required function exists
    if not hasattr(calibration, 'pixel_to_world_via_plane'):
        raise AttributeError("Function 'pixel_to_world_via_plane' not found in calibration.py")
    logger.info("Successfully imported calibration functions.")
except ImportError:
    logger.error("Could not find calibration.py. Make sure it's in the same directory or Python path.")
    messagebox.showerror("Import Error", "Could not find calibration.py. Make sure it's in the same directory or Python path.")
    exit()
except AttributeError as e:
     logger.error(f"Attribute Error importing from calibration.py: {e}")
     messagebox.showerror("Attribute Error", f"Error importing from calibration.py. Check function names:\n{e}")
     exit()


# --- Constants ---
IMG_DISPLAY_WIDTH = 480 # Adjust as needed
IMG_DISPLAY_HEIGHT = 360 # Adjust as needed
POINT_RADIUS = 5 # Smaller radius for clicking accuracy
CLICK_THRESHOLD_SQ = (POINT_RADIUS * 2)**2 # Squared threshold for faster distance check
ENHANCED_COLOR = "gray" # Colormap for enhanced frame display
MEASURE_BOX_COLOR = "cyan"

# --- Main Application Class ---
class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Calibration GUI")
        # Adjusted width to accommodate plot next to controls
        self.root.geometry("1800x850") # Made wider
        logger.info("Initializing Calibration GUI.")

        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam') # Or 'alt', 'default', 'classic'

        # --- State Variables ---
        self.image_path = None
        self.original_image = None # OpenCV image (BGR)
        self.display_image_orig_pil = None # PIL Image for display base
        self.tk_image_orig = None
        self.orig_img_scale_x = 1.0 # Scaling factor for display
        self.orig_img_scale_y = 1.0 # Scaling factor for display
        self.orig_img_offset_x = 0 # Offset for centering
        self.orig_img_offset_y = 0 # Offset for centering
        self.correlation_map = None # Numpy array
        self.display_image_enhanced_pil = None # PIL Image for display
        self.tk_image_enhanced = None
        self.correlation_bool = None # Boolean map after thresholding
        self.components_centroids = None # Torch tensor after component filter
        self.display_image_components_pil = None # PIL Image for display
        self.tk_image_components = None
        self.original_final_centroids = None # Store original filtered centroids for reset
        self.final_centroids = None # Torch tensor after centroid filter (potentially modified)
        self.display_image_final_pil = None # PIL Image for display
        self.tk_image_final = None
        self.calibration_params = {} # To store K, dist, R, T, rot_angle
        self.image_size = None # (width, height)
        self.processing_lock = threading.Lock() # Prevent concurrent processing
        # Measurement tool state
        self.measure_start_x = None
        self.measure_start_y = None
        self.measure_rect_id = None
        # Dot spacing state
        self.dot_spacing_var = tk.DoubleVar(value=6.35) # Default value
        self.dot_spacing = 6.35 # Actual value used in calculations
        logger.debug("State variables initialized.")

        # --- GUI Setup ---
        # Main PanedWindow for vertical split (Controls+Plot / Visualizations)
        self.main_pane = ttk.PanedWindow(root, orient=tk.VERTICAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Top frame contains controls on the left and plot on the right
        self.top_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.top_frame, weight=0) # Don't expand vertically initially

        # --- Controls Widgets Frame (inside top_frame, on the left) ---
        self.controls_frame = ttk.Frame(self.top_frame, padding="10")
        self.controls_frame.grid(row=0, column=0, sticky="nsew")

        # --- Plot Frame (inside top_frame, on the right) ---
        self.plot_frame = ttk.LabelFrame(self.top_frame, text="Reprojection Plot", padding="5")
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)

        # Configure resizing behavior for top_frame columns
        self.top_frame.columnconfigure(0, weight=0) # Controls don't expand horizontally much
        self.top_frame.columnconfigure(1, weight=1) # Plot area expands horizontally
        self.top_frame.rowconfigure(0, weight=1) # Allow vertical expansion within top frame

        # Bottom frame for image visualizations (using PanedWindow for horizontal split)
        self.vis_pane = ttk.PanedWindow(self.main_pane, orient=tk.HORIZONTAL)
        self.main_pane.add(self.vis_pane, weight=1) # Allow vertical expansion

        # Left Visualization Frame (Original + Components)
        self.vis_left_frame = ttk.Frame(self.vis_pane, padding="5")
        self.vis_pane.add(self.vis_left_frame, weight=1)
        self.vis_left_frame.rowconfigure(0, weight=1) # Canvas row
        self.vis_left_frame.rowconfigure(2, weight=1) # Canvas row
        self.vis_left_frame.columnconfigure(0, weight=1)

        # Right Visualization Frame (Enhanced + Final)
        self.vis_right_frame = ttk.Frame(self.vis_pane, padding="5")
        self.vis_pane.add(self.vis_right_frame, weight=1)
        self.vis_right_frame.rowconfigure(0, weight=1) # Canvas row
        self.vis_right_frame.rowconfigure(2, weight=1) # Canvas row
        self.vis_right_frame.columnconfigure(0, weight=1)

        logger.debug("GUI layout frames created.")


        # --- Control Widgets (placed in self.controls_frame) ---
        # File Loading
        self.load_button = ttk.Button(self.controls_frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_label = ttk.Label(self.controls_frame, text="No image loaded.", width=40)
        self.file_label.grid(row=0, column=1, columnspan=1, padx=5, pady=5, sticky="ew")

        # Dot Spacing Input
        ttk.Label(self.controls_frame, text="Dot Spacing (mm):").grid(row=0, column=2, padx=(10, 2), pady=5, sticky="e")
        self.dot_spacing_entry = ttk.Entry(self.controls_frame, textvariable=self.dot_spacing_var, width=8)
        self.dot_spacing_entry.grid(row=0, column=3, padx=2, pady=5, sticky="w")
        self.set_spacing_button = ttk.Button(self.controls_frame, text="Set Spacing", command=self._confirm_dot_spacing)
        self.set_spacing_button.grid(row=0, column=4, padx=(5, 10), pady=5, sticky="w")

        # Parameters Frame
        self.params_frame = ttk.LabelFrame(self.controls_frame, text="Parameters & Steps", padding="10")
        self.params_frame.grid(row=1, column=0, columnspan=5, padx=5, pady=5, sticky="nsew") # Span 5 cols
        self.params_frame.columnconfigure(1, weight=1) # Make slider column expand

        # --- Step 1: Enhancement ---
        ttk.Label(self.params_frame, text="1. Enhance Frame").grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="w")
        ttk.Label(self.params_frame, text="Circle Size (px):").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        self.circle_size_var = tk.IntVar(value=51)
        self.circle_size_scale = ttk.Scale(self.params_frame, from_=5, to=101, orient=HORIZONTAL, variable=self.circle_size_var, command=self._update_effective_circle_label)
        self.circle_size_scale.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.effective_circle_size_var = tk.StringVar(value=f"{self._get_odd_circle_size(self.circle_size_var.get())}")
        self.effective_circle_size_label = ttk.Label(self.params_frame, textvariable=self.effective_circle_size_var, width=12)
        self.effective_circle_size_label.grid(row=1, column=2, padx=5, pady=2, sticky="w")
        self._update_effective_circle_label()

        self.enhance_button = ttk.Button(self.params_frame, text="Run Enhance", command=self.run_enhancement_threaded, state=tk.DISABLED)
        self.enhance_button.grid(row=1, column=3, padx=10, pady=5)
        ttk.Label(self.params_frame, text="(Drag on Original Image to measure)").grid(row=2, column=1, columnspan=3, padx=5, pady=0, sticky="w")

        # --- Step 2: Component Filter ---
        ttk.Label(self.params_frame, text="2. Filter Components").grid(row=3, column=0, columnspan=4, padx=5, pady=(10,5), sticky="w")
        ttk.Label(self.params_frame, text="Corr. Thresh (0-1):").grid(row=4, column=0, padx=5, pady=2, sticky="e")
        self.corr_thresh_var = tk.DoubleVar(value=0.3)
        self.corr_thresh_scale = ttk.Scale(self.params_frame, from_=0.0, to=1.0, orient=HORIZONTAL, variable=self.corr_thresh_var)
        self.corr_thresh_scale.grid(row=4, column=1, padx=5, pady=2, sticky="ew")
        self.corr_thresh_label = ttk.Label(self.params_frame, text=f"{self.corr_thresh_var.get():.2f}", width=4)
        self.corr_thresh_label.grid(row=4, column=2, padx=5, pady=2, sticky="w")
        ttk.Label(self.params_frame, text="Comp. StdDev Thresh:").grid(row=5, column=0, padx=5, pady=2, sticky="e")
        self.comp_std_thresh_var = tk.DoubleVar(value=2.0)
        self.comp_std_thresh_scale = ttk.Scale(self.params_frame, from_=0.1, to=5.0, orient=HORIZONTAL, variable=self.comp_std_thresh_var)
        self.comp_std_thresh_scale.grid(row=5, column=1, padx=5, pady=2, sticky="ew")
        self.comp_std_thresh_label = ttk.Label(self.params_frame, text=f"{self.comp_std_thresh_var.get():.1f}", width=4)
        self.comp_std_thresh_label.grid(row=5, column=2, padx=5, pady=2, sticky="w")
        self.comp_filter_button = ttk.Button(self.params_frame, text="Run Filter Comp.", command=self.run_component_filter_threaded, state=tk.DISABLED)
        self.comp_filter_button.grid(row=4, column=3, rowspan=2, padx=10, pady=5, sticky="ns")
        self.corr_thresh_var.trace_add("write", lambda *args: self.corr_thresh_label.config(text=f"{self.corr_thresh_var.get():.2f}"))
        self.comp_std_thresh_var.trace_add("write", lambda *args: self.comp_std_thresh_label.config(text=f"{self.comp_std_thresh_var.get():.1f}"))

        # --- Step 3: Centroid Filter ---
        ttk.Label(self.params_frame, text="3. Filter Centroids").grid(row=6, column=0, columnspan=4, padx=5, pady=(10,5), sticky="w")
        ttk.Label(self.params_frame, text="Dist. StdDev Thresh:").grid(row=7, column=0, padx=5, pady=2, sticky="e")
        self.cent_thresh_var = tk.DoubleVar(value=3.0)
        self.cent_thresh_scale = ttk.Scale(self.params_frame, from_=0.1, to=5.0, orient=HORIZONTAL, variable=self.cent_thresh_var)
        self.cent_thresh_scale.grid(row=7, column=1, padx=5, pady=2, sticky="ew")
        self.cent_thresh_label = ttk.Label(self.params_frame, text=f"{self.cent_thresh_var.get():.1f}", width=4)
        self.cent_thresh_label.grid(row=7, column=2, padx=5, pady=2, sticky="w")
        self.centroid_buttons_frame = ttk.Frame(self.params_frame)
        self.centroid_buttons_frame.grid(row=7, column=3, padx=10, pady=0, sticky="w")
        self.cent_filter_button = ttk.Button(self.centroid_buttons_frame, text="Run Filter Cent.", command=self.run_centroid_filter_threaded, state=tk.DISABLED)
        self.cent_filter_button.pack(side=tk.LEFT, padx=(0, 5))
        self.reset_centroids_button = ttk.Button(self.centroid_buttons_frame, text="Reset Points", command=self._reset_final_centroids, state=tk.DISABLED)
        self.reset_centroids_button.pack(side=tk.LEFT)
        self.cent_thresh_var.trace_add("write", lambda *args: self.cent_thresh_label.config(text=f"{self.cent_thresh_var.get():.1f}"))

        # --- Step 4: Calibration Actions Frame ---
        self.calib_frame = ttk.LabelFrame(self.controls_frame, text="4. Calibration", padding="10")
        self.calib_frame.grid(row=2, column=0, columnspan=5, padx=5, pady=5, sticky="nsew") # Span 5 cols
        self.calibrate_button = ttk.Button(self.calib_frame, text="Run Calibration", command=self.run_calibration_threaded, state=tk.DISABLED)
        self.calibrate_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.save_button = ttk.Button(self.calib_frame, text="Save Parameters", command=self.save_parameters, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.status_label = ttk.Label(self.calib_frame, text="Status: Idle", relief=tk.SUNKEN, padding=2)
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
        logger.debug("Control widgets placed in controls_frame.")

        # --- Matplotlib Plot Setup (placed in self.plot_frame) ---
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Reprojection Plot (World Coords)")
        self.ax.set_xlabel("World X (mm)")
        self.ax.set_ylabel("World Y (mm)")
        self.ax.grid(True)
        self.ax.axis('equal')
        self.fig.tight_layout()

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.plot_canvas_widget = self.plot_canvas.get_tk_widget()
        self.plot_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Use pack to fill the plot_frame
        logger.debug("Matplotlib plot area created and placed next to controls.")

        # --- Visualization Canvases (placed in self.vis_left_frame and self.vis_right_frame) ---
        # Left Frame
        self.canvas_orig = tk.Canvas(self.vis_left_frame, bg='dark gray', width=IMG_DISPLAY_WIDTH, height=IMG_DISPLAY_HEIGHT, relief=tk.SUNKEN, borderwidth=1)
        self.canvas_orig.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Label(self.vis_left_frame, text="Original Image (Drag to Measure Circle)", anchor="center").grid(row=1, column=0, sticky="ew")

        self.canvas_components = tk.Canvas(self.vis_left_frame, bg='dark gray', width=IMG_DISPLAY_WIDTH, height=IMG_DISPLAY_HEIGHT, relief=tk.SUNKEN, borderwidth=1)
        self.canvas_components.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Label(self.vis_left_frame, text="Filtered Components (Orange)", anchor="center").grid(row=3, column=0, sticky="ew")

        # Right Frame
        self.canvas_enhanced = tk.Canvas(self.vis_right_frame, bg='dark gray', width=IMG_DISPLAY_WIDTH, height=IMG_DISPLAY_HEIGHT, relief=tk.SUNKEN, borderwidth=1)
        self.canvas_enhanced.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Label(self.vis_right_frame, text="Enhanced Frame (Correlation)", anchor="center").grid(row=1, column=0, sticky="ew")

        self.canvas_final = tk.Canvas(self.vis_right_frame, bg='dark gray', width=IMG_DISPLAY_WIDTH, height=IMG_DISPLAY_HEIGHT, relief=tk.SUNKEN, borderwidth=1)
        self.canvas_final.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Label(self.vis_right_frame, text="Final Centroids (Red - Click to Remove)", anchor="center").grid(row=3, column=0, sticky="ew")
        logger.debug("Visualization canvases created and placed.")


        # Bind resize events to redraw image canvases
        self.canvas_orig.bind("<Configure>", lambda e: self._display_on_canvas(self.canvas_orig, self.display_image_orig_pil, ref_name="tk_image_orig"))
        self.canvas_enhanced.bind("<Configure>", lambda e: self._display_on_canvas(self.canvas_enhanced, self.display_image_enhanced_pil, ref_name="tk_image_enhanced"))
        self.canvas_components.bind("<Configure>", lambda e: self._display_on_canvas(self.canvas_components, self.display_image_orig_pil, ref_name="tk_image_components"))
        self.canvas_final.bind("<Configure>", lambda e: self._display_on_canvas(self.canvas_final, self.display_image_orig_pil, ref_name="tk_image_final"))

        # Bind mouse events for measurement tool on original canvas
        self.canvas_orig.bind("<ButtonPress-1>", self._start_measure)
        self.canvas_orig.bind("<B1-Motion>", self._drag_measure)
        self.canvas_orig.bind("<ButtonRelease-1>", self._end_measure)

        # Bind click event for point removal on final canvas
        self.canvas_final.bind("<Button-1>", self._remove_final_centroid)


    # --- Helper Functions ---
    def _update_status(self, message, level=logging.INFO):
        """Updates the status label and logs the message."""
        logger.log(level, message)
        self.root.after(0, lambda: self.status_label.config(text=f"Status: {message}"))

    def _get_odd_circle_size(self, size):
        """Ensures the circle size is odd."""
        size = int(round(size))
        if size % 2 == 0:
            return size + 1
        return size

    def _update_effective_circle_label(self, *args):
        """Updates the label showing the effective odd circle size."""
        current_val = self.circle_size_var.get()
        odd_val = self._get_odd_circle_size(current_val)
        self.effective_circle_size_var.set(f"{odd_val}")

    def _resize_image_to_fit(self, pil_image, canvas_width, canvas_height):
        """Resizes a PIL image to fit the target dimensions while maintaining aspect ratio."""
        if pil_image is None or canvas_width <= 1 or canvas_height <= 1:
            return None, 1.0, 1.0, 0, 0

        img_w, img_h = pil_image.size
        aspect_ratio = img_w / img_h
        target_aspect = canvas_width / canvas_height

        if aspect_ratio > target_aspect:
            new_w = canvas_width
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = canvas_height
            new_w = int(new_h * aspect_ratio)

        scale_x = new_w / img_w if img_w > 0 else 1.0
        scale_y = new_h / img_h if img_h > 0 else 1.0

        offset_x = (canvas_width - new_w) // 2
        offset_y = (canvas_height - new_h) // 2

        resized_img = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return resized_img, scale_x, scale_y, offset_x, offset_y

    def _display_on_canvas(self, canvas, pil_image, ref_name):
        """Displays a PIL image on the specified canvas, handling resizing and point overlay."""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        base_pil_image = pil_image

        if canvas == self.canvas_components:
            base_pil_image = self.display_image_orig_pil
            if base_pil_image is not None and self.components_centroids is not None and len(self.components_centroids) > 0:
                 effective_radius = max(1, self._get_odd_circle_size(self.circle_size_var.get()) // 2)
                 img_to_display = self._overlay_points(base_pil_image.copy(), self.components_centroids, color="orange", radius=effective_radius)
                 self.display_image_components_pil = img_to_display
            else:
                 img_to_display = base_pil_image
                 self.display_image_components_pil = img_to_display
        elif canvas == self.canvas_final:
            base_pil_image = self.display_image_orig_pil
            if base_pil_image is not None and self.final_centroids is not None and len(self.final_centroids) > 0:
                 effective_radius = max(1, self._get_odd_circle_size(self.circle_size_var.get()) // 2)
                 img_to_display = self._overlay_points(base_pil_image.copy(), self.final_centroids, color="red", radius=effective_radius)
                 self.display_image_final_pil = img_to_display
            else:
                 img_to_display = base_pil_image
                 self.display_image_final_pil = img_to_display
        else:
            img_to_display = pil_image

        if img_to_display is None or canvas_width <= 1 or canvas_height <= 1:
            canvas.delete("all")
            setattr(self, ref_name, None)
            if canvas == self.canvas_orig:
                self.orig_img_scale_x = 1.0
                self.orig_img_scale_y = 1.0
                self.orig_img_offset_x = 0
                self.orig_img_offset_y = 0
            return

        orig_pil_for_scale = self.display_image_orig_pil if self.display_image_orig_pil else img_to_display

        resized_img, current_scale_x, current_scale_y, current_offset_x, current_offset_y = self._resize_image_to_fit(
            orig_pil_for_scale, canvas_width, canvas_height
        )

        self.orig_img_scale_x = current_scale_x
        self.orig_img_scale_y = current_scale_y
        self.orig_img_offset_x = current_offset_x
        self.orig_img_offset_y = current_offset_y

        if resized_img:
             new_w, new_h = resized_img.size
             display_resized_img = img_to_display.resize((new_w, new_h), Image.Resampling.LANCZOS)
             tk_image = ImageTk.PhotoImage(display_resized_img)
             setattr(self, ref_name, tk_image)
             canvas.delete("all")
             canvas.create_image(current_offset_x, current_offset_y, anchor=tk.NW, image=tk_image)
        else:
             canvas.delete("all")
             setattr(self, ref_name, None)


    def _overlay_points(self, base_pil_image, points_tensor, color="red", radius=POINT_RADIUS):
        """Overlays points from a torch tensor onto a PIL image for display."""
        if base_pil_image is None or points_tensor is None or len(points_tensor) == 0:
            return base_pil_image

        img_with_overlay = base_pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(img_with_overlay)
        points = points_tensor.cpu().numpy()

        for x, y in points:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=color, fill=color
            )
        return img_with_overlay

    def _run_in_thread(self, target_func, *args):
        """Runs a function in a separate thread to avoid blocking the GUI."""
        if not self.processing_lock.acquire(blocking=False):
            self._update_status("Processing already in progress... Please wait.", level=logging.WARNING)
            self.root.after(0, lambda: messagebox.showwarning("Busy", "Processing is already running. Please wait for it to complete."))
            return

        logger.info(f"Starting background thread for: {target_func.__name__}")
        def thread_wrapper():
            try:
                target_func(*args)
            except Exception as e:
                 logger.error(f"Error in background thread ({target_func.__name__}): {e}", exc_info=True)
                 self.root.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred in {target_func.__name__}:\n{e}"))
                 self._update_status(f"Error during {target_func.__name__}", level=logging.ERROR)
            finally:
                logger.info(f"Background thread finished for: {target_func.__name__}")
                self.processing_lock.release()

        thread = threading.Thread(target=thread_wrapper)
        thread.daemon = True
        thread.start()

    # --- Coordinate Conversion ---
    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """Converts canvas coordinates to original image coordinates."""
        if self.orig_img_scale_x == 0 or self.orig_img_scale_y == 0:
            logger.warning("Cannot convert coordinates, image scale is zero.")
            return None, None

        scale_x = self.orig_img_scale_x if self.orig_img_scale_x != 0 else 1.0
        scale_y = self.orig_img_scale_y if self.orig_img_scale_y != 0 else 1.0

        img_x = (canvas_x - self.orig_img_offset_x) / scale_x
        img_y = (canvas_y - self.orig_img_offset_y) / scale_y
        return img_x, img_y

    def _image_to_canvas_coords(self, img_x, img_y):
        """Converts original image coordinates to canvas coordinates."""
        canvas_x = (img_x * self.orig_img_scale_x) + self.orig_img_offset_x
        canvas_y = (img_y * self.orig_img_scale_y) + self.orig_img_offset_y
        return canvas_x, canvas_y

    # --- Measurement Tool Methods ---
    def _start_measure(self, event):
        """Handles mouse button press for measurement."""
        if self.original_image is None: return
        self.measure_start_x = event.x
        self.measure_start_y = event.y
        self.measure_rect_id = self.canvas_orig.create_rectangle(
            self.measure_start_x, self.measure_start_y,
            self.measure_start_x, self.measure_start_y,
            outline=MEASURE_BOX_COLOR, width=2
        )
        logger.debug(f"Measurement started at canvas coords: ({event.x}, {event.y})")

    def _drag_measure(self, event):
        """Handles mouse drag for measurement."""
        if self.measure_rect_id is None: return
        self.canvas_orig.coords(
            self.measure_rect_id,
            self.measure_start_x, self.measure_start_y,
            event.x, event.y
        )

    def _end_measure(self, event):
        """Handles mouse button release for measurement."""
        if self.measure_rect_id is None: return

        end_x, end_y = event.x, event.y
        start_x, start_y = self.measure_start_x, self.measure_start_y

        self.canvas_orig.delete(self.measure_rect_id)
        self.measure_rect_id = None
        self.measure_start_x = None
        self.measure_start_y = None

        canvas_dx = abs(end_x - start_x)
        canvas_dy = abs(end_y - start_y)
        canvas_size = math.sqrt(canvas_dx**2 + canvas_dy**2)

        avg_scale = (self.orig_img_scale_x + self.orig_img_scale_y) / 2.0
        if avg_scale == 0:
            logger.warning("Cannot calculate measured size, image scale is zero.")
            return

        image_pixel_size = int(round(canvas_size / avg_scale))
        logger.info(f"Measurement ended. Canvas size: {canvas_size:.1f}, Image pixel size: {image_pixel_size}")

        if image_pixel_size > 0:
            self.circle_size_var.set(image_pixel_size)
            self._update_status(f"Measured size: {image_pixel_size}px. Effective odd size: {self._get_odd_circle_size(image_pixel_size)}px. Click 'Run Enhance'.")
        else:
            logger.warning("Measured size is zero, not updating circle size.")

    # --- Dot Spacing Confirmation ---
    def _confirm_dot_spacing(self):
        """Validates and sets the dot spacing value."""
        try:
            value = self.dot_spacing_var.get()
            if value <= 0:
                raise ValueError("Dot spacing must be positive.")
            self.dot_spacing = value
            logger.info(f"Dot spacing set to: {self.dot_spacing} mm")
            self._update_status(f"Dot spacing confirmed: {self.dot_spacing} mm.")
        except (tk.TclError, ValueError) as e:
            logger.error(f"Invalid dot spacing input: {e}")
            messagebox.showerror("Invalid Input", f"Please enter a valid positive number for dot spacing.\nError: {e}")
            self.dot_spacing_var.set(self.dot_spacing)
            self._update_status("Invalid dot spacing input.", level=logging.ERROR)

    # --- Action Functions ---
    def load_image(self):
        logger.info("Load image button clicked.")
        path = filedialog.askopenfilename(
            title="Select Calibration Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.tif *.bmp")]
        )
        if not path:
            logger.warning("No image file selected.")
            return

        self.image_path = path
        self.file_label.config(text=os.path.basename(path))
        self._update_status(f"Loading image: {os.path.basename(path)}...")
        try:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                raise ValueError("Could not read image file using OpenCV.")
            self.image_size = (self.original_image.shape[1], self.original_image.shape[0])
            logger.info(f"Image loaded successfully. Size: {self.image_size}")

            img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image_orig_pil = Image.fromarray(img_rgb)

            self._display_on_canvas(self.canvas_orig, self.display_image_orig_pil, ref_name="tk_image_orig")

            logger.debug("Resetting intermediate steps and enabling enhancement.")
            self.correlation_map = None
            self.correlation_bool = None
            self.components_centroids = None
            self.original_final_centroids = None
            self.final_centroids = None
            self.calibration_params = {}
            self.display_image_enhanced_pil = None
            self.display_image_components_pil = None
            self.display_image_final_pil = None
            self._display_on_canvas(self.canvas_enhanced, None, ref_name="tk_image_enhanced")
            self._display_on_canvas(self.canvas_components, None, ref_name="tk_image_components")
            self._display_on_canvas(self.canvas_final, None, ref_name="tk_image_final")
            # Clear the plot canvas as well
            self._clear_reprojection_plot()


            self.enhance_button.config(state=tk.NORMAL)
            self.comp_filter_button.config(state=tk.DISABLED)
            self.cent_filter_button.config(state=tk.DISABLED)
            self.reset_centroids_button.config(state=tk.DISABLED)
            self.calibrate_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.dot_spacing_entry.config(state=tk.NORMAL)
            self.set_spacing_button.config(state=tk.NORMAL)
            self._update_status("Image loaded. Confirm Dot Spacing, adjust Circle Size (or measure) and click 'Run Enhance'.")
        except Exception as e:
            logger.error(f"Failed to load or process image '{path}': {e}", exc_info=True)
            messagebox.showerror("Error Loading Image", f"Failed to load or process image:\n{e}")
            self._update_status("Error loading image.", level=logging.ERROR)
            self.enhance_button.config(state=tk.DISABLED)

    # --- Wrapper functions to run processing in threads via buttons ---
    def run_enhancement_threaded(self):
        logger.info("Enhance button clicked.")
        self._run_in_thread(self.run_enhancement)

    def run_component_filter_threaded(self):
        logger.info("Filter Components button clicked.")
        self._run_in_thread(self.run_component_filter)

    def run_centroid_filter_threaded(self):
        logger.info("Filter Centroids button clicked.")
        self._run_in_thread(self.run_centroid_filter)

    def run_calibration_threaded(self):
        logger.info("Run Calibration button clicked.")
        self._run_in_thread(self._execute_calibration)
    # ----------------------------------------------------------------

    def run_enhancement(self):
        """Actual enhancement logic (runs in background thread)."""
        if self.original_image is None:
            logger.warning("Enhancement run attempted without loaded image.")
            self.root.after(0, lambda: messagebox.showwarning("Warning", "Please load an image first."))
            self._update_status("Load image before enhancing.", level=logging.WARNING)
            return

        circle_size = self._get_odd_circle_size(self.circle_size_var.get())
        self._update_status(f"Running Enhancement (Circle Size = {circle_size}px)...")
        try:
            logger.info(f"Calling calibration.enhance_frame with circle_size={circle_size}")
            self.correlation_map = calibration.enhance_frame(self.original_image, circle_size)
            logger.info("calibration.enhance_frame completed.")

            display_corr_norm = cv2.normalize(self.correlation_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if ENHANCED_COLOR != "gray":
                 display_corr_color = cv2.applyColorMap(display_corr_norm, getattr(cv2, f'COLORMAP_{ENHANCED_COLOR.upper()}', cv2.COLORMAP_JET))
                 display_corr_rgb = cv2.cvtColor(display_corr_color, cv2.COLOR_BGR2RGB)
                 self.display_image_enhanced_pil = Image.fromarray(display_corr_rgb)
            else:
                 self.display_image_enhanced_pil = Image.fromarray(display_corr_norm).convert("RGB")
            logger.debug("Enhanced image prepared for display.")

            def update_gui_enhancement():
                logger.debug("Updating GUI after enhancement.")
                self._display_on_canvas(self.canvas_enhanced, self.display_image_enhanced_pil, ref_name="tk_image_enhanced")
                self.comp_filter_button.config(state=tk.NORMAL)
                self.components_centroids = None
                self.original_final_centroids = None
                self.final_centroids = None
                self.calibration_params = {}
                self.display_image_components_pil = None
                self.display_image_final_pil = None
                self._display_on_canvas(self.canvas_components, None, ref_name="tk_image_components")
                self._display_on_canvas(self.canvas_final, None, ref_name="tk_image_final")
                # Clear plot too
                self._clear_reprojection_plot()

                self.cent_filter_button.config(state=tk.DISABLED)
                self.reset_centroids_button.config(state=tk.DISABLED)
                self.calibrate_button.config(state=tk.DISABLED)
                self.save_button.config(state=tk.DISABLED)
                self._update_status("Enhancement complete. Adjust Component Filters and click 'Run Filter Comp'.")

            self.root.after(0, update_gui_enhancement)

        except Exception as e:
             logger.error(f"Failed during frame enhancement: {e}", exc_info=True)
             self.root.after(0, lambda: messagebox.showerror("Enhancement Error", f"Failed during frame enhancement:\n{e}"))
             self._update_status("Enhancement failed.", level=logging.ERROR)
             self.root.after(0, lambda: self.comp_filter_button.config(state=tk.DISABLED))


    def run_component_filter(self):
        """Actual component filtering logic (runs in background thread)."""
        if self.correlation_map is None:
            logger.warning("Component filter run attempted without correlation map.")
            self.root.after(0, lambda: messagebox.showwarning("Warning", "Please run enhancement first (Step 1)."))
            self._update_status("Run enhancement before filtering components.", level=logging.WARNING)
            return

        correlation_threshold = self.corr_thresh_var.get()
        component_std_threshold = self.comp_std_thresh_var.get()
        self._update_status(f"Running Component Filter (Corr Thr={correlation_threshold:.2f}, Std Thr={component_std_threshold:.1f})...")
        try:
            logger.info(f"Thresholding correlation map at {correlation_threshold:.2f}")
            self.correlation_bool = self.correlation_map > correlation_threshold
            logger.info(f"Calling calibration.filter_components with std_dev threshold={component_std_threshold:.1f}")
            self.components_centroids = calibration.filter_components(self.correlation_bool, component_std_threshold)
            num_components = len(self.components_centroids) if self.components_centroids is not None else 0
            logger.info(f"calibration.filter_components completed. Found {num_components} components.")

            def update_gui_components_result():
                if self.components_centroids is None or len(self.components_centroids) == 0:
                    logger.warning("No components found with the current thresholds.")
                    messagebox.showwarning("Filtering Warning", "No components found with the current thresholds. Try adjusting thresholds and run again.")
                    self._display_on_canvas(self.canvas_components, None, ref_name="tk_image_components")
                    self.cent_filter_button.config(state=tk.DISABLED)
                    self.reset_centroids_button.config(state=tk.DISABLED)
                    self._update_status("Component Filter: No components found.", level=logging.WARNING)
                else:
                    logger.debug("Updating GUI after component filtering.")
                    self._display_on_canvas(self.canvas_components, self.display_image_orig_pil, ref_name="tk_image_components")
                    self.cent_filter_button.config(state=tk.NORMAL)
                    self._update_status(f"Component Filter: {num_components} components found. Adjust Centroid Filter and click 'Run Filter Cent'.")

                self.original_final_centroids = None
                self.final_centroids = None
                self.calibration_params = {}
                self.display_image_final_pil = None
                self._display_on_canvas(self.canvas_final, None, ref_name="tk_image_final")
                # Clear plot too
                self._clear_reprojection_plot()

                self.calibrate_button.config(state=tk.DISABLED)
                self.reset_centroids_button.config(state=tk.DISABLED)
                self.save_button.config(state=tk.DISABLED)

            self.root.after(0, update_gui_components_result)

        except Exception as e:
            logger.error(f"Failed during component filtering: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Component Filter Error", f"Failed during component filtering:\n{e}"))
            self._update_status("Component filtering failed.", level=logging.ERROR)
            self.root.after(0, lambda: self.cent_filter_button.config(state=tk.DISABLED))


    def run_centroid_filter(self):
        """Actual centroid filtering logic (runs in background thread)."""
        if self.components_centroids is None or len(self.components_centroids) == 0:
            logger.warning("Centroid filter run attempted without component centroids.")
            self.root.after(0, lambda: messagebox.showwarning("Warning", "Please filter components first (Step 2)."))
            self._update_status("Run component filter before filtering centroids.", level=logging.WARNING)
            return

        threshold = self.cent_thresh_var.get()
        self._update_status(f"Running Centroid Filter (Dist StdDev Thr={threshold:.1f})...")
        try:
            logger.info(f"Calling calibration.filter_centroids_2 with threshold={threshold:.1f}")
            centroids_to_filter = self.components_centroids.clone()
            self.original_final_centroids = calibration.filter_centroids_2(centroids_to_filter, threshold)
            self.final_centroids = self.original_final_centroids.clone() if self.original_final_centroids is not None else None

            num_final = len(self.final_centroids) if self.final_centroids is not None else 0
            logger.info(f"calibration.filter_centroids_2 completed. Found {num_final} final centroids.")

            def update_gui_final_centroids_result():
                if self.final_centroids is None or len(self.final_centroids) == 0:
                    logger.warning("No centroids remaining after filtering.")
                    messagebox.showwarning("Filtering Warning", "No centroids remaining after filtering. Try adjusting threshold and run again.")
                    self._display_on_canvas(self.canvas_final, None, ref_name="tk_image_final")
                    self.calibrate_button.config(state=tk.DISABLED)
                    self.reset_centroids_button.config(state=tk.DISABLED)
                    self._update_status("Centroid Filter: No final centroids.", level=logging.WARNING)
                else:
                    logger.debug("Updating GUI after centroid filtering.")
                    self._display_on_canvas(self.canvas_final, self.display_image_orig_pil, ref_name="tk_image_final")
                    self.calibrate_button.config(state=tk.NORMAL)
                    self.reset_centroids_button.config(state=tk.NORMAL)
                    self._update_status(f"Centroid Filter: {num_final} final centroids. Ready to Calibrate or manually remove points.")

                # Clear plot too
                self._clear_reprojection_plot()

                self.save_button.config(state=tk.DISABLED)
                self.calibration_params = {}

            self.root.after(0, update_gui_final_centroids_result)

        except Exception as e:
            logger.error(f"Failed during centroid filtering: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Centroid Filter Error", f"Failed during centroid filtering:\n{e}"))
            self._update_status("Centroid filtering failed.", level=logging.ERROR)
            self.root.after(0, lambda: self.calibrate_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.reset_centroids_button.config(state=tk.DISABLED))

    # --- NEW: Point Removal and Reset ---
    def _remove_final_centroid(self, event):
        """Handles clicks on the final_centroids canvas to remove points."""
        if self.final_centroids is None or len(self.final_centroids) == 0:
             logger.debug("Click on final canvas ignored: No centroids loaded.")
             return

        if self.display_image_orig_pil is None:
             logger.warning("Click on final canvas ignored: Original image not loaded.")
             return

        canvas_x, canvas_y = event.x, event.y
        img_x, img_y = self._canvas_to_image_coords(canvas_x, canvas_y)

        if img_x is None or img_y is None:
            logger.warning("Could not convert click coordinates to image coordinates.")
            return

        logger.debug(f"Click on final canvas at ({canvas_x}, {canvas_y}) -> Img ({img_x:.1f}, {img_y:.1f})")

        points = self.final_centroids.cpu().numpy()
        click_point = np.array([img_x, img_y])
        distances_sq = np.sum((points - click_point)**2, axis=1)

        if len(distances_sq) == 0:
             logger.warning("No points found for distance calculation in click handler.")
             return

        min_dist_sq = np.min(distances_sq)
        closest_idx = np.argmin(distances_sq)

        if min_dist_sq < CLICK_THRESHOLD_SQ:
            removed_point = points[closest_idx]
            logger.info(f"Removing final centroid near ({removed_point[0]:.1f}, {removed_point[1]:.1f}) due to click.")

            keep_mask = torch.ones(len(self.final_centroids), dtype=torch.bool, device=self.final_centroids.device)
            keep_mask[closest_idx] = False
            self.final_centroids = self.final_centroids[keep_mask]

            num_remaining = len(self.final_centroids)
            self._display_on_canvas(self.canvas_final, self.display_image_orig_pil, ref_name="tk_image_final")
            self._update_status(f"Removed point near ({removed_point[0]:.0f}, {removed_point[1]:.0f}). {num_remaining} centroids remaining.")

            # Clear plot when points are removed
            self._clear_reprojection_plot()

            if num_remaining > 0:
                 self.calibrate_button.config(state=tk.NORMAL)
            else:
                 self.calibrate_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.calibration_params = {}

        else:
            logger.debug(f"Click was not close enough to any centroid (min dist sq: {min_dist_sq:.1f}, threshold sq: {CLICK_THRESHOLD_SQ})")

    def _reset_final_centroids(self):
        """Resets the final_centroids back to the originally filtered state."""
        if self.original_final_centroids is None:
            logger.warning("Reset called but no original final centroids stored.")
            messagebox.showwarning("Reset Warning", "Cannot reset points. Please run 'Filter Centroids' first.")
            return

        if self.final_centroids is not None and torch.equal(self.final_centroids, self.original_final_centroids):
             logger.info("Reset called, but points are already in original state.")
             self._update_status("Points already in original state.")
             return

        logger.info("Resetting final centroids to original filtered state.")
        self.final_centroids = self.original_final_centroids.clone()

        num_final = len(self.final_centroids) if self.final_centroids is not None else 0
        self._display_on_canvas(self.canvas_final, self.display_image_orig_pil, ref_name="tk_image_final")

        # Clear plot on reset
        self._clear_reprojection_plot()

        if num_final > 0:
            self.calibrate_button.config(state=tk.NORMAL)
            self._update_status(f"Reset points. {num_final} centroids restored. Ready to Calibrate.")
        else:
            self.calibrate_button.config(state=tk.DISABLED)
            self._update_status(f"Reset points. No centroids found in original filter.")

        self.save_button.config(state=tk.DISABLED)
        self.calibration_params = {}


    def _execute_calibration(self):
        """The actual calibration logic, run in a background thread."""
        if self.final_centroids is None or len(self.final_centroids) == 0:
             logger.warning("Calibration attempted without final centroids.")
             self.root.after(0, lambda: messagebox.showwarning("Warning", "No final centroids available. Run previous steps or reset/remove points first."))
             self._update_status("Run steps or adjust points to get final centroids before calibrating.", level=logging.WARNING)
             return
        if self.image_size is None:
             logger.warning("Calibration attempted without image size.")
             self.root.after(0, lambda: messagebox.showwarning("Warning", "Image size not determined. Load image again."))
             self._update_status("Image size missing. Reload image.", level=logging.WARNING)
             return

        current_centroids = self.final_centroids.clone()
        num_centroids = len(current_centroids)
        self._update_status(f"Running Calibration with {num_centroids} centroids...")
        try:
            logger.info(f"Calling calibration.get_world_points with dot spacing {self.dot_spacing}")
            pixel_pts, world_pts, rot_angle_rad, _ = calibration.get_world_points(current_centroids, self.dot_spacing)
            rot_angle_deg = rot_angle_rad * 180 / np.pi
            num_pixel_pts = len(pixel_pts) if pixel_pts is not None else 0
            num_world_pts = len(world_pts) if world_pts is not None else 0
            logger.info(f"calibration.get_world_points completed. Found {num_pixel_pts} pixel points and {num_world_pts} world points. Rotation Angle: {rot_angle_deg:.2f} deg")

            if num_pixel_pts != num_world_pts:
                 err_msg = f"Mismatch between pixel points ({num_pixel_pts}) and world points ({num_world_pts})"
                 logger.error(err_msg)
                 raise ValueError(err_msg)
            if num_pixel_pts < 4:
                 err_msg = f"Need at least 4 points for calibration, found {num_pixel_pts}. Check filtering steps or remove fewer points."
                 logger.error(err_msg)
                 raise ValueError(err_msg)

            logger.info(f"Calling calibration.calibrate_pinhole_with_distortion with {num_pixel_pts} points and image size {self.image_size}")
            K, dist, R, T = calibration.calibrate_pinhole_with_distortion(
                pixel_pts, world_pts, self.image_size
            )
            logger.info("calibration.calibrate_pinhole_with_distortion completed successfully.")

            logger.info("Calling calibration.pixel_to_world_via_plane for verification plot.")
            pixel_pts_np = pixel_pts.cpu().numpy() if isinstance(pixel_pts, torch.Tensor) else np.asarray(pixel_pts)
            reprojected_world_pts = calibration.pixel_to_world_via_plane(pixel_pts_np, K, dist, R, T)
            logger.info("Reprojection calculation complete.")

            self.calibration_params = {'K': K, 'dist': dist, 'R': R, 'T': T, 'rot_angle': rot_angle_rad}
            logger.debug("Stored calibration parameters: K, dist, R, T, rot_angle")

            # Pass reprojected points to GUI update function
            self.root.after(0, self.update_gui_calibration_success, rot_angle_deg, num_pixel_pts, reprojected_world_pts)

        except ValueError as e:
             logger.error(f"Calibration failed: {e}", exc_info=False)
             self.root.after(0, lambda: messagebox.showerror("Calibration Error", f"Calibration failed:\n{e}"))
             self._update_status(f"Calibration failed: {e}", level=logging.ERROR)
             self.root.after(0, lambda: self.save_button.config(state=tk.DISABLED))
             # Clear plot on failure
             self.root.after(0, self._clear_reprojection_plot)
        except RuntimeError as e:
             logger.error(f"OpenCV calibration failed: {e}", exc_info=True)
             self.root.after(0, lambda: messagebox.showerror("Calibration Error", f"OpenCV calibration failed:\n{e}"))
             self._update_status("Calibration failed (OpenCV error).", level=logging.ERROR)
             self.root.after(0, lambda: self.save_button.config(state=tk.DISABLED))
             # Clear plot on failure
             self.root.after(0, self._clear_reprojection_plot)
        except AttributeError as e:
            logger.error(f"Calibration step failed: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Calibration Error", f"Calibration step failed:\n{e}\n\nMake sure 'pixel_to_world_via_plane' exists in calibration.py"))
            self._update_status("Calibration failed (AttributeError).", level=logging.ERROR)
            self.root.after(0, lambda: self.save_button.config(state=tk.DISABLED))
            # Clear plot on failure
            self.root.after(0, self._clear_reprojection_plot)
        except Exception as e:
            logger.error(f"An error occurred during calibration: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Calibration Error", f"An unexpected error occurred during calibration:\n{e}"))
            self._update_status("Calibration failed (unexpected error).", level=logging.ERROR)
            self.root.after(0, lambda: self.save_button.config(state=tk.DISABLED))
            # Clear plot on failure
            self.root.after(0, self._clear_reprojection_plot)

    def update_gui_calibration_success(self, rot_angle_deg, num_pixel_pts, reprojected_world_pts):
        """Updates the GUI after successful calibration and shows the plot."""
        logger.debug("Updating GUI after successful calibration.")
        self.save_button.config(state=tk.NORMAL)
        status_msg = f"Calibration successful! Angle={rot_angle_deg:.2f} deg. Parameters ready to save."
        self._update_status(status_msg)
        logger.info("--- Calibration Results ---")
        logger.info(f"K (Intrinsics):\n{self.calibration_params.get('K')}")
        logger.info(f"Distortion Coefficients:\n{self.calibration_params.get('dist')}")
        logger.info(f"R (Rotation Matrix):\n{self.calibration_params.get('R')}")
        logger.info(f"T (Translation Vector):\n{self.calibration_params.get('T')}")
        logger.info(f"Rotation Angle (rad): {self.calibration_params.get('rot_angle'):.4f}")
        logger.info(f"Rotation Angle (deg): {rot_angle_deg:.2f}")
        logger.info("--------------------------")

        # Show the plot within the GUI
        self._show_reprojection_plot(reprojected_world_pts)

        # Show success message *after* plot is displayed
        messagebox.showinfo("Calibration Success", f"Calibration completed successfully using {num_pixel_pts} points.\nRotation Angle: {rot_angle_deg:.2f} deg\nParameters are ready to save.\nPlot updated.\n(Check console/log for details)")


    def _show_reprojection_plot(self, world_points):
        """Displays the reprojected world points using Matplotlib ON THE EMBEDDED CANVAS."""
        if world_points is None or len(world_points) == 0:
            logger.warning("No reprojected world points to plot.")
            # Clear the plot explicitly if no points
            self._clear_reprojection_plot()
            messagebox.showwarning("Plot Warning", "Could not generate reprojection plot: No points available.")
            return

        try:
            logger.info(f"Generating reprojection plot with {len(world_points)} points in GUI.")
            # Clear the previous plot content
            self.ax.clear()

            # Plot on the existing axes (self.ax)
            self.ax.scatter(world_points[:, 0], world_points[:, 1], c='r', marker='.')
            self.ax.set_title('Reprojected World Points (via Plane Intersection)')
            self.ax.set_xlabel('World X (mm)')
            self.ax.set_ylabel('World Y (mm)')
            self.ax.grid(True)
            self.ax.axis('equal') # Ensure aspect ratio is maintained

            # Redraw the canvas to show the new plot
            self.plot_canvas.draw()
            logger.info("Reprojection plot updated in GUI.")
        except Exception as e:
            logger.error(f"Failed to generate Matplotlib plot in GUI: {e}", exc_info=True)
            messagebox.showerror("Plot Error", f"Failed to display the reprojection plot in the GUI:\n{e}")
            # Attempt to clear plot on error
            self._clear_reprojection_plot()

    def _clear_reprojection_plot(self):
        """Clears the embedded Matplotlib plot."""
        try:
            self.ax.clear()
            # Re-apply labels/title/grid after clearing if desired
            self.ax.set_title("Reprojection Plot (World Coords)")
            self.ax.set_xlabel("World X (mm)")
            self.ax.set_ylabel("World Y (mm)")
            self.ax.grid(True)
            self.ax.axis('equal')
            # Draw the cleared state
            self.plot_canvas.draw()
            logger.debug("Cleared reprojection plot.")
        except Exception as e:
             logger.error(f"Error clearing plot canvas: {e}", exc_info=True)


    def save_parameters(self):
        logger.info("Save parameters button clicked.")
        if not self.calibration_params:
            logger.warning("Save attempted without calibration parameters.")
            messagebox.showwarning("Warning", "No calibration parameters to save. Run calibration first.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Calibration Parameters",
            defaultextension=".npz",
            filetypes=[("NumPy Archive", "*.npz")],
            initialdir=os.path.dirname(self.image_path) if self.image_path else os.getcwd()
        )
        if not save_path:
            logger.warning("Save parameters dialog cancelled.")
            return

        self._update_status(f"Saving parameters to {os.path.basename(save_path)}...")
        try:
            np.savez(save_path, **self.calibration_params)
            logger.info(f"Parameters saved successfully to {save_path}")
            self._update_status(f"Parameters saved to {os.path.basename(save_path)}.")
            messagebox.showinfo("Save Successful", f"Calibration parameters saved to:\n{save_path}")
        except Exception as e:
            logger.error(f"Failed to save parameters to {save_path}: {e}", exc_info=True)
            messagebox.showerror("Save Error", f"Failed to save parameters:\n{e}")
            self._update_status("Failed to save parameters.", level=logging.ERROR)


# --- Run the Application ---
if __name__ == "__main__":
    logger.info("Starting Calibration Application.")
    # Check for matplotlib availability happens implicitly with imports now
    # Check if backend setting worked (optional)
    try:
         logger.info(f"Matplotlib version {matplotlib.__version__} found. Using {matplotlib.get_backend()} backend.")
    except Exception as e:
         logger.warning(f"Could not verify Matplotlib backend: {e}", exc_info=True)


    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()
    logger.info("Calibration Application finished.")
