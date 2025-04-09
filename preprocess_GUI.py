import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import moviepy.editor as mp
import threading
import os
import time
import cv2 # OpenCV for frame reading
from PIL import Image, ImageTk # Pillow for Tkinter image compatibility
import numpy as np

# Constants for canvas interaction
CANVAS_WIDTH = 480
CANVAS_HEIGHT = 360
HANDLE_SIZE = 4 # Half-size of the drag handles


class VideoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Video Editor")
        # Increased size to accommodate visuals
        self.root.geometry("600x750")

        # --- Video & Processing State ---
        self.input_filepath = None
        self.output_filepath = None
        self.cap = None # OpenCV VideoCapture object
        self.original_duration = 0
        self.original_width = 0
        self.original_height = 0
        self.fps = 0
        self.total_frames = 0
        self.current_frame_display_time = 0.0 # Time of frame shown in canvas
        self.processing_thread = None

        # --- Image Display & Scaling ---
        self.photo_image = None # Reference to prevent garbage collection
        self.display_scale_factor = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0

        # --- Cropping State ---
        self.crop_rect_id = None # Canvas ID for crop rectangle
        self.crop_handle_ids = {} # Canvas IDs for handles { 'nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w'}
        self.dragging_handle = None # Which handle is being dragged
        self.drag_start_pos = None

        # --- GUI Setup ---
        # Frame for controls (top part)
        self.controls_frame = ttk.Frame(root)
        self.controls_frame.pack(pady=5, padx=10, fill="x")

        # Frame for Visuals (bottom part)
        self.visuals_frame = ttk.Frame(root)
        self.visuals_frame.pack(pady=5, padx=10, fill="both", expand=True)

        # --- Top: Input File Selection ---
        self.frame_input = ttk.LabelFrame(self.controls_frame, text="1. Load Video")
        self.frame_input.pack(pady=5, fill="x")
        self.btn_load = ttk.Button(self.frame_input, text="Load Video", command=self.load_video)
        self.btn_load.pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_filename = ttk.Label(self.frame_input, text="No file selected", wraplength=450)
        self.lbl_filename.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Top: Crop Parameters ---
        self.frame_params = ttk.LabelFrame(self.controls_frame, text="2. Crop Parameters")
        self.frame_params.pack(pady=5, fill="x")

        # Temporal (Duration) Entries
        param_time_frame = ttk.Frame(self.frame_params)
        param_time_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(param_time_frame, text="Start (s):").pack(side=tk.LEFT, padx=(0,2))
        self.entry_start = ttk.Entry(param_time_frame, width=8)
        self.entry_start.pack(side=tk.LEFT, padx=(0, 10))
        self.entry_start.insert(0, "0")
        ttk.Label(param_time_frame, text="End (s):").pack(side=tk.LEFT, padx=(0,2))
        self.entry_end = ttk.Entry(param_time_frame, width=8)
        self.entry_end.pack(side=tk.LEFT, padx=(0, 5))
        self.lbl_duration_info = ttk.Label(param_time_frame, text="Duration: -")
        self.lbl_duration_info.pack(side=tk.LEFT, padx=5)
        # Add Trace to update visuals when entries change focus
        self.entry_start.bind("<FocusOut>", self.update_visuals_from_entries)
        self.entry_end.bind("<FocusOut>", self.update_visuals_from_entries)


        # Spatial (Frame) Entries
        param_space_frame = ttk.Frame(self.frame_params)
        param_space_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(param_space_frame, text="X1:").pack(side=tk.LEFT, padx=(0,2))
        self.entry_x1 = ttk.Entry(param_space_frame, width=6)
        self.entry_x1.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(param_space_frame, text="Y1:").pack(side=tk.LEFT, padx=(0,2))
        self.entry_y1 = ttk.Entry(param_space_frame, width=6)
        self.entry_y1.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(param_space_frame, text="X2:").pack(side=tk.LEFT, padx=(0,2))
        self.entry_x2 = ttk.Entry(param_space_frame, width=6)
        self.entry_x2.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(param_space_frame, text="Y2:").pack(side=tk.LEFT, padx=(0,2))
        self.entry_y2 = ttk.Entry(param_space_frame, width=6)
        self.entry_y2.pack(side=tk.LEFT, padx=(0, 5))
        self.lbl_dimension_info = ttk.Label(param_space_frame, text="Dims: -")
        self.lbl_dimension_info.pack(side=tk.LEFT, padx=5)
        # Add Trace
        self.entry_x1.bind("<FocusOut>", self.update_visuals_from_entries)
        self.entry_y1.bind("<FocusOut>", self.update_visuals_from_entries)
        self.entry_x2.bind("<FocusOut>", self.update_visuals_from_entries)
        self.entry_y2.bind("<FocusOut>", self.update_visuals_from_entries)

        # --- Visuals: Frame Display Canvas ---
        self.frame_canvas = ttk.LabelFrame(self.visuals_frame, text="3. Frame Preview & Spatial Crop")
        self.frame_canvas.pack(pady=5, fill="both", expand=True)
        self.canvas = tk.Canvas(self.frame_canvas, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="grey", relief=tk.SUNKEN, borderwidth=1)
        self.canvas.pack(pady=5, padx=5, fill="both", expand=True)
        # Bind mouse events for dragging crop box
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # --- Visuals: Timeline ---
        self.frame_timeline = ttk.LabelFrame(self.visuals_frame, text="4. Timeline & Temporal Crop")
        self.frame_timeline.pack(pady=5, fill="x")
        self.time_slider_var = tk.DoubleVar()
        self.time_slider = ttk.Scale(self.frame_timeline, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.time_slider_var, command=self.on_slider_move)
        self.time_slider.pack(fill="x", padx=5, pady=2)
        self.time_slider.bind("<ButtonRelease-1>", self.on_slider_release) # Update frame on release
        self.time_slider.config(state=tk.DISABLED)

        timeline_btns_frame = ttk.Frame(self.frame_timeline)
        timeline_btns_frame.pack(fill="x", padx=5)
        self.btn_set_start = ttk.Button(timeline_btns_frame, text="Set Start", width=8, command=self.set_start_time_from_slider, state=tk.DISABLED)
        self.btn_set_start.pack(side=tk.LEFT, padx=2)
        self.btn_set_end = ttk.Button(timeline_btns_frame, text="Set End", width=8, command=self.set_end_time_from_slider, state=tk.DISABLED)
        self.btn_set_end.pack(side=tk.LEFT, padx=2)
        self.lbl_current_time = ttk.Label(timeline_btns_frame, text="Time: 0.00s")
        self.lbl_current_time.pack(side=tk.LEFT, padx=10)

        # --- Bottom: Processing & Status ---
        self.frame_process = ttk.LabelFrame(self.controls_frame, text="5. Process and Save") # Moved back up for space
        self.frame_process.pack(pady=5, fill="x")
        self.btn_process = ttk.Button(self.frame_process, text="Process & Save Video", command=self.start_processing_thread, state=tk.DISABLED)
        self.btn_process.pack(pady=5, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Load a video.")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=100, mode="determinate")
        # Progress bar packed/unpacked as needed

    # --- Coordinate Conversion ---
    def video_to_canvas(self, vx, vy):
        """Converts original video coordinates to canvas coordinates."""
        cx = int(vx * self.display_scale_factor + self.display_offset_x)
        cy = int(vy * self.display_scale_factor + self.display_offset_y)
        return cx, cy

    def canvas_to_video(self, cx, cy):
        """Converts canvas coordinates to original video coordinates."""
        vx = (cx - self.display_offset_x) / self.display_scale_factor
        vy = (cy - self.display_offset_y) / self.display_scale_factor
        # Clamp to video dimensions
        vx = max(0, min(self.original_width, vx))
        vy = max(0, min(self.original_height, vy))
        return int(round(vx)), int(round(vy))

    # --- Video Loading and Frame Display ---
    def load_video(self):
        filepath = filedialog.askopenfilename(title="Select Video File", filetypes=(("Video Files", "*.mp4 *.avi *.mov"), ("All files", "*.*")))
        if not filepath: return
        self.input_filepath = filepath
        if self.cap: self.cap.release() # Release previous capture

        try:
            self.cap = cv2.VideoCapture(self.input_filepath)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video file: {self.input_filepath}")

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.original_duration = self.total_frames / self.fps if self.fps > 0 else 0

            if self.original_duration <= 0 or self.original_width <= 0 or self.original_height <= 0:
                 raise ValueError("Video properties (duration/dimensions) invalid.")

            self.lbl_filename.config(text=os.path.basename(filepath))
            self.lbl_duration_info.config(text=f"Duration: {self.original_duration:.2f}s")
            self.lbl_dimension_info.config(text=f"Dims: {self.original_width}x{self.original_height}")

            # Reset entries and slider
            self.entry_start.delete(0, tk.END); self.entry_start.insert(0, "0")
            self.entry_end.delete(0, tk.END); self.entry_end.insert(0, f"{self.original_duration:.2f}")
            self.entry_x1.delete(0, tk.END); self.entry_x1.insert(0, "0")
            self.entry_y1.delete(0, tk.END); self.entry_y1.insert(0, "0")
            self.entry_x2.delete(0, tk.END); self.entry_x2.insert(0, f"{self.original_width}")
            self.entry_y2.delete(0, tk.END); self.entry_y2.insert(0, f"{self.original_height}")

            self.time_slider.config(to=self.original_duration, state=tk.NORMAL)
            self.time_slider_var.set(0)

            # Show first frame
            self.show_frame_at_time(0)
            self.update_crop_box_on_canvas() # Draw initial box

            self.btn_process.config(state=tk.NORMAL)
            self.btn_set_start.config(state=tk.NORMAL)
            self.btn_set_end.config(state=tk.NORMAL)
            self.status_var.set(f"Loaded: {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Error Loading Video", f"Failed to load video.\nCheck file path and format.\nError: {e}")
            self.reset_ui() # Reset UI state on failure

    def reset_ui(self):
        if self.cap: self.cap.release(); self.cap = None
        self.input_filepath = None
        self.original_duration = 0
        self.original_width = 0
        self.original_height = 0
        self.lbl_filename.config(text="No file selected")
        self.lbl_duration_info.config(text="Duration: -")
        self.lbl_dimension_info.config(text="Dims: -")
        self.entry_start.delete(0, tk.END); self.entry_start.insert(0, "0")
        self.entry_end.delete(0, tk.END); self.entry_end.insert(0, "")
        self.entry_x1.delete(0, tk.END); self.entry_x1.insert(0, "")
        self.entry_y1.delete(0, tk.END); self.entry_y1.insert(0, "")
        self.entry_x2.delete(0, tk.END); self.entry_x2.insert(0, "")
        self.entry_y2.delete(0, tk.END); self.entry_y2.insert(0, "")
        self.time_slider.config(state=tk.DISABLED); self.time_slider_var.set(0)
        self.btn_process.config(state=tk.DISABLED)
        self.btn_set_start.config(state=tk.DISABLED)
        self.btn_set_end.config(state=tk.DISABLED)
        self.canvas.delete("all") # Clear canvas
        self.photo_image = None
        self.status_var.set("Ready")


    def show_frame_at_time(self, time_sec):
        """Reads and displays a frame at the given time on the canvas."""
        if not self.cap or not self.cap.isOpened() or self.original_width <= 0:
            return

        self.current_frame_display_time = time_sec
        self.lbl_current_time.config(text=f"Time: {time_sec:.2f}s")

        # Seek using milliseconds
        self.cap.set(cv2.CAP_PROP_POS_MSEC, int(time_sec * 1000))
        ret, frame = self.cap.read()

        if ret:
            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # --- Calculate scaling to fit canvas ---
            img_w, img_h = pil_image.size
            scale_w = CANVAS_WIDTH / img_w
            scale_h = CANVAS_HEIGHT / img_h
            self.display_scale_factor = min(scale_w, scale_h) # Maintain aspect ratio

            new_w = int(img_w * self.display_scale_factor)
            new_h = int(img_h * self.display_scale_factor)

            # Center the image on the canvas
            self.display_offset_x = (CANVAS_WIDTH - new_w) // 2
            self.display_offset_y = (CANVAS_HEIGHT - new_h) // 2

            resized_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Convert to Tkinter PhotoImage
            self.photo_image = ImageTk.PhotoImage(resized_image)

            # Display on canvas
            self.canvas.delete("all") # Clear previous frame/drawings
            self.canvas.create_image(self.display_offset_x, self.display_offset_y, anchor=tk.NW, image=self.photo_image)
            # Keep reference! self.canvas.image = self.photo_image # Alternative way

            self.update_crop_box_on_canvas() # Redraw crop box on new frame

        else:
            # Could not read frame (e.g., end of video)
            # Keep last frame displayed or clear canvas? Let's clear.
            # self.canvas.delete("all")
            # self.canvas.create_text(CANVAS_WIDTH/2, CANVAS_HEIGHT/2, text="Cannot read frame", fill="white")
            pass # Keep last frame for now

    # --- Timeline Interaction ---
    def on_slider_move(self, value_str):
        """Called continuously while slider is moving."""
        if not self.cap: return
        time_sec = float(value_str)
        self.lbl_current_time.config(text=f"Time: {time_sec:.2f}s")
        # Optional: Show frame preview while dragging (can be slow!)
        # self.show_frame_at_time(time_sec)

    def on_slider_release(self, event):
        """Called when slider mouse button is released."""
        if not self.cap: return
        time_sec = self.time_slider_var.get()
        self.show_frame_at_time(time_sec)

    def set_start_time_from_slider(self):
        if not self.cap: return
        current_time = self.time_slider_var.get()
        self.entry_start.delete(0, tk.END)
        self.entry_start.insert(0, f"{current_time:.2f}")
        # Optionally add visual indicator on timeline later

    def set_end_time_from_slider(self):
        if not self.cap: return
        current_time = self.time_slider_var.get()
        self.entry_end.delete(0, tk.END)
        self.entry_end.insert(0, f"{current_time:.2f}")
        # Optionally add visual indicator on timeline later

    # --- Spatial Crop Box Interaction ---
    def update_crop_box_on_canvas(self):
        """Draws/updates the crop rectangle and handles based on entry values."""
        if not self.cap or self.original_width <= 0: return # No video loaded/valid

        # Delete previous drawings if they exist
        if self.crop_rect_id: self.canvas.delete(self.crop_rect_id)
        for handle_id in self.crop_handle_ids.values(): self.canvas.delete(handle_id)
        self.crop_handle_ids = {}

        try:
            # Get video coordinates from entries (default to full frame if invalid/empty)
            vx1 = int(float(self.entry_x1.get())) if self.entry_x1.get() else 0
            vy1 = int(float(self.entry_y1.get())) if self.entry_y1.get() else 0
            vx2 = int(float(self.entry_x2.get())) if self.entry_x2.get() else self.original_width
            vy2 = int(float(self.entry_y2.get())) if self.entry_y2.get() else self.original_height

            # Basic validation/clamping
            vx1 = max(0, min(self.original_width, vx1))
            vy1 = max(0, min(self.original_height, vy1))
            vx2 = max(vx1, min(self.original_width, vx2)) # ensure x2 > x1
            vy2 = max(vy1, min(self.original_height, vy2)) # ensure y2 > y1

        except ValueError: # Handle case where entry has non-numeric text
             vx1, vy1 = 0, 0
             vx2, vy2 = self.original_width, self.original_height

        # Convert video coordinates to canvas coordinates
        cx1, cy1 = self.video_to_canvas(vx1, vy1)
        cx2, cy2 = self.video_to_canvas(vx2, vy2)

        # Draw main rectangle (semi-transparent outline)
        self.crop_rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="red", width=2, tags="crop_box")

        # Draw handles at corners and midpoints
        handles = {
            'nw': (cx1, cy1), 'ne': (cx2, cy1), 'sw': (cx1, cy2), 'se': (cx2, cy2),
            'n': ((cx1+cx2)//2, cy1), 's': ((cx1+cx2)//2, cy2),
            'w': (cx1, (cy1+cy2)//2), 'e': (cx2, (cy1+cy2)//2)
        }
        for key, (hx, hy) in handles.items():
            handle_id = self.canvas.create_rectangle(hx - HANDLE_SIZE, hy - HANDLE_SIZE,
                                                      hx + HANDLE_SIZE, hy + HANDLE_SIZE,
                                                      fill="white", outline="black", tags=("crop_handle", key)) # Tag with key
            self.crop_handle_ids[key] = handle_id


    def get_handle_at_pos(self, x, y):
        """Finds which handle (if any) is at the given canvas coordinates."""
        # Check corner/edge handles first (smaller target)
        for key, handle_id in self.crop_handle_ids.items():
            coords = self.canvas.coords(handle_id)
            if coords and coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
                 return key # Return 'nw', 'ne', 'n', etc.

        # Check if inside the main rectangle (for moving) - optional
        # rect_coords = self.canvas.coords(self.crop_rect_id)
        # if rect_coords and rect_coords[0] <= x <= rect_coords[2] and rect_coords[1] <= y <= rect_coords[3]:
        #     return 'move' # Special key for moving the whole box

        return None

    def on_canvas_press(self, event):
        """Handles mouse button press on the canvas."""
        if not self.cap: return
        self.dragging_handle = self.get_handle_at_pos(event.x, event.y)
        if self.dragging_handle:
            self.drag_start_pos = (event.x, event.y)
            # Optional: Change cursor
            # self.canvas.config(cursor="hand2")
        else:
            self.drag_start_pos = None

    def on_canvas_drag(self, event):
        """Handles mouse drag on the canvas."""
        if not self.dragging_handle or not self.drag_start_pos:
            return

        dx = event.x - self.drag_start_pos[0]
        dy = event.y - self.drag_start_pos[1]

        # Get current video coordinates from entries
        try:
            vx1 = int(float(self.entry_x1.get()))
            vy1 = int(float(self.entry_y1.get()))
            vx2 = int(float(self.entry_x2.get()))
            vy2 = int(float(self.entry_y2.get()))
        except ValueError: return # Should not happen if updated correctly

        # Convert mouse delta (canvas space) to video space delta (approximate)
        dvx = dx / self.display_scale_factor
        dvy = dy / self.display_scale_factor

        new_vx1, new_vy1, new_vx2, new_vy2 = vx1, vy1, vx2, vy2

        # Update coordinates based on dragged handle
        handle = self.dragging_handle
        if 'n' in handle: new_vy1 += dvy
        if 's' in handle: new_vy2 += dvy
        if 'w' in handle: new_vx1 += dvx
        if 'e' in handle: new_vx2 += dvx
        # Add logic for 'move' if implemented

        # --- Validation and Clamping in VIDEO coordinates ---
        new_vx1 = max(0, min(self.original_width - 1, new_vx1)) # Ensure min width/height of 1px
        new_vy1 = max(0, min(self.original_height - 1, new_vy1))
        new_vx2 = max(new_vx1 + 1, min(self.original_width, new_vx2))
        new_vy2 = max(new_vy1 + 1, min(self.original_height, new_vy2))

        # Ensure N/S or W/E haven't crossed over during drag
        if 'n' in handle and new_vy1 >= new_vy2: new_vy1 = new_vy2 - 1
        if 's' in handle and new_vy2 <= new_vy1: new_vy2 = new_vy1 + 1
        if 'w' in handle and new_vx1 >= new_vx2: new_vx1 = new_vx2 - 1
        if 'e' in handle and new_vx2 <= new_vx1: new_vx2 = new_vx1 + 1

        # Update entries *during* drag for immediate feedback (optional, can slow down)
        self.update_entries(int(round(new_vx1)), int(round(new_vy1)), int(round(new_vx2)), int(round(new_vy2)))

        # Redraw the box based on the *updated* entry values
        self.update_crop_box_on_canvas()

        # Update drag start position for next delta calculation
        self.drag_start_pos = (event.x, event.y)


    def on_canvas_release(self, event):
        """Handles mouse button release on the canvas."""
        if not self.dragging_handle: return
        # Final update of entries based on the last drag position
        self.update_entries_from_visuals() # Recalculate from final box coords

        self.dragging_handle = None
        self.drag_start_pos = None
        # Optional: Reset cursor
        # self.canvas.config(cursor="")

    # --- Linking Entries and Visuals ---
    def update_entries(self, vx1, vy1, vx2, vy2):
        """Updates the coordinate entry fields."""
        self.entry_x1.delete(0, tk.END); self.entry_x1.insert(0, str(vx1))
        self.entry_y1.delete(0, tk.END); self.entry_y1.insert(0, str(vy1))
        self.entry_x2.delete(0, tk.END); self.entry_x2.insert(0, str(vx2))
        self.entry_y2.delete(0, tk.END); self.entry_y2.insert(0, str(vy2))

    def update_entries_from_visuals(self):
         """Updates entries based on the current canvas crop box position."""
         if not self.crop_rect_id: return
         coords = self.canvas.coords(self.crop_rect_id)
         if not coords: return
         cx1, cy1, cx2, cy2 = coords
         vx1, vy1 = self.canvas_to_video(cx1, cy1)
         vx2, vy2 = self.canvas_to_video(cx2, cy2)
         self.update_entries(vx1, vy1, vx2, vy2)

    def update_visuals_from_entries(self, event=None):
         """Updates slider and crop box when entries lose focus."""
         if not self.cap: return
         try:
             start_t = float(self.entry_start.get())
             end_t = float(self.entry_end.get())
             # Update slider position only if it's different to avoid loop
             if abs(self.time_slider_var.get() - start_t) > 0.01: # Adjust tolerance as needed
                 self.time_slider_var.set(max(0, min(self.original_duration, start_t)))
                 # Update frame only if slider changed significantly
                 self.show_frame_at_time(self.time_slider_var.get())
             self.update_crop_box_on_canvas()
         except ValueError:
             # Handle invalid entry data - maybe show warning or reset to previous valid
             self.status_var.set("Warning: Invalid number in entry field.")
             # Optionally revert entries or just redraw box with default/last known good values
             self.update_crop_box_on_canvas() # Redraw based on current (possibly invalid) state


    # --- Processing (Adapted from previous version) ---
    def start_processing_thread(self):
        if not self.input_filepath:
            messagebox.showwarning("No File", "Please load a video file first.")
            return
        if self.original_duration <= 0:
             messagebox.showwarning("Invalid Video Info", "Cannot process: Video duration not loaded correctly.")
             return
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        # Get params directly from entries
        try:
            start_time = float(self.entry_start.get())
            end_time = float(self.entry_end.get())
            x1 = int(float(self.entry_x1.get()))
            y1 = int(float(self.entry_y1.get()))
            x2 = int(float(self.entry_x2.get()))
            y2 = int(float(self.entry_y2.get()))

            # Validation (can add more detailed checks here if needed)
            if not (0 <= start_time < end_time <= self.original_duration + 0.01):
                raise ValueError("Invalid start/end time.")
            if not (0 <= x1 < x2 <= self.original_width and 0 <= y1 < y2 <= self.original_height):
                raise ValueError("Invalid crop coordinates.")

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check crop parameters.\n{e}")
            return

        # Ask for Output File
        output_filepath = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")),
            initialfile=f"{os.path.splitext(os.path.basename(self.input_filepath))[0]}_edited.mp4"
        )
        if not output_filepath:
            self.status_var.set("Save cancelled")
            return

        self.output_filepath = output_filepath # Store for thread

        # Disable UI elements
        self.btn_process.config(state=tk.DISABLED)
        self.btn_load.config(state=tk.DISABLED)
        self.time_slider.config(state=tk.DISABLED)
        self.status_var.set("Starting processing...")
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.progress_bar["value"] = 0
        self.root.update_idletasks()

        # Start thread
        self.processing_thread = threading.Thread(
            target=self.process_video_thread,
            args=(start_time, end_time, x1, y1, x2, y2), # Pass validated values
            daemon=True
        )
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar["value"] = value
        self.root.update_idletasks()

    def process_video_thread(self, start_time, end_time, x1, y1, x2, y2):
            """Runs in background thread."""
            clip = None
            edited_clip = None
            try:
                self.root.after(0, lambda: self.status_var.set("Processing: Loading video..."))

                # Perform Editing with MoviePy
                perform_spatial_crop = not (x1 == 0 and y1 == 0 and x2 == self.original_width and y2 == self.original_height)

                clip = mp.VideoFileClip(self.input_filepath)
                self.root.after(0, lambda: self.update_progress(10))

                # 1. Temporal Crop
                actual_duration = clip.duration
                safe_end_time = min(end_time, actual_duration)
                safe_start_time = min(start_time, safe_end_time) # Ensure start is not beyond end

                # --- DEBUG CHECK 1: Validate times ---
                # Use a small epsilon for comparison to avoid float issues
                epsilon = 0.001
                if safe_start_time >= safe_end_time - epsilon:
                    # If times are essentially the same or inverted, try making a tiny clip or raise error
                    if safe_end_time > epsilon:
                        safe_start_time = safe_end_time - epsilon # Create minimal duration clip
                        print(f"Warning: Adjusted start time to {safe_start_time:.3f} due to near-zero duration.")
                    else: # Cannot create clip if end time is near zero
                        raise ValueError(f"Calculated clip duration is zero or negative (Start: {safe_start_time:.3f}, End: {safe_end_time:.3f}). Cannot process.")

                edited_clip = clip.subclip(safe_start_time, safe_end_time)
                self.root.after(0, lambda st=safe_start_time, et=safe_end_time: self.status_var.set(f"Processing: Time cropped ({st:.2f}s - {et:.2f}s)..."))
                self.root.after(0, lambda: self.update_progress(30))

                # 2. Spatial Crop
                if perform_spatial_crop:
                    self.root.after(0, lambda: self.status_var.set("Processing: Spatially cropping..."))
                    edited_clip = edited_clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)
                    self.root.after(0, lambda: self.status_var.set(f"Processing: Spatially cropped (x1={x1}, y1={y1}, x2={x2}, y2={y2})..."))
                    self.root.after(0, lambda: self.update_progress(50))
                else:
                    self.root.after(0, lambda: self.status_var.set("Processing: Skipping full-frame spatial crop..."))
                    self.root.after(0, lambda: self.update_progress(50))

                # --- DEBUG CHECK 2: Check edited clip duration ---
                final_duration = edited_clip.duration
                print(f"DEBUG: Final clip duration before writing: {final_duration:.3f} seconds")
                if final_duration <= 0:
                    raise ValueError("Processed clip has zero or negative duration. Cannot write file.")

                # 3. Write Output File
                self.root.after(0, lambda: self.status_var.set("Processing: Writing output file... (this takes time)"))

                # --- DEBUG STEP 1: Enable FFmpeg console output ---
                print("\n--- Starting MoviePy Write (FFmpeg Output Below) ---") # Marker in console
                edited_clip.write_videofile(
                    self.output_filepath,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    logger="bar" # CHANGE: Enable the progress bar logger in the CONSOLE
                    # logger=None # Original value
                )
                print("--- MoviePy Write Finished ---\n") # Marker in console

                # Final progress update
                self.root.after(0, lambda: self.update_progress(100))
                self.processing_finished(success=True, message=f"Video saved to {os.path.basename(self.output_filepath)}")

            except Exception as e:
                import traceback
                # Print full traceback to console for detailed debugging
                print("\n--- ERROR DURING PROCESSING ---")
                print(traceback.format_exc())
                print("-----------------------------\n")
                error_message = f"Error during processing:\n{e}"
                # Schedule error message display in main thread
                self.root.after(0, lambda msg=error_message: messagebox.showerror("Processing Error", msg))
                self.processing_finished(success=False, message="Processing failed")
            finally:
                # Ensure clips are closed even if error occurs mid-process
                try:
                    if clip: clip.close()
                except Exception: pass
                try:
                    if edited_clip: edited_clip.close()
                except Exception: pass

    def processing_finished(self, success, message):
        """Updates the GUI after processing (runs in main thread)."""
        self.status_var.set(message)
        is_input_valid = bool(self.input_filepath and self.original_duration > 0)
        self.btn_process.config(state=tk.NORMAL if is_input_valid else tk.DISABLED)
        self.btn_load.config(state=tk.NORMAL)
        self.time_slider.config(state=tk.NORMAL if is_input_valid else tk.DISABLED)
        self.progress_bar.pack_forget()
        self.progress_bar["value"] = 0
        if success:
            pass # message in status bar is enough


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEditorApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Exiting application gracefully...")
        root.destroy()
