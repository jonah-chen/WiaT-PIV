# preprocess_GUI_v5.py
# - Explicitly initializes attributes in __init__.
# - Displays loaded calibration image alongside video frame.
# - Automatically crops/resizes loaded calibration image to match video aspect ratio/resolution.
# - Draws a non-interactive mirrored crop rectangle on the calibration image canvas.

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import moviepy.editor as mp
import threading
import os
import time
import cv2 # OpenCV for frame reading and image saving
from PIL import Image, ImageTk # Pillow for Tkinter image compatibility and image loading/saving
import numpy as np
import math # For aspect ratio calculations

# Constants
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 300
HANDLE_SIZE = 4
ASPECT_RATIO_TOLERANCE = 0.01
VIDEO_CROP_COLOR = "red"
CALIB_CROP_COLOR = "cyan" # Different color for mirrored box


class VideoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Video Editor & Calib Frame Cropper (v5)")
        self.root.geometry("1100x800")

        # --- Explicit Initialization ---
        # Video State
        self.input_filepath = None
        self.output_filepath = None
        self.cap = None
        self.video_properties = {} # Initialize as empty dict
        self.original_duration = 0.0
        self.original_width = 0
        self.original_height = 0
        self.fps = 0.0
        self.total_frames = 0
        self.current_frame_display_time = 0.0
        # Calibration Image State
        self.calib_image_path = None
        self.calib_image_pil = None # Processed (aspect/res adjusted) PIL image
        self.output_calib_image_path = None
        # Display State (Tkinter references)
        self.photo_image_video = None
        self.photo_image_calib = None
        # Display State (Scaling/Offset - separate for each canvas)
        self.display_params = {
            'video': {'scale_x': 1.0, 'scale_y': 1.0, 'offset_x': 0, 'offset_y': 0},
            'calib': {'scale_x': 1.0, 'scale_y': 1.0, 'offset_x': 0, 'offset_y': 0}
        }
        # Cropping State
        self.crop_rect_id_video = None # Red box on video canvas
        self.crop_rect_id_calib = None # Blue box on calib canvas
        self.crop_handle_ids = {} # Handles only on video canvas
        self.dragging_handle = None
        self.drag_start_pos = None
        # Processing State
        self.processing_thread = None

        # --- GUI Setup ---
        # Controls Frame (Top)
        self.controls_frame = ttk.Frame(root); self.controls_frame.pack(pady=5, padx=10, fill="x")
        # Visuals Frame (Split Horizontally)
        self.visuals_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL); self.visuals_pane.pack(pady=5, padx=10, fill="both", expand=True)
        # Left Frame (Video)
        self.video_preview_frame = ttk.Frame(self.visuals_pane, padding=5); self.visuals_pane.add(self.video_preview_frame, weight=1)
        self.video_preview_frame.rowconfigure(1, weight=1); self.video_preview_frame.columnconfigure(0, weight=1)
        # Right Frame (Calib)
        self.calib_preview_frame = ttk.Frame(self.visuals_pane, padding=5); self.visuals_pane.add(self.calib_preview_frame, weight=1)
        self.calib_preview_frame.rowconfigure(1, weight=1); self.calib_preview_frame.columnconfigure(0, weight=1)

        # --- Input File Selection ---
        self.frame_input = ttk.LabelFrame(self.controls_frame, text="1. Load Files"); self.frame_input.pack(pady=5, fill="x"); self.frame_input.columnconfigure(1, weight=1)
        self.btn_load_video = ttk.Button(self.frame_input, text="Load Video", command=self.load_video); self.btn_load_video.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.lbl_filename_video = ttk.Label(self.frame_input, text="No video selected", wraplength=400); self.lbl_filename_video.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.btn_load_calib = ttk.Button(self.frame_input, text="Load Calib Image (Optional)", command=self.load_calib_image); self.btn_load_calib.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.lbl_filename_calib = ttk.Label(self.frame_input, text="No calibration image loaded", wraplength=400); self.lbl_filename_calib.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        # --- Crop Parameters ---
        self.frame_params = ttk.LabelFrame(self.controls_frame, text="2. Crop Parameters"); self.frame_params.pack(pady=5, fill="x")
        # Temporal
        param_time_frame = ttk.Frame(self.frame_params); param_time_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(param_time_frame, text="Start (s):").pack(side=tk.LEFT); self.entry_start = ttk.Entry(param_time_frame, width=8); self.entry_start.pack(side=tk.LEFT, padx=(2,10)); self.entry_start.insert(0, "0")
        ttk.Label(param_time_frame, text="End (s):").pack(side=tk.LEFT); self.entry_end = ttk.Entry(param_time_frame, width=8); self.entry_end.pack(side=tk.LEFT, padx=(2,5))
        self.lbl_duration_info = ttk.Label(param_time_frame, text="Duration: -"); self.lbl_duration_info.pack(side=tk.LEFT, padx=5)
        self.entry_start.bind("<FocusOut>", self.update_visuals_from_entries); self.entry_end.bind("<FocusOut>", self.update_visuals_from_entries)
        # Spatial
        param_space_frame = ttk.Frame(self.frame_params); param_space_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(param_space_frame, text="X1:").pack(side=tk.LEFT); self.entry_x1 = ttk.Entry(param_space_frame, width=6); self.entry_x1.pack(side=tk.LEFT, padx=(2,5))
        ttk.Label(param_space_frame, text="Y1:").pack(side=tk.LEFT); self.entry_y1 = ttk.Entry(param_space_frame, width=6); self.entry_y1.pack(side=tk.LEFT, padx=(2,10))
        ttk.Label(param_space_frame, text="X2:").pack(side=tk.LEFT); self.entry_x2 = ttk.Entry(param_space_frame, width=6); self.entry_x2.pack(side=tk.LEFT, padx=(2,5))
        ttk.Label(param_space_frame, text="Y2:").pack(side=tk.LEFT); self.entry_y2 = ttk.Entry(param_space_frame, width=6); self.entry_y2.pack(side=tk.LEFT, padx=(2,5))
        self.lbl_dimension_info = ttk.Label(param_space_frame, text="Dims: -"); self.lbl_dimension_info.pack(side=tk.LEFT, padx=5)
        self.entry_x1.bind("<FocusOut>", self.update_visuals_from_entries); self.entry_y1.bind("<FocusOut>", self.update_visuals_from_entries)
        self.entry_x2.bind("<FocusOut>", self.update_visuals_from_entries); self.entry_y2.bind("<FocusOut>", self.update_visuals_from_entries)

        # --- Left Frame: Video Preview & Timeline ---
        ttk.Label(self.video_preview_frame, text="Video Preview & Spatial Crop", anchor="center").pack(pady=(0,5))
        self.canvas_video = tk.Canvas(self.video_preview_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="grey", relief=tk.SUNKEN, borderwidth=1); self.canvas_video.pack(pady=5, padx=5, fill="both", expand=True)
        self.canvas_video.bind("<Button-1>", self.on_canvas_press); self.canvas_video.bind("<B1-Motion>", self.on_canvas_drag); self.canvas_video.bind("<ButtonRelease-1>", self.on_canvas_release)
        # Timeline
        self.frame_timeline = ttk.LabelFrame(self.video_preview_frame, text="Timeline & Temporal Crop"); self.frame_timeline.pack(pady=5, fill="x")
        self.time_slider_var = tk.DoubleVar(); self.time_slider = ttk.Scale(self.frame_timeline, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.time_slider_var, command=self.on_slider_move); self.time_slider.pack(fill="x", padx=5, pady=2); self.time_slider.config(state=tk.DISABLED); self.time_slider.bind("<ButtonRelease-1>", self.on_slider_release)
        timeline_btns_frame = ttk.Frame(self.frame_timeline); timeline_btns_frame.pack(fill="x", padx=5)
        self.btn_set_start = ttk.Button(timeline_btns_frame, text="Set Start", width=8, command=self.set_start_time_from_slider, state=tk.DISABLED); self.btn_set_start.pack(side=tk.LEFT, padx=2)
        self.btn_set_end = ttk.Button(timeline_btns_frame, text="Set End", width=8, command=self.set_end_time_from_slider, state=tk.DISABLED); self.btn_set_end.pack(side=tk.LEFT, padx=2)
        self.lbl_current_time = ttk.Label(timeline_btns_frame, text="Time: 0.00s"); self.lbl_current_time.pack(side=tk.LEFT, padx=10)

        # --- Right Frame: Calibration Image Preview ---
        ttk.Label(self.calib_preview_frame, text="Calibration Image Preview", anchor="center").pack(pady=(0,5))
        self.canvas_calib = tk.Canvas(self.calib_preview_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="dark slate gray", relief=tk.SUNKEN, borderwidth=1); self.canvas_calib.pack(pady=5, padx=5, fill="both", expand=True)
        self.canvas_calib.bind("<Configure>", self._redraw_calib_canvas)
        self.canvas_video.bind("<Configure>", self._redraw_video_canvas)

        # --- Bottom: Processing & Status ---
        self.frame_process = ttk.LabelFrame(self.controls_frame, text="3. Process and Save"); self.frame_process.pack(pady=5, fill="x")
        self.btn_process = ttk.Button(self.frame_process, text="Process & Save Files", command=self.start_processing_thread, state=tk.DISABLED); self.btn_process.pack(pady=5, padx=5)
        self.status_var = tk.StringVar(); self.status_var.set("Ready. Load a video.")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W); self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=100, mode="determinate")

    # --- Image Display Helper ---
    def _display_image(self, canvas, pil_image, tk_photo_ref_attr, display_params_key):
        """Displays PIL image, stores Tkinter ref, updates scaling/offset."""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # Store current scaling/offset before potentially clearing
        params = self.display_params[display_params_key]

        if pil_image is None or canvas_width <= 1 or canvas_height <= 1:
            canvas.delete("all")
            setattr(self, tk_photo_ref_attr, None)
            params['scale_x'] = 1.0; params['scale_y'] = 1.0 # Reset scaling
            params['offset_x'] = 0; params['offset_y'] = 0
            # Also delete the mirrored crop box if clearing the calib canvas
            if display_params_key == 'calib' and self.crop_rect_id_calib:
                 self.canvas_calib.delete(self.crop_rect_id_calib)
                 self.crop_rect_id_calib = None
            return None

        # Calculate new scaling/offset to fit this canvas
        img_w, img_h = pil_image.size
        aspect_ratio = img_w / img_h if img_h > 0 else 1
        target_aspect = canvas_width / canvas_height if canvas_height > 0 else 1
        new_w, new_h = (canvas_width, int(canvas_width/aspect_ratio)) if aspect_ratio > target_aspect else (int(canvas_height*aspect_ratio), canvas_height)
        new_w, new_h = max(1, new_w), max(1, new_h) # Ensure dimensions > 0

        # Update stored scaling/offset for this specific canvas
        params['scale_x'] = new_w / img_w if img_w > 0 else 1.0
        params['scale_y'] = new_h / img_h if img_h > 0 else 1.0
        params['offset_x'] = (canvas_width - new_w) // 2
        params['offset_y'] = (canvas_height - new_h) // 2

        try:
            resized_img = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            tk_photo = ImageTk.PhotoImage(resized_img)
            setattr(self, tk_photo_ref_attr, tk_photo) # Keep reference
            canvas.delete("all")
            img_item = canvas.create_image(params['offset_x'], params['offset_y'], anchor=tk.NW, image=tk_photo)
            # Redraw the appropriate crop box after image is displayed
            self.update_crop_box_on_canvas() # This now redraws both if needed
            return img_item
        except Exception as e:
            print(f"Error displaying image: {e}"); setattr(self, tk_photo_ref_attr, None)
            params['scale_x']=1.0; params['scale_y']=1.0; params['offset_x']=0; params['offset_y']=0
            canvas.delete("all"); canvas.create_text(canvas_width/2, canvas_height/2, text="Display Error", fill="red"); return None

    # --- Canvas Redraw Callbacks ---
    def _redraw_video_canvas(self, event=None):
        if self.cap: video_pil = self._get_current_video_pil()
        else: video_pil = None
        self._display_image(self.canvas_video, video_pil, "photo_image_video", 'video')
        # update_crop_box_on_canvas is called within _display_image now

    def _redraw_calib_canvas(self, event=None):
        self._display_image(self.canvas_calib, self.calib_image_pil, "photo_image_calib", 'calib')
        # update_crop_box_on_canvas is called within _display_image now


    # --- Coordinate Conversion (Specify Target Canvas) ---
    def _video_coords_to_canvas_coords(self, vx, vy, target_canvas_key):
        """Converts original video coordinates to specified canvas coordinates."""
        params = self.display_params[target_canvas_key]
        cx = int(vx * params['scale_x'] + params['offset_x'])
        cy = int(vy * params['scale_y'] + params['offset_y'])
        return cx, cy

    def _canvas_coords_to_video_coords(self, cx, cy, source_canvas_key):
        """Converts specified canvas coordinates to original video coordinates."""
        params = self.display_params[source_canvas_key]
        scale_x, scale_y = params['scale_x'], params['scale_y']
        if scale_x == 0 or scale_y == 0: return 0, 0 # Avoid division by zero
        vx = (cx - params['offset_x']) / scale_x
        vy = (cy - params['offset_y']) / scale_y
        # Clamp to original video dimensions (which should match adjusted calib img dims)
        vx = max(0, min(self.original_width if hasattr(self, 'original_width') else 0, vx))
        vy = max(0, min(self.original_height if hasattr(self, 'original_height') else 0, vy))
        return int(round(vx)), int(round(vy))

    # --- File Loading ---
    def load_video(self):
        filepath = filedialog.askopenfilename(title="Select Video File", filetypes=(("Video Files", "*.mp4 *.avi *.mov"), ("All files", "*.*")))
        if not filepath: return
        self.input_filepath = filepath
        if self.cap: self.cap.release()
        try:
            self.cap = cv2.VideoCapture(self.input_filepath)
            if not self.cap.isOpened(): raise IOError(f"Cannot open: {self.input_filepath}")
            # Store video properties in the instance attribute
            self.video_properties['fps'] = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_properties['total_frames'] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_properties['width'] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_properties['height'] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_properties['duration'] = self.video_properties['total_frames'] / self.video_properties['fps'] if self.video_properties['fps'] > 0 else 0.0
            # Also store directly for easier access in some places
            self.original_width = self.video_properties['width']
            self.original_height = self.video_properties['height']
            self.original_duration = self.video_properties['duration']
            self.fps = self.video_properties['fps']
            self.total_frames = self.video_properties['total_frames']

            if self.original_duration <= 0 or self.original_width <= 0: raise ValueError("Video props invalid.")
            self.lbl_filename_video.config(text=os.path.basename(filepath))
            self.lbl_duration_info.config(text=f"Dur: {self.original_duration:.2f}s"); self.lbl_dimension_info.config(text=f"Dims: {self.original_width}x{self.original_height}")
            # Reset entries
            self.entry_start.delete(0, tk.END); self.entry_start.insert(0, "0"); self.entry_end.delete(0, tk.END); self.entry_end.insert(0, f"{self.original_duration:.2f}")
            self.entry_x1.delete(0, tk.END); self.entry_x1.insert(0, "0"); self.entry_y1.delete(0, tk.END); self.entry_y1.insert(0, "0")
            self.entry_x2.delete(0, tk.END); self.entry_x2.insert(0, f"{self.original_width}"); self.entry_y2.delete(0, tk.END); self.entry_y2.insert(0, f"{self.original_height}")
            self.time_slider.config(to=self.original_duration, state=tk.NORMAL); self.time_slider_var.set(0)
            self.show_frame_at_time(0) # Displays frame and calls update_crop_box
            self.btn_process.config(state=tk.NORMAL); self.btn_set_start.config(state=tk.NORMAL); self.btn_set_end.config(state=tk.NORMAL)
            self.status_var.set(f"Video Loaded: {os.path.basename(filepath)}")
            # Auto-process calib image if already loaded
            if self.calib_image_path:
                 self.process_and_display_calib_image(self.calib_image_path)
        except Exception as e: messagebox.showerror("Error Loading Video", f"Failed: {e}"); self.reset_ui()

    def load_calib_image(self):
        filepath = filedialog.askopenfilename(title="Select Calibration Image File", filetypes=(("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All files", "*.*")))
        if not filepath: return
        self.calib_image_path = filepath
        self.process_and_display_calib_image(filepath)

    def process_and_display_calib_image(self, filepath):
        # Uses self.original_width/_height which are set in load_video
        if self.original_width == 0 or self.original_height == 0:
            messagebox.showinfo("Load Video First", "Please load the video before loading/processing the calibration image.")
            self.calib_image_path = None; self.lbl_filename_calib.config(text="No calibration image loaded")
            self._display_image(self.canvas_calib, None, "photo_image_calib", 'calib'); return
        try:
            img = Image.open(filepath); img_calib_orig = img.copy(); img.close()
            calib_w, calib_h = img_calib_orig.size; vid_w, vid_h = self.original_width, self.original_height
            calib_aspect = calib_w/calib_h if calib_h>0 else 0; vid_aspect = vid_w/vid_h if vid_h>0 else 0
            processed_img = img_calib_orig
            # Aspect Ratio Crop
            if abs(calib_aspect - vid_aspect) > ASPECT_RATIO_TOLERANCE:
                print(f"INFO: Adjusting calib aspect {calib_aspect:.3f} to video {vid_aspect:.3f}.")
                self.status_var.set("Adjusting calib aspect..."); self.root.update_idletasks()
                if calib_aspect > vid_aspect: # Crop sides
                    target_w = int(round(calib_h * vid_aspect)); crop_margin = (calib_w - target_w) / 2
                    box = (max(0, math.floor(crop_margin)), 0, min(calib_w, math.ceil(calib_w - crop_margin)), calib_h)
                else: # Crop top/bottom
                    target_h = int(round(calib_w / vid_aspect)); crop_margin = (calib_h - target_h) / 2
                    box = (0, max(0, math.floor(crop_margin)), calib_w, min(calib_h, math.ceil(calib_h - crop_margin)))
                if box[0] < box[2] and box[1] < box[3]: processed_img = processed_img.crop(box); calib_w, calib_h = processed_img.size
                else: print(f"WARN: Invalid crop box {box}. Skipping aspect crop.")
            # Resize
            if calib_w != vid_w or calib_h != vid_h:
                print(f"INFO: Resizing calib {calib_w}x{calib_h} to video {vid_w}x{vid_h}.")
                self.status_var.set("Resizing calib image..."); self.root.update_idletasks()
                processed_img = processed_img.resize((vid_w, vid_h), Image.Resampling.LANCZOS)
            self.calib_image_pil = processed_img; self.lbl_filename_calib.config(text=os.path.basename(filepath))
            self._display_image(self.canvas_calib, self.calib_image_pil, "photo_image_calib", 'calib')
            self.status_var.set("Files Loaded: Video & Processed Calib Image.")
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not process/load calib image:\n{e}")
            self.calib_image_pil = None; self.lbl_filename_calib.config(text=f"Error processing {os.path.basename(filepath)}")
            self._display_image(self.canvas_calib, None, "photo_image_calib", 'calib')

    def reset_ui(self):
        if self.cap: self.cap.release(); self.cap = None
        self.input_filepath = None; self.calib_image_path = None; self.calib_image_pil = None
        self.video_properties = {}; self.original_duration=0.0; self.original_width=0; self.original_height=0; self.fps=0.0; self.total_frames=0
        self.lbl_filename_video.config(text="No video selected"); self.lbl_filename_calib.config(text="No calib loaded")
        self.lbl_duration_info.config(text="Duration: -"); self.lbl_dimension_info.config(text="Dims: -")
        for entry in [self.entry_start, self.entry_end, self.entry_x1, self.entry_y1, self.entry_x2, self.entry_y2]: entry.delete(0,tk.END)
        self.entry_start.insert(0,"0")
        self.time_slider.config(state=tk.DISABLED); self.time_slider_var.set(0)
        self.btn_process.config(state=tk.DISABLED); self.btn_set_start.config(state=tk.DISABLED); self.btn_set_end.config(state=tk.DISABLED)
        self._display_image(self.canvas_video, None, "photo_image_video", 'video')
        self._display_image(self.canvas_calib, None, "photo_image_calib", 'calib') # Clears calib canvas too
        # Explicitly delete crop box IDs
        if self.crop_rect_id_video: self.canvas_video.delete(self.crop_rect_id_video); self.crop_rect_id_video = None
        if self.crop_rect_id_calib: self.canvas_calib.delete(self.crop_rect_id_calib); self.crop_rect_id_calib = None
        self.crop_handle_ids = {}
        self.status_var.set("Ready")

    def _get_current_video_pil(self):
         if not self.cap or not self.cap.isOpened(): return None
         self.cap.set(cv2.CAP_PROP_POS_MSEC, int(self.current_frame_display_time * 1000))
         ret, frame = self.cap.read()
         return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else None

    def show_frame_at_time(self, time_sec):
        """Reads video frame, displays it, updates scaling, draws crop boxes."""
        if not self.cap or not self.cap.isOpened() or self.original_width <= 0: return
        self.current_frame_display_time = time_sec; self.lbl_current_time.config(text=f"Time: {time_sec:.2f}s")
        video_pil = self._get_current_video_pil()
        # Display image and implicitly update crop box via _display_image's call to update_crop_box_on_canvas
        self._display_image(self.canvas_video, video_pil, "photo_image_video", 'video')


    # --- Timeline Interaction (No change) ---
    def on_slider_move(self, v): self.lbl_current_time.config(text=f"Time: {float(v):.2f}s")
    def on_slider_release(self, e): self.show_frame_at_time(self.time_slider_var.get())
    def set_start_time_from_slider(self): t=f"{self.time_slider_var.get():.2f}";self.entry_start.delete(0,tk.END);self.entry_start.insert(0,t)
    def set_end_time_from_slider(self): t=f"{self.time_slider_var.get():.2f}";self.entry_end.delete(0,tk.END);self.entry_end.insert(0,t)

    # --- Spatial Crop Box Interaction ---
    def update_crop_box_on_canvas(self):
        """Draws crop box on video canvas and mirrored box on calib canvas."""
        if not self.cap or self.original_width <= 0: return

        # Get video pixel coordinates from entries
        try:
            vx1,vy1,vx2,vy2 = map(lambda e:int(float(e.get())),(self.entry_x1,self.entry_y1,self.entry_x2,self.entry_y2))
            vx1=max(0,min(self.original_width,vx1));vy1=max(0,min(self.original_height,vy1));vx2=max(vx1,min(self.original_width,vx2));vy2=max(vy1,min(self.original_height,vy2))
        except ValueError: # Default to full frame on error
            vx1,vy1,vx2,vy2 = 0,0,self.original_width,self.original_height

        # --- Draw on Video Canvas ---
        canvas_vid = self.canvas_video
        if self.crop_rect_id_video: canvas_vid.delete(self.crop_rect_id_video)
        for hid in self.crop_handle_ids.values(): canvas_vid.delete(hid)
        self.crop_handle_ids.clear()
        # Convert video pixel coords to VIDEO canvas coords
        cx1_v, cy1_v = self._video_coords_to_canvas_coords(vx1, vy1, 'video')
        cx2_v, cy2_v = self._video_coords_to_canvas_coords(vx2, vy2, 'video')
        self.crop_rect_id_video = canvas_vid.create_rectangle(cx1_v, cy1_v, cx2_v, cy2_v, outline=VIDEO_CROP_COLOR, width=2, tags="crop_box")
        # Draw handles on video canvas
        h_coords = {'nw':(cx1_v,cy1_v),'ne':(cx2_v,cy1_v),'sw':(cx1_v,cy2_v),'se':(cx2_v,cy2_v),'n':((cx1_v+cx2_v)//2,cy1_v),'s':((cx1_v+cx2_v)//2,cy2_v),'w':(cx1_v,(cy1_v+cy2_v)//2),'e':(cx2_v,(cy1_v+cy2_v)//2)}
        for k,(hx,hy) in h_coords.items(): self.crop_handle_ids[k] = canvas_vid.create_rectangle(hx-HANDLE_SIZE,hy-HANDLE_SIZE,hx+HANDLE_SIZE,hy+HANDLE_SIZE,fill="white",outline="black",tags=("crop_handle",k))

        # --- Draw Mirrored Box on Calibration Canvas ---
        if self.calib_image_pil: # Only draw if calib image is loaded
            canvas_cal = self.canvas_calib
            if self.crop_rect_id_calib: canvas_cal.delete(self.crop_rect_id_calib)
             # Convert the SAME video pixel coords to CALIB canvas coords
            cx1_c, cy1_c = self._video_coords_to_canvas_coords(vx1, vy1, 'calib')
            cx2_c, cy2_c = self._video_coords_to_canvas_coords(vx2, vy2, 'calib')
            self.crop_rect_id_calib = canvas_cal.create_rectangle(cx1_c, cy1_c, cx2_c, cy2_c, outline=CALIB_CROP_COLOR, width=2, dash=(4, 2)) # Dashed line

    def get_handle_at_pos(self, x, y): # Only check on video canvas
        for k,hid in self.crop_handle_ids.items():
            coords=self.canvas_video.coords(hid);
            if coords and coords[0]<=x<=coords[2] and coords[1]<=y<=coords[3]: return k
        return None

    def on_canvas_press(self, e): # Only trigger on video canvas
        if e.widget != self.canvas_video or not self.cap: return
        self.dragging_handle = self.get_handle_at_pos(e.x, e.y)
        self.drag_start_pos = (e.x, e.y) if self.dragging_handle else None

    def on_canvas_drag(self, e): # Only trigger on video canvas
        if e.widget != self.canvas_video or not self.dragging_handle or not self.drag_start_pos: return
        dx,dy = e.x-self.drag_start_pos[0], e.y-self.drag_start_pos[1]
        params_vid = self.display_params['video']
        scale_x, scale_y = params_vid['scale_x'], params_vid['scale_y']
        dvx = dx / scale_x if scale_x else 0; dvy = dy / scale_y if scale_y else 0
        h = self.dragging_handle
        try: vx1,vy1,vx2,vy2 = map(lambda el: int(float(el.get())), (self.entry_x1,self.entry_y1,self.entry_x2,self.entry_y2))
        except ValueError: return
        nx1,ny1,nx2,ny2 = vx1,vy1,vx2,vy2
        if 'n' in h: ny1 = max(0, min(vy2-1, ny1+dvy))
        if 's' in h: ny2 = max(vy1+1, min(self.original_height, ny2+dvy))
        if 'w' in h: nx1 = max(0, min(vx2-1, nx1+dvx))
        if 'e' in h: nx2 = max(vx1+1, min(self.original_width, nx2+dvx))
        self.update_entries(int(round(nx1)),int(round(ny1)),int(round(nx2)),int(round(ny2)))
        self.update_crop_box_on_canvas() # Update both boxes
        self.drag_start_pos=(e.x,e.y)

    def on_canvas_release(self, e): # Only trigger on video canvas
        if e.widget != self.canvas_video or not self.dragging_handle: return
        # Final update based on video canvas coords
        self.update_entries_from_visuals() # This calls canvas_to_video for video canvas
        self.dragging_handle = None; self.drag_start_pos = None

    # --- Linking Entries and Visuals ---
    def update_entries(self,vx1,vy1,vx2,vy2): # Updates entries, triggers visuals update via trace
        self.entry_x1.delete(0,tk.END);self.entry_x1.insert(0,str(vx1)); self.entry_y1.delete(0,tk.END);self.entry_y1.insert(0,str(vy1))
        self.entry_x2.delete(0,tk.END);self.entry_x2.insert(0,str(vx2)); self.entry_y2.delete(0,tk.END);self.entry_y2.insert(0,str(vy2))

    def update_entries_from_visuals(self): # Uses video canvas coords to update entries
        if not self.crop_rect_id_video: return
        c = self.canvas_video.coords(self.crop_rect_id_video)
        if not c: return
        vx1,vy1 = self._canvas_coords_to_video_coords(c[0], c[1], 'video')
        vx2,vy2 = self._canvas_coords_to_video_coords(c[2], c[3], 'video')
        self.update_entries(vx1, vy1, vx2, vy2)

    def update_visuals_from_entries(self, e=None): # Triggered by entry change
        if not self.cap: return
        try:
            st = float(self.entry_start.get())
            if abs(self.time_slider_var.get()-st)>0.01:
                self.time_slider_var.set(max(0,min(self.original_duration,st)))
                self.show_frame_at_time(self.time_slider_var.get()) # Updates frame & crop box
            else:
                self.update_crop_box_on_canvas() # Just update crop box if time is same
        except ValueError: self.status_var.set("Warn: Invalid number.")

    # --- Processing ---
    # start_processing_thread same as v4
    def start_processing_thread(self):
        if not self.input_filepath: messagebox.showwarning("No Video", "Load video first."); return
        if self.processing_thread and self.processing_thread.is_alive(): messagebox.showwarning("Busy", "Processing active."); return
        try: # Get crop params
            start_time, end_time = float(self.entry_start.get()), float(self.entry_end.get())
            x1, y1, x2, y2 = map(lambda e: int(float(e.get())), [self.entry_x1, self.entry_y1, self.entry_x2, self.entry_y2])
            if not (0 <= start_time < end_time <= self.original_duration + 0.01): raise ValueError("Invalid time.")
            if not (0 <= x1 < x2 <= self.original_width and 0 <= y1 < y2 <= self.original_height): raise ValueError("Invalid crop coords.")
        except ValueError as e: messagebox.showerror("Invalid Input", f"Check params:\n{e}"); return
        video_output_filepath = filedialog.asksaveasfilename(title="Save Cropped Video As...", defaultextension=".mp4", filetypes=(("MP4", "*.mp4"), ("All", "*.*")), initialfile=f"{os.path.splitext(os.path.basename(self.input_filepath))[0]}_cropped.mp4")
        if not video_output_filepath: self.status_var.set("Save cancelled"); return
        self.output_filepath = video_output_filepath
        calib_output_filepath = None
        if self.calib_image_pil: # Check if processed calib image exists
            calib_initial_name = f"{os.path.splitext(os.path.basename(self.calib_image_path if self.calib_image_path else 'calibration'))[0]}_adj_cropped.png"
            calib_output_filepath = filedialog.asksaveasfilename(title="Save Adjusted & Cropped Calibration Image As...", defaultextension=".png", filetypes=(("PNG", "*.png"), ("BMP", "*.bmp"), ("TIFF", "*.tif"),("All", "*.*")), initialfile=calib_initial_name)
            if not calib_output_filepath:
                if not messagebox.askyesno("Cancel?", "Calib image save cancelled. Continue only with video?"): self.status_var.set("Save cancelled"); return
        self.output_calib_image_path = calib_output_filepath
        self.btn_process.config(state=tk.DISABLED); self.btn_load_video.config(state=tk.DISABLED); self.btn_load_calib.config(state=tk.DISABLED); self.time_slider.config(state=tk.DISABLED)
        self.status_var.set("Starting processing..."); self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5); self.progress_bar["value"] = 0; self.root.update_idletasks()
        self.processing_thread = threading.Thread(target=self.process_video_thread, args=(start_time, end_time, x1, y1, x2, y2), daemon=True); self.processing_thread.start()

    # update_progress same as v4
    def update_progress(self, value):
        if self.progress_bar.winfo_exists(): self.progress_bar["value"] = value; self.root.update_idletasks()

    # process_video_thread same as v4
    def process_video_thread(self, start_time, end_time, x1, y1, x2, y2):
        clip=None; edited_clip=None; success_video=False; success_calib=False; final_message="Processing failed"
        try:
            calib_save_msg = ""
            if self.calib_image_pil and self.output_calib_image_path:
                self.root.after(0, lambda: self.status_var.set("Processing: Cropping adjusted calib image..."))
                try:
                    cropped_calib_img = self.calib_image_pil.crop((x1, y1, x2, y2))
                    cropped_calib_img.save(self.output_calib_image_path); success_calib = True
                    calib_save_msg = f"Cropped calib saved: {os.path.basename(self.output_calib_image_path)}"
                    self.root.after(0, lambda m=calib_save_msg: self.status_var.set(f"Processing: {m}")); print(f"INFO: {calib_save_msg}")
                except Exception as e: calib_save_msg = f"Error final crop/save calib: {e}"; print(f"ERROR: {calib_save_msg}"); self.root.after(0, lambda m=calib_save_msg: messagebox.showerror("Calib Error", m))
            self.root.after(0, lambda: self.status_var.set("Processing: Video...")); clip = mp.VideoFileClip(self.input_filepath); self.root.after(0, lambda: self.update_progress(10))
            actual_dur=clip.duration; safe_end=min(end_time,actual_dur); safe_start=min(start_time,safe_end)
            if safe_start>=safe_end-0.001:
                if safe_end>0.001: safe_start=safe_end-0.001
                else: raise ValueError("Clip duration zero/negative.")
            edited_clip = clip.subclip(safe_start, safe_end); self.root.after(0, lambda: self.update_progress(30))
            spatial_crop = not (x1==0 and y1==0 and x2==self.original_width and y2==self.original_height)
            if spatial_crop: edited_clip = edited_clip.crop(x1=x1, y1=y1, x2=x2, y2=y2); self.root.after(0, lambda: self.update_progress(50))
            else: self.root.after(0, lambda: self.update_progress(50))
            self.root.after(0, lambda: self.status_var.set("Processing: Writing video...")); print("\n--- MoviePy Write ---")
            edited_clip.write_videofile(self.output_filepath, codec="libx264", audio_codec="aac", temp_audiofile='temp-audio.m4a', remove_temp=True, logger="bar")
            print("--- Write Finished ---\n"); success_video = True; self.root.after(0, lambda: self.update_progress(100))
            if success_video and success_calib: final_message = f"Video saved: {os.path.basename(self.output_filepath)}. {calib_save_msg}."
            elif success_video: final_message = f"Video saved: {os.path.basename(self.output_filepath)}. (Calib not saved)."
            elif success_calib: final_message = f"Video failed. {calib_save_msg}."
            else: final_message = "Failed: Video & Calib."
            self.processing_finished(success=(success_video or success_calib), message=final_message)
        except Exception as e:
            import traceback; print("\n--- ERROR PROCESSING ---"); print(traceback.format_exc()); print("---"); error_message = f"Error processing: {e}"
            self.root.after(0, lambda msg=error_message: messagebox.showerror("Error", msg))
            if success_calib: final_message = f"Video failed. {calib_save_msg}."
            else: final_message = f"Video failed. {calib_save_msg}" if calib_save_msg else "Video processing failed."
            self.processing_finished(success=success_calib, message=final_message)
        finally:
            try:
                if clip: clip.close()
                if edited_clip: edited_clip.close()
            except Exception: pass

    # processing_finished same as v4
    def processing_finished(self, success, message):
            self.status_var.set(message); is_input_valid = bool(self.input_filepath and self.original_width > 0)
            self.btn_process.config(state=tk.NORMAL if is_input_valid else tk.DISABLED)
            self.btn_load_video.config(state=tk.NORMAL); self.btn_load_calib.config(state=tk.NORMAL)
            self.time_slider.config(state=tk.NORMAL if is_input_valid else tk.DISABLED)
            if self.progress_bar.winfo_ismapped(): self.progress_bar.pack_forget()
            self.progress_bar["value"] = 0

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEditorApp(root)
    root.mainloop()
